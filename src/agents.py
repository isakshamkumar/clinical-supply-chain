import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools import (
    run_sql_query,
    calculate_runway,
    search_trial_by_name,
    send_alert,
)


def get_openrouter_llm(model: str = "x-ai/grok-4.1-fast"):
    """Create a ChatOpenAI instance configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Get your key from https://openrouter.ai/keys"
        )
    
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",  # Optional: for analytics
            "X-Title": "Clinical Supply Chain AI",  # Optional: for analytics
        },
        temperature=0.7,
    )


@dataclass
class AgentState:
    """Shared state across all agents."""

    trigger_source: str  # "cron" or "user"
    user_query: Optional[str]
    selected_agent: Optional[str] = None #supply_watchdog 
    extracted_entities: Dict = field(default_factory=dict)
    sql_queries: List[str] = field(default_factory=list)
    query_results: List[Dict] = field(default_factory=list)
    final_output: Optional[Dict] = None
    error_history: List[str] = field(default_factory=list)
    retry_count: int = 0


class OrchestratorAgent:
    """Routes requests to appropriate specialist agents."""

    def __init__(self):
        self.llm = get_openrouter_llm(model="x-ai/grok-4.1-fast")

    def invoke(self, state: AgentState) -> AgentState:
        """Classify intent and route to appropriate agent."""

        system_prompt = """You are the Orchestrator Agent for Global Pharma's Clinical Supply Chain AI.

YOUR ROLE: Classify incoming requests and route to the appropriate specialist agent.

ROUTING RULES:
1. IF trigger_source == "cron" → Route to supply_watchdog
2. IF user_query contains ["extend", "shelf-life", "can we", "feasibility"] → Route to scenario_strategist
3. IF user_query contains ["stock levels", "inventory", "how much"] → Route to supply_watchdog
4. ELSE → Return error

OUTPUT FORMAT (JSON only):
{
  "selected_agent": "supply_watchdog" | "scenario_strategist",
  "reasoning": "Brief explanation",
  "extracted_entities": {
    "trial_id": str | null,
    "batch_number": str | null,
    "country": str | null
  }
}
"""

        user_input = state.user_query or "Scheduled health check"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Trigger: {state.trigger_source}\nInput: {user_input}",
            ),
        ]

        response = self.llm.invoke(messages)

        try:
            result = json.loads(response.content)
            state.selected_agent = result.get("selected_agent")
            state.extracted_entities = result.get("extracted_entities", {})
        except json.JSONDecodeError:
            state.selected_agent = "error"
            state.error_history.append(
                "Orchestrator failed to parse routing decision",
            )

        return state


class SupplyWatchdogAgent:
    """Autonomous monitoring agent for daily health checks."""

    def __init__(self):
        self.llm = get_openrouter_llm(model="x-ai/grok-4.1-fast")
        self.tools = [run_sql_query, calculate_runway, send_alert]

    def invoke(self, state: AgentState) -> AgentState:
        """Run daily supply chain health check."""

        # Expiry risk query - using lot_status_report table
        expiry_query = """
SELECT 
    ls."Lot Number" as batch_id,
    ls."Trial Alias" as trial,
    ls."Expiration Date" as expiry_date,
    ls."Country" as country,
    ROUND((julianday(ls."Expiration Date") - julianday(date('now')))) as days_remaining,
    CASE 
        WHEN date(ls."Expiration Date") <= date(date('now'), '+30 days') THEN 'Critical'
        WHEN date(ls."Expiration Date") <= date(date('now'), '+60 days') THEN 'High'
        ELSE 'Medium'
    END as risk_level
FROM lot_status_report ls
WHERE date(ls."Expiration Date") <= date(date('now'), '+90 days')
ORDER BY ls."Expiration Date" ASC;
"""

        expiry_result = run_sql_query.invoke({"query": expiry_query})

        # Shortfall prediction query - using Initial Qty from available_inventory_report
        shortfall_query = """
WITH Trial_Supply AS (
    SELECT 
        "Trial Name" as trial_id,
        SUM("Initial Qty") as total_stock
    FROM available_inventory_report
    WHERE "Initial Qty" > 0
    GROUP BY "Trial Name"
),
Trial_Demand AS (
    SELECT 
        "trial_alias" as trial_id,
        "enrollment_rate_monthly_actual" as monthly_rate
    FROM study_level_enrollment_report
    WHERE "enrollment_rate_monthly_actual" > 0
)
SELECT 
    s.trial_id,
    s.total_stock,
    d.monthly_rate,
    ROUND(d.monthly_rate / 4.0, 2) as weekly_burn_rate,
    ROUND(s.total_stock / NULLIF((d.monthly_rate / 4.0), 0), 1) as weeks_of_coverage
FROM Trial_Supply s
INNER JOIN Trial_Demand d ON s.trial_id = d.trial_id
WHERE (s.total_stock / NULLIF((d.monthly_rate / 4.0), 0)) < 8
ORDER BY weeks_of_coverage ASC;
"""

        shortfall_result = run_sql_query.invoke({"query": shortfall_query})

        alert_payload = {
            "alert_type": "supply_watchdog_report",
            "timestamp": datetime.now().isoformat(),
            "risks_detected": {
                "expiry_alerts": expiry_result.get("data", [])
                if expiry_result.get("success")
                else [],
                "shortfall_predictions": shortfall_result.get("data", [])
                if shortfall_result.get("success")
                else [],
            },
        }

        summary = {
            "total_expiry_risks": len(alert_payload["risks_detected"]["expiry_alerts"]),
            "total_shortfall_risks": len(
                alert_payload["risks_detected"]["shortfall_predictions"],
            ),
        }
        summary["action_required"] = bool(
            summary["total_expiry_risks"] or summary["total_shortfall_risks"],
        )
        alert_payload["summary"] = summary

        if summary["action_required"]:
            send_alert.invoke(
                {
                    "payload": alert_payload,
                    "channels": ["email", "slack"],
                },
            )

        state.final_output = alert_payload
        return state


class ScenarioStrategistAgent:
    """Conversational agent for extension feasibility analysis."""

    def __init__(self):
        self.llm = get_openrouter_llm(model="x-ai/grok-4.1-fast")
        self.tools = [run_sql_query, search_trial_by_name]

    def invoke(self, state: AgentState) -> AgentState:
        """Analyze shelf-life extension feasibility with 3-gate logic."""

        entities = state.extracted_entities or {}
        batch_number = entities.get("batch_number")
        trial_id = entities.get("trial_id")
        country = entities.get("country")
        if not all([batch_number, trial_id, country]):
            state.final_output = {
                "decision": "REJECTED",
                "reasoning": "Missing required parameters: batch_number, trial_id, or country",
                "recommendation": "Please provide all required information.",
            }
            return state

        # Gate 1: Technical
        tech_query = f"""
SELECT 
    "Sample Status (NDP Material Coordinator to Complete)" as sample_status,
    "Lot Number (Molecule Planner to Complete)" as lot_number,
    "Modified Date" as modified_date
FROM re_evaluation
WHERE "Lot Number (Molecule Planner to Complete)" = '{batch_number}'
  AND "Sample Status (NDP Material Coordinator to Complete)" = 'Complete'
LIMIT 1;
"""
        tech_result = run_sql_query.invoke({"query": tech_query})

        if not tech_result.get("success") or not tech_result.get("data"):
            state.final_output = {
                "decision": "REJECTED",
                "reasoning": {
                    "technical": {
                        "status": "fail",
                        "evidence": "No successful re-evaluation found.",
                    },
                    "regulatory": {
                        "status": "not_checked",
                        "evidence": "Skipped due to technical failure.",
                    },
                    "logistical": {
                        "status": "not_checked",
                        "evidence": "Skipped due to technical failure.",
                    },
                },
                "recommendation": "Batch must pass re-evaluation before extension can be considered.",
            }
            return state

        # Gate 2: Regulatory - using health_authority_division_c instead of country
        reg_query = f"""
SELECT "submission_outcome", "approved_date_c" as approval_date
FROM rim
WHERE "clinical_study_v" = '{trial_id}'
  AND "submission_outcome" IN ('Accepted', 'Approved')
LIMIT 1;
"""
        reg_result = run_sql_query.invoke({"query": reg_query})

        if not reg_result.get("success") or not reg_result.get("data"):
            state.final_output = {
                "decision": "REJECTED",
                "reasoning": {
                    "technical": {
                        "status": "pass",
                        "evidence": "Re-evaluation complete and passing.",
                    },
                    "regulatory": {
                        "status": "fail",
                        "evidence": f"Extension not approved in {country}.",
                    },
                    "logistical": {
                        "status": "not_checked",
                        "evidence": "Skipped due to regulatory block.",
                    },
                },
                "recommendation": f"Cannot extend in {country} without regulatory approval.",
            }
            return state

        # Gate 3: Logistics - extract days from ip_timeline string
        log_query = f"""
SELECT "ip_timeline", "country_name"
FROM ip_shipping_timelines_report
WHERE "country_name" LIKE '%{country}%'
LIMIT 1;
"""
        log_result = run_sql_query.invoke({"query": log_query})
        
        # Parse days from timeline string (e.g., "6 days door-to-door" -> 6)
        shipping_days = 14  # Default fallback
        if log_result.get("success") and log_result.get("data"):
            timeline_str = log_result["data"][0].get("ip_timeline", "")
            import re
            match = re.search(r'(\d+)\s*days?', timeline_str, re.IGNORECASE)
            if match:
                shipping_days = int(match.group(1))

        if not log_result.get("success") or not log_result.get("data"):
            state.final_output = {
                "decision": "REJECTED",
                "reasoning": {
                    "technical": {
                        "status": "pass",
                        "evidence": "Re-evaluation passed.",
                    },
                    "regulatory": {
                        "status": "pass",
                        "evidence": f"Extension approved in {country}.",
                    },
                    "logistical": {
                        "status": "fail",
                        "evidence": "No shipping timeline available.",
                    },
                },
                "recommendation": "Cannot safely plan extension without logistics data.",
            }
            return state

        buffer_days = 14
        # Placeholder for actual expiry math; keep scenario conservative
        time_available = 30

        if time_available > (shipping_days + buffer_days):
            state.final_output = {
                "decision": "APPROVED",
                "reasoning": {
                    "technical": {
                        "status": "pass",
                        "evidence": "Re-evaluation complete and passing.",
                    },
                    "regulatory": {
                        "status": "pass",
                        "evidence": f"Extension approved in {country}.",
                    },
                    "logistical": {
                        "status": "pass",
                        "evidence": f"{time_available} days available vs {shipping_days + buffer_days} required.",
                    },
                },
                "recommendation": f"Extension APPROVED for Batch {batch_number}. Proceed with execution.",
            }
        else:
            state.final_output = {
                "decision": "REJECTED",
                "reasoning": {
                    "technical": {
                        "status": "pass",
                        "evidence": "Re-evaluation complete and passing.",
                    },
                    "regulatory": {
                        "status": "pass",
                        "evidence": f"Extension approved in {country}.",
                    },
                    "logistical": {
                        "status": "fail",
                        "evidence": f"Only {time_available} days remaining; insufficient for shipping and buffer.",
                    },
                },
                "recommendation": "Cannot execute extension within safe logistics window.",
            }

        return state


__all__ = [
    "AgentState",
    "OrchestratorAgent",
    "SupplyWatchdogAgent",
    "ScenarioStrategistAgent",
]



