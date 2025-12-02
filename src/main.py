import json
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from src.agents import (
    AgentState,
    OrchestratorAgent,
    SupplyWatchdogAgent,
    ScenarioStrategistAgent,
)
from src.tools import db


def create_workflow() -> Any:
    """Build and compile the multi-agent LangGraph workflow."""

    workflow = StateGraph(AgentState)

    orchestrator = OrchestratorAgent()
    watchdog = SupplyWatchdogAgent()
    strategist = ScenarioStrategistAgent()

    workflow.add_node("orchestrator", orchestrator.invoke)
    workflow.add_node("supply_watchdog", watchdog.invoke)
    workflow.add_node("scenario_strategist", strategist.invoke)

    def route_after_orchestrator(state: AgentState) -> str:
        if state.selected_agent == "supply_watchdog":
            return "supply_watchdog"
        if state.selected_agent == "scenario_strategist":
            return "scenario_strategist"
        return END

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "supply_watchdog": "supply_watchdog",
            "scenario_strategist": "scenario_strategist",
            END: END,
        },
    )

    workflow.add_edge("supply_watchdog", END)
    workflow.add_edge("scenario_strategist", END)

    return workflow.compile()


def demo_scenarios(app: Any) -> None:
    """Run example cron and user flows to demonstrate the graph."""

    # Scenario 1: Scheduled health check (cron)
    cron_state = AgentState(
        trigger_source="cron",
        user_query=None,
    )
    cron_result: AgentState = app.invoke(cron_state)
    print("\n=== SCENARIO 1: Daily Health Check (Autonomous) ===")
    print(json.dumps(cron_result.final_output or {}, indent=2, default=str))

    # Scenario 2: User query (extension feasibility)
    user_state = AgentState(
        trigger_source="user",
        user_query=(
            "Can we extend the shelf-life of Batch LOT-2024-456 "
            "for the German trial CT-2004-PSX?"
        ),
    )
    user_result: AgentState = app.invoke(user_state)
    print("\n=== SCENARIO 2: Extension Feasibility Check (User Query) ===")
    print(json.dumps(user_result.final_output or {}, indent=2, default=str))


if __name__ == "__main__":
    app = create_workflow()
    try:
        demo_scenarios(app)
    finally:
        db.close()



