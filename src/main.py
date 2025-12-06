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

 
    # Scenario 2: User query (extension feasibility)
    # For the demo, call the Strategist directly with a known-good example
    # so that all three gates (technical, regulatory, logistics) can pass.
    from src.agents import ScenarioStrategistAgent  # local import to avoid cycles

    strategist = ScenarioStrategistAgent()
    user_state = AgentState(
        trigger_source="user",
        user_query=(
            "Can we extend the shelf-life of Batch LOT-51070012 "
            "for trial CT-2004-PSX in Saint Kitts and Nevis?"
        ),
        selected_agent="scenario_strategist",
        extracted_entities={
            "batch_number": "LOT-51070012",
            "trial_id": "CT-2004-PSX",
            "country": "Saint Kitts and Nevis",
        },
    )
   
    user_result_state = strategist.invoke(user_state)
    print("\n=== SCENARIO 2: Extension Feasibility Check (User Query) ===")
    print(json.dumps(user_result_state.final_output or {}, indent=2, default=str))


if __name__ == "__main__":
    app = create_workflow()
    try:
        demo_scenarios(app)
    finally:
        db.close()


