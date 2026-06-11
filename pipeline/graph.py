from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import FTOState
from .nodes import (
    node_expand_queries,
    node_retrieve_patents,
    node_filter_relevant_patents,
    node_score_retrieval_quality,
    node_increment_retry,
    node_no_results,
    node_seed_claims_to_chromadb,
    node_retrieve_top_claims,
    node_assess_risk,
    node_human_review,
    node_write_report,
)
from .routing import route_after_quality_check

workflow = StateGraph(FTOState)

workflow.add_node("query_expander", node_expand_queries)
workflow.add_node("patent_retriever", node_retrieve_patents)
workflow.add_node("relevance_filter", node_filter_relevant_patents)
workflow.add_node("quality_checker", node_score_retrieval_quality)
workflow.add_node("retry_handler", node_increment_retry)
workflow.add_node("no_results", node_no_results)
workflow.add_node("chromadb_seeder", node_seed_claims_to_chromadb)
workflow.add_node("claim_retriever", node_retrieve_top_claims)
workflow.add_node("risk_assessor", node_assess_risk)
workflow.add_node("human_review", node_human_review)
workflow.add_node("analyst", node_write_report)

workflow.set_entry_point("query_expander")
workflow.add_edge("query_expander", "patent_retriever")
workflow.add_edge("patent_retriever", "relevance_filter")
workflow.add_edge("relevance_filter", "quality_checker")
workflow.add_conditional_edges(
    "quality_checker",
    route_after_quality_check,
    {
        "no_results": "no_results",
        "retry": "retry_handler",
        "chromadb_seeder": "chromadb_seeder",
    },
)
workflow.add_edge("no_results", END)
workflow.add_edge("retry_handler", "query_expander")
workflow.add_edge("chromadb_seeder", "claim_retriever")
workflow.add_edge("claim_retriever", "risk_assessor")
workflow.add_edge("risk_assessor", "human_review")
workflow.add_edge("human_review", "analyst")
workflow.add_edge("analyst", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review"])


if __name__ == "__main__":
    print("starting pipeline\n")

    config = {"configurable": {"thread_id": "fto-run-1"}}
    initial_input = {
        "user_idea": "A drone that spray paints houses",
        "sub_queries": [],
        "raw_patents": [],
        "decomposed_claims": [],
        "risk_assessments": [],
        "cleared_patents": [],
        "retrieval_quality_score": 0.0,
        "retry_count": 0,
        "human_approved": False,
        "final_report": "",
    }

    try:
        state_after_phase1 = app.invoke(initial_input, config=config)

        print("\nrisk assessments:")
        for r in state_after_phase1.get("risk_assessments", []):
            print(f"  {r['patent_id']}  {r['risk_level']}  {r['overlap_score']}")

        print("\nreview required")
        approval = input("approve report? (yes/no): ").strip().lower()

        if approval != "yes":
            print("cancelled")
            raise SystemExit(0)

        final_state = app.invoke(None, config=config)

        print("\nfinal report:\n")
        print(final_state["final_report"])

    except SystemExit:
        raise
    except Exception as e:
        print(f"error: {e}")
        raise
