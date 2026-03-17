from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from food_agent.utils.nodes import (
    error_handling,
    query_db,
    refine_query,
    refinement_router,
    validate_refinement,
)
from food_agent.utils.state import GraphState

builder = StateGraph(GraphState)

builder.add_node("refine_query", refine_query)
builder.add_node("validate_refinement", validate_refinement)
builder.add_node("query_db", query_db)
builder.add_node("error_handling", error_handling)

builder.add_edge(START, "refine_query")
builder.add_edge("refine_query", "validate_refinement")
builder.add_conditional_edges(
    "validate_refinement",
    refinement_router,
    {"retry": "refine_query", "query_db": "query_db", "give_up": "error_handling"},
)
builder.add_edge("query_db", END)

query = """Hey, how are you! I'm talking to a Chatbot for the first time.
I am not sure how this work, but here we go: I would like to give me results
for Ice Cream with caramell. Maybe by Ben and Jerry's, but other brands are fine, too."""

agent = builder.compile()

# draw_graph(agent)

inputs: GraphState = {
    "messages": [HumanMessage(content=query)],
    "refined_query": "",
    "search_params": None,
    "retrieved_products": [],
    "retry_count": 0,
    "error_log": None,
}
agent.invoke(inputs)
