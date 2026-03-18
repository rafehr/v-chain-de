from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from food_agent.utils.nodes import (
    chat_node,
    error_handling,
    input_classifier,
    input_router,
    query_db,
    refine_query,
    refinement_router,
    validate_refinement,
)
from food_agent.utils.state import GraphState
from food_agent.utils.visualization import draw_graph

builder = StateGraph(GraphState)

builder.add_node("refine_query", refine_query)
builder.add_node("validate_refinement", validate_refinement)
builder.add_node("query_db", query_db)
builder.add_node("error_handling", error_handling)
builder.add_node("input_classifier", input_classifier)
builder.add_node("chat_node", chat_node)

builder.add_edge(START, "input_classifier")
builder.add_conditional_edges(
    "input_classifier", input_router, {"query": "refine_query", "no_query": "chat_node"}
)
builder.add_edge("refine_query", "validate_refinement")
builder.add_conditional_edges(
    "validate_refinement",
    refinement_router,
    {"retry": "refine_query", "query_db": "query_db", "give_up": "error_handling"},
)
builder.add_edge("query_db", END)
builder.add_edge("chat_node", END)

query = """Hey, how are you! I'm talking to a Chatbot for the first time.
I am not sure how this work, but here we go: I would like to give me results
for Ice Cream with caramell. Maybe by Ben and Jerry's, but other brands are fine, too."""
# query = "Hey, what is your name?"

agent = builder.compile()

draw_graph(agent)

inputs: GraphState = {
    "messages": [HumanMessage(content=query)],
    "refined_query": "",
    "search_params": None,
    "retrieved_products": [],
    "retry_count": 0,
    "error_log": None,
    "is_query": False,
}
agent.invoke(inputs)
