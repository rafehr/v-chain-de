from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from food_agent.utils.nodes import refine_query
from food_agent.utils.state import GraphState

builder = StateGraph(GraphState)

builder.add_node("refine_query", refine_query)

builder.add_edge(START, "refine_query")
builder.add_edge("refine_query", END)

query = """Hey, how are you! I'm talking to a Chatbot for the first time.
I am not sure how this work, but here we go: I would like to give me results
for Ice Cream with caramell. Maybe by Ben and Jerry's, but other brands are fine, too."""

agent = builder.compile()
inputs: GraphState = {
    "messages": [HumanMessage(content=query)],
    "refined_query": "",
    "search_params": None,
    "retrieved_products": [],
    "retry_count": 0,
}
agent.invoke(inputs)
