from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    search_params: Dict | None
    retrieved_products: List[Dict]
    retry_count: int
