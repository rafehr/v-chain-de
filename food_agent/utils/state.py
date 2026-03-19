from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import NotRequired


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    refined_query: str
    search_params: Dict | None
    retrieved_products: List[Dict]
    retry_count: int
    error_log: str | None
    is_query: bool
    remaining_steps: NotRequired[RemainingSteps]
