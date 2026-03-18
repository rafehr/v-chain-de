from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from food_agent.utils.models import get_model
from food_agent.utils.prompts import REFINEMENT_PROMPT
from food_agent.utils.state import GraphState


class RefinedQuery(BaseModel):
    original_query: str = Field(description="The original, non-refined query.")

    product_name: str | None = Field(description="The name of the product")
    brands: str | None = Field(description="The brand name(s)")
    ingredients: str | None = Field(
        description="The ingredients the product should contain."
    )
    categories: str | None = Field(
        description="The categories a product belongs to (e.g. breakfast spreads)."
    )


def input_router(state: GraphState):
    pass


def input_classifier(state: GraphState) -> dict:
    return {}


def chat(state: GraphState) -> dict:
    return {}


def refine_query(state: GraphState) -> dict:
    count = state.get("retry_count", 0) + 1
    parser = PydanticOutputParser(pydantic_object=RefinedQuery)
    user_query = state["messages"][-1].content
    format_instructions = parser.get_format_instructions()
    prompt = REFINEMENT_PROMPT.partial(format_instructions=format_instructions)
    model = get_model(
        model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_new_tokens=100
    )
    chain = prompt | model
    response = chain.invoke({"user_query": user_query})
    return {"refined_query": response.content, "retry_count": count}


def validate_refinement(state: GraphState) -> dict:
    parser = PydanticOutputParser(pydantic_object=RefinedQuery)
    try:
        refined_query = parser.parse(state["refined_query"])
        print(refined_query)
        return {"refined_query": refined_query, "error_log": None}
    except Exception as e:
        return {"error_log": str(e)}


def error_handling(state: GraphState):
    print("Better luck next time!")


def query_db(state: GraphState) -> dict:
    print("Success!")
    return {}


def refinement_router(state: GraphState) -> str:
    if not state["error_log"]:
        return "query_db"

    if state.get("retry_count", 0) >= 3:
        return "give_up"

    return "retry"
