from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from food_agent.utils.models import get_model
from food_agent.utils.state import GraphState


class RefinedQuery(BaseModel):
    pass


def refine_query(state: GraphState) -> dict:
    refinement_task = (
        "Your job is to refine the query. Nothing else. This means "
        "your answer will only be the refined query. For example "
        "Product: Chocolate, Brand: Hersheys"
        "I do not want you to utter anything additional."
    )

    prompt = ChatPromptTemplate(
        [("system", refinement_task), ("placeholder", "{messages}")]
    )
    model = get_model(
        model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_new_tokens=100
    )
    chain = prompt | model
    response = chain.invoke(dict(state))
    print(response.content)
    return {"refined_query": "Refined Query"}
