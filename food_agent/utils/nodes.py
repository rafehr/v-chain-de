from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from food_agent.utils.models import get_model
from food_agent.utils.prompts import (
    CLASSIFICATION_PROMPT,
    QUERY_ANSWER_PROMPT,
    REFINEMENT_PROMPT,
)
from food_agent.utils.state import GraphState
from food_agent.utils.vectorstore import db

model = HuggingFaceCrossEncoder(model_name="mixedbread-ai/mxbai-rerank-xsmall-v1")
reranker = CrossEncoderReranker(model=model, top_n=10)


class RouteInput(BaseModel):
    is_query: bool = Field(
        description="True if the user asks for food products, False otherwise."
    )


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
    if state["is_query"]:
        return "query"
    return "no_query"


def input_classifier(state: GraphState) -> dict:
    parser = PydanticOutputParser(pydantic_object=RouteInput)
    user_input = state["messages"][-1].content
    format_instructions = parser.get_format_instructions()
    prompt = CLASSIFICATION_PROMPT.partial(format_instructions=format_instructions)
    model = get_model(
        model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_new_tokens=100
    )
    chain = prompt | model | parser
    try:
        response = chain.invoke({"user_input": user_input})
        return {"is_query": response.is_query}
    except Exception as e:
        print(e)
        return {"is_query": False}


def chat_node(state: GraphState) -> dict:
    model = get_model(
        model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_new_tokens=100
    )
    response = model.invoke(state["messages"][-1].content)
    return {"messages": [response]}


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
    return {
        "refined_query": response.content,
        "retry_count": count,
        "steps": 1,
    }


def validate_refinement(state: GraphState) -> dict:
    parser = PydanticOutputParser(pydantic_object=RefinedQuery)
    try:
        refined_query = parser.parse(state["refined_query"])
        refined_query = refined_query.model_dump()
        final_query = []
        for key, value in refined_query.items():
            if value and key != "original_query":
                final_query.append(f"{key}: {value}")
        final_query = " ".join(final_query)
        return {
            "refined_query": final_query,
            "error_log": None,
        }
    except Exception as e:
        return {"error_log": str(e)}


def error_handling(state: GraphState):
    return {"messages": AIMessage(["Better luck next time!"])}


def query_db(state: GraphState, config: RunnableConfig) -> dict:
    refined_query = str(state.get("refined_query", ""))

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 12})
    documents = retriever.invoke(refined_query, config=config)
    final_docs = reranker.compress_documents(documents=documents, query=refined_query)
    retrieved_products = [doc.metadata for doc in final_docs]
    return {"retrieved_products": retrieved_products}


def refinement_router(state: GraphState, config: RunnableConfig) -> str:
    remaining = state.get("remaining_steps", 0)

    if state.get("retry_count", 0) >= 3 or remaining <= 2:
        return "give_up"

    if not state["error_log"]:
        return "query_db"

    return "retry"


def generate_answer(state: GraphState) -> dict:
    original_query = state["messages"][-1].content
    products = state.get("retrieved_products", [])
    prompt = QUERY_ANSWER_PROMPT
    model = get_model(
        model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_new_tokens=2000
    )
    chain = prompt | model
    response: BaseMessage = chain.invoke(
        {"original_query": original_query, "context": _format_products(products)}
    )
    return {"messages": [response]}


def _format_products(products: list[dict]) -> str:
    products_str = ""
    count = 1
    for product in products:
        url = "https://de.openfoodfacts.org/produkt/" + product["id"]
        product_info = (
            f"{count}. Product name: {product['product_name']}\n"
            f"Brands: {product['brands']}\n"
            f"Ingredients: {product['ingredients']}\n"
            f"Categories: {product['categories']}\n"
            f"Open food facts url: {url}\n"
        )
        products_str = products_str + product_info
        count += 1
    return products_str
