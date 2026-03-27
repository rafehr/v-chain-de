import uuid

import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from food_agent.agent import agent
from food_agent.utils.state import GraphState


def add_numbers(Num1, Num2):
    return Num1 + Num2


def generate_response(
    query: str, history: list[list[str | None]], session_id: str
) -> str:
    inputs: GraphState = {
        "messages": [HumanMessage(content=query)],
        "refined_query": "",
        "search_params": None,
        "retrieved_products": [],
        "retry_count": 0,
        "error_log": None,
        "is_query": False,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 20,
    }

    try:
        result = agent.invoke(inputs, config=config)
    except GraphRecursionError:
        result = {"messges": ["Recursion limit exceeded"]}

    return result["messages"][-1].content


if __name__ == "__main__":
    with gr.Blocks() as demo:
        session_id = gr.State(str(uuid.uuid4()))

        gr.ChatInterface(
            fn=generate_response,
            additional_inputs=[session_id],
        )

    # Launch the interface
    demo.launch(server_name="127.0.0.1", server_port=7860)
