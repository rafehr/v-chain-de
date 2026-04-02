import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from food_agent.agent import agent
from food_agent.utils.state import GraphState

load_dotenv()

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME")


def generate_response(query: str, history: list[list[str | None]], session_id: str):
    inputs: GraphState = {
        "messages": [HumanMessage(content=query)],
        "refined_query": "",
        "original_query": None,
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

    full_response = ""

    try:
        for chunk, metadata in agent.stream(
            inputs, config=config, stream_mode="messages"
        ):
            if isinstance(chunk, BaseMessage):
                if metadata.get("langgraph_node") == "generate_answer":
                    if chunk.content:
                        full_response += chunk.content
                        yield full_response
    except GraphRecursionError:
        yield {"messges": ["Recursion limit exceeded"]}
    except Exception as e:
        yield f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        session_id = gr.State(str(uuid.uuid4()))

        gr.ChatInterface(
            fn=generate_response,
            additional_inputs=[session_id],
        )

    # Launch the interface
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=7860)
