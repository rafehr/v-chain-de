import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


@lru_cache(maxsize=10)
def get_model(
    model_name: str, temperature: float, max_new_tokens: int = 4096
) -> ChatHuggingFace:
    """
    Instantiates a model using HuggingFaceEndpoint.
    """
    llm = HuggingFaceEndpoint(
        model=model_name,
        task="conversational",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=ACCESS_TOKEN,
    )
    return ChatHuggingFace(llm=llm)
