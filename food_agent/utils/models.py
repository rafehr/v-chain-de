import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.utils import convert_to_secret_str
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

HF_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CLAUDE_ACCESS_TOKEN = os.getenv("CLAUDE_API_TOKEN")


@lru_cache(maxsize=10)
def get_model(
    model_name: str,
    temperature: float,
    max_new_tokens: int = 4096,
) -> ChatAnthropic:
    """
    Instantiates a model using HuggingFaceEndpoint.
    """

    if CLAUDE_ACCESS_TOKEN is None:
        raise ValueError("CLAUDE_ACCESS_TOKEN not set.")

    secret_key = convert_to_secret_str(CLAUDE_ACCESS_TOKEN)
    llm = ChatAnthropic(
        model_name=model_name,
        temperature=0,
        max_tokens_to_sample=1024,
        timeout=None,
        stop=None,
        api_key=secret_key,
    )
    return llm


@lru_cache(maxsize=10)
def get_hf_model(
    model_name: str,
    temperature: float,
    max_new_tokens: int = 4096,
    json_output: bool = False,
) -> ChatHuggingFace:
    """
    Instantiates a model using HuggingFaceEndpoint.
    """

    llm = HuggingFaceEndpoint(
        model=model_name,
        task="conversational",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        huggingfacehub_api_token=HF_ACCESS_TOKEN,
    )
    return ChatHuggingFace(llm=llm)
