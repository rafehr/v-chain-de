import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
if not VECTOR_DB_DIR:
    raise ValueError("Vector DB dir not set in .env.")

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = str(BASE_DIR / VECTOR_DB_DIR)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

db = Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)

print(f"Database loaded successfully. Number of products: {db._collection.count()}")
