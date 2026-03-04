import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.utils import stream_jsonl

load_dotenv()

FILTERED_DATA_PATH = os.getenv("FILTERED_OFF_DATA_PATH")
if not FILTERED_DATA_PATH:
    raise ValueError("Data path not set in .env.")

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
if not VECTOR_DB_DIR:
    raise ValueError("Vector DB dir not set in .env.")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / FILTERED_DATA_PATH
VECTOR_DB_PATH = str(BASE_DIR / VECTOR_DB_DIR)


def convert_jsonl_to_doc(entry: Dict) -> Document:
    content = entry["description"]

    metadata = {
        "id": entry.get("id"),
        "product_name": entry.get("product_name"),
        "brands": entry.get("brand_names"),
        "ingredients": entry.get("ingredients"),
        "categories": entry.get("categories"),
    }
    return Document(page_content=content, metadata=metadata)


def create_vector_db(file_path: Path, vector_db: Chroma, batch_size: int = 500) -> None:
    file_size = os.path.getsize(DATA_FILE)
    existing_ids = set(vector_db.get()["ids"])
    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Embedding product information",
    ) as pbar:
        batch = []
        for entry, line_length in stream_jsonl(DATA_FILE):
            pbar.update(line_length)

            product_id = str(entry.get("id"))

            if product_id in existing_ids:
                continue

            doc = convert_jsonl_to_doc(entry)
            batch.append(doc)

            if len(batch) >= batch_size:
                ids = [d.metadata["id"] for d in batch]
                vector_db.add_documents(documents=batch, ids=ids)
                batch = []

        if batch:
            ids = [d.metadata["id"] for d in batch]
            vector_db.add_documents(documents=batch, ids=ids)


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(embedding_function=embeddings, persist_directory=VECTOR_DB_PATH)
    create_vector_db(DATA_FILE, vector_db)


if __name__ == "__main__":
    main()
