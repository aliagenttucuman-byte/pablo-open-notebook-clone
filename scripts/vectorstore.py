"""
Vector store local con ChromaDB.
Persistencia en disco, busqueda semantica por similitud coseno.
"""
import os
import uuid
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional


DEFAULT_PERSIST_DIR = os.environ.get(
    "OPEN_NOTEBOOK_DB", r"C:/Users/pablo/AppData/Local/hermes/skills/open-notebook-clone/data/chroma"
)


_client = None


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR):
    global _client
    if _client is None:
        os.makedirs(persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
    return _client


def get_collection(name: str, persist_dir: str = DEFAULT_PERSIST_DIR):
    return get_client(persist_dir).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    collection_name: str,
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    """Agrega documentos al vector store. Devuelve los IDs usados."""
    if not texts:
        return []
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]
    col = get_collection(collection_name)
    col.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return ids


def query(
    collection_name: str,
    query_embedding: List[float],
    n_results: int = 5,
    where: Optional[dict] = None,
) -> Dict:
    """Busqueda por similitud. Devuelve dict con documents, metadatas, distances."""
    col = get_collection(collection_name)
    kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def count(collection_name: str) -> int:
    return get_collection(collection_name).count()


def collections_list() -> List[str]:
    return [c.name for c in get_client().list_collections()]


def delete_collection(name: str) -> None:
    get_client().delete_collection(name)


if __name__ == "__main__":
    print("Colecciones existentes:", collections_list())
    for c in collections_list():
        print(f"  - {c}: {count(c)} docs")
