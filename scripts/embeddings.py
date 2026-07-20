"""
Embeddings locales con Ollama (nomic-embed-text).
100% offline, sin costo, sin API keys.
"""
import os
import json
import urllib.request
import urllib.error
from typing import List, Union


OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    url = f"{OLLAMA_BASE}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _embed_one(text: str, model: str) -> List[float]:
    """Embed de un solo texto. Funciona con cualquier modelo Ollama."""
    payload = {"model": model, "prompt": text}
    response = _post("/api/embeddings", payload, timeout=180)
    return response.get("embedding", [])


def embed(text: Union[str, List[str]], model: str = DEFAULT_MODEL, is_query: bool = False) -> List[List[float]]:
    """
    Genera embeddings con Ollama. Devuelve una lista de vectores (uno por texto).
    Si recibe un solo string, devuelve un solo vector.

    is_query=True agrega el prefijo search_query: (usado en busquedas).
    is_query=False agrega search_document: (usado al ingestar).
    """
    if isinstance(text, str):
        single = True
        inputs = [text]
    else:
        single = False
        inputs = list(text)

    # nomic-embed-text requiere un prefijo para distinguir consulta vs documento
    if "nomic-embed" in model.lower():
        prefix = "search_query: " if is_query else "search_document: "
        inputs = [prefix + t for t in inputs]

    # nomic-embed-text falla con prompt como lista: procesamos uno por uno
    embeddings = [_embed_one(t, model) for t in inputs]
    if single:
        return embeddings[0] if embeddings else []
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Similitud coseno entre dos vectores."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def health_check() -> dict:
    """Verifica que Ollama esté corriendo y devuelve los modelos disponibles."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        return {"ok": True, "models": models, "base_url": OLLAMA_BASE}
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e), "base_url": OLLAMA_BASE}


if __name__ == "__main__":
    h = health_check()
    print("Ollama health:", h)
    if h.get("ok"):
        v = embed("Hola, esto es una prueba de embeddings.")
        print(f"Vector dim: {len(v)}")
        print(f"Primeros 5 valores: {v[:5]}")
