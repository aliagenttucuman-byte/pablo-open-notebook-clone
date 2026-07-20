"""
Embeddings locales con Ollama (nomic-embed-text).
100% offline, sin costo, sin API keys.
Soporta batch embeddings via /api/embed.
"""
import os
import json
import urllib.request
import urllib.error
import concurrent.futures
from typing import List, Union


OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
MAX_WORKERS = int(os.environ.get("OLLAMA_EMBED_WORKERS", "4"))


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


def _embed_batch_endpoint(inputs: List[str], model: str) -> List[List[float]]:
    """Usa /api/embed (batch nativo de Ollama, soporta listas grandes)."""
    payload = {"model": model, "input": inputs}
    response = _post("/api/embed", payload, timeout=600)
    return response.get("embeddings", [])


def _embed_single_endpoint(text: str, model: str) -> List[float]:
    """Fallback: /api/embeddings con un solo prompt (compatibilidad maxima)."""
    payload = {"model": model, "prompt": text}
    response = _post("/api/embeddings", payload, timeout=180)
    return response.get("embedding", [])


def embed(text: Union[str, List[str]], model: str = DEFAULT_MODEL, is_query: bool = False) -> List[List[float]]:
    """
    Genera embeddings con Ollama. Devuelve una lista de vectores (uno por texto).
    Si recibe un solo string, devuelve un solo vector.

    is_query=True agrega el prefijo search_query: (usado en busquedas).
    is_query=False agrega search_document: (usado al ingestar).

    Intenta primero /api/embed (batch nativo) y si falla hace fallback a
    /api/embeddings en paralelo.
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

    # 1) Intentar batch nativo (/api/embed)
    try:
        embeddings = _embed_batch_endpoint(inputs, model)
        if len(embeddings) == len(inputs):
            return embeddings[0] if single else embeddings
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # Fallback a single endpoint si /api/embed no existe (versiones viejas)
    except Exception as e:
        # Cualquier otro error, intentar fallback
        pass

    # 2) Fallback: /api/embeddings en paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(inputs))) as ex:
        embeddings = list(ex.map(lambda t: _embed_single_endpoint(t, model), inputs))

    return embeddings[0] if single else embeddings


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
    """Verifica que Ollama este corriendo y devuelve los modelos disponibles."""
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
        # Test batch
        batch = embed(["primer texto", "segundo texto", "tercer texto"])
        print(f"Batch OK: {len(batch)} vectores de dim {len(batch[0])}")
