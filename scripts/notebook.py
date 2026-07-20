"""
Orquestador principal de un Notebook.
Permite: crear, ingestar fuentes, preguntar con citas, generar resumenes y podcasts.
"""
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional

from ingesta import ingest_file, ingest_url
from embeddings import embed, health_check as ollama_health
from vectorstore import (
    add_documents,
    query as vector_query,
    get_collection,
    count,
    delete_collection,
    collections_list,
)


OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_LLM = os.environ.get("OLLAMA_LLM_MODEL", "gemma3:4b")


def _ollama_generate(prompt: str, model: str = DEFAULT_LLM, system: str = "") -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def create_notebook(name: str) -> str:
    """Crea un notebook vacio. Devuelve el nombre."""
    get_collection(name)
    return name


def list_notebooks() -> List[str]:
    return collections_list()


def delete_notebook(name: str) -> None:
    delete_collection(name)


def add_source(notebook: str, source: str, language: str = "es") -> Dict:
    """
    Agrega una fuente (archivo o URL) al notebook.
    Genera chunks, embeddings y los guarda en el vector store.
    Devuelve un resumen de lo agregado.
    """
    if source.startswith(("http://", "https://")):
        texts, metas = ingest_url(source)
    else:
        texts, metas = ingest_file(source, language=language)
    if not texts:
        return {"ok": False, "error": "No se extrajo texto de la fuente."}
    embeddings = embed(texts)
    ids = add_documents(notebook, texts, embeddings, metas)
    return {
        "ok": True,
        "notebook": notebook,
        "source": source,
        "chunks": len(texts),
        "ids": ids[:5] + (["..."] if len(ids) > 5 else []),
        "metadata_sample": metas[0],
    }


def notebook_info(notebook: str) -> Dict:
    return {
        "notebook": notebook,
        "documentos": count(notebook),
    }


def ask(
    notebook: str,
    question: str,
    n_context: int = 5,
    model: str = DEFAULT_LLM,
) -> Dict:
    """
    Pregunta al notebook. Devuelve respuesta con citas a fuentes.
    """
    if count(notebook) == 0:
        return {"ok": False, "error": f"Notebook '{notebook}' vacio."}
    q_emb = embed(question, is_query=True)
    results = vector_query(notebook, q_emb, n_results=n_context)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return {"ok": False, "error": "Sin contexto relevante."}
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        cite = meta.get("source") or meta.get("url", "?")
        if meta.get("page"):
            cite += f" (pag. {meta['page']})"
        context_blocks.append(f"[Fuente {i}: {cite}]\n{doc}")
    context = "\n\n".join(context_blocks)
    prompt = (
        f"Basandote SOLO en el siguiente contexto, responde la pregunta.\n"
        f"Al final lista las fuentes usadas con el formato [Fuente N: nombre (pag. X)].\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {question}\n\n"
        f"RESPUESTA:"
    )
    answer = _ollama_generate(prompt, model=model)
    return {
        "ok": True,
        "notebook": notebook,
        "question": question,
        "answer": answer,
        "sources": metadatas,
        "context_used": len(documents),
    }


def summarize(notebook: str, style: str = "ejecutivo", model: str = DEFAULT_LLM) -> Dict:
    """
    Genera un resumen del notebook.
    Estilos: 'ejecutivo', 'tecnico', 'preguntas_respuestas'.
    """
    if count(notebook) == 0:
        return {"ok": False, "error": f"Notebook '{notebook}' vacio."}
    results = vector_query(notebook, embed(f"resumen {style}"), n_results=20)
    documents = results.get("documents", [[]])[0]
    if not documents:
        return {"ok": False, "error": "Sin contexto."}
    full_text = "\n\n".join(documents)
    if len(full_text) > 12000:
        full_text = full_text[:12000] + "..."
    styles = {
        "ejecutivo": "Resume el siguiente contenido en un parrafo ejecutivo (max 5 lineas). Destaca los puntos mas importantes para un lector que no tiene tiempo de leer el documento completo.",
        "tecnico": "Resume el siguiente contenido con detalle tecnico. Manten los terminos tecnicos y los datos numericos clave. Estructura por secciones.",
        "preguntas_respuestas": "Genera 10 preguntas frecuentes (FAQ) sobre el siguiente contenido y respindelas brevemente.",
    }
    system = styles.get(style, styles["ejecutivo"])
    answer = _ollama_generate(full_text, model=model, system=system)
    return {
        "ok": True,
        "notebook": notebook,
        "style": style,
        "summary": answer,
    }


def status() -> Dict:
    """Estado del sistema."""
    h = ollama_health()
    return {
        "ollama": h,
        "notebooks": collections_list(),
    }


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Comando no soportado: {cmd}")
