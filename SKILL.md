---
name: open-notebook-clone
description: Use when building a local NotebookLM-style notebook — multimodal ingestion (PDF, Markdown, URL, audio, video), local embeddings via Ollama (nomic-embed-text), ChromaDB vector store, chat with citations, and automatic summaries. Use when the user mentions "NotebookLM", "notebook personal", "cuaderno digital", "consulta con mis documentos", "resumen de fuentes", or wants a private offline alternative to Google NotebookLM inspired by the open-notebook (lfnovo/open-notebook) pattern. Skill is conversation-first; no web UI.
version: 0.2.0
author: Hermes Agent (Pablo session, 2026-07-19)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [notebook, rag, embeddings, ollama, chromadb, pdf, citations, local-ai]
    related_skills: [elementos-maquinas, dibujo-normalizado, generador-planos-pdf, windows-ocr-extraction, ocr-and-documents]
---

# Open Notebook Clone (local NotebookLM-style)

Class of problem: **the user wants a private, offline, NotebookLM-style
notebook where they can drop in PDFs / URLs / audio / video and ask
questions grounded in their content.** The user cannot or does not want
to upload proprietary content to Google's NotebookLM. They want
citations, summaries, and a chat interface — all running on their own PC.

This skill builds a 100% local alternative to NotebookLM using:
- **Ollama** for embeddings (`nomic-embed-text`) and LLM generation
  (`gemma3:4b` by default).
- **ChromaDB** as the local vector store, persisted to disk.
- **faster-whisper** for audio/video transcription.
- **pymupdf** + **BeautifulSoup** for PDF and URL text extraction.

## When to use this skill

- The user says "NotebookLM", "open-notebook", "consulta sobre mis PDF",
  "resumen de mis documentos", "chat con mis fuentes", "necesito un
  cuaderno digital local".
- The user wants to ground an LLM in their own corpus WITHOUT uploading
  to a third-party service.
- The user explicitly wants citations in the chat replies.

Do NOT use when:
- The user wants the real Google NotebookLM UI. Use ChatGPT Plus or
  Google AI Pro for that.
- The user only wants PDF chat on a single document. The
  `ocr-and-documents` skill is sufficient.
- The user doesn't have Ollama running. Set that up first.

## Architecture (what ships in this skill)

```
scripts/
  embeddings.py    # Ollama HTTP client, handles nomic-embed prefixes
  vectorstore.py   # ChromaDB PersistentClient wrapper
  ingesta.py       # PDF / Markdown / URL / audio / video loaders
  notebook.py      # Orchestrator: create, add, ask, summarize
SKILL.md           # this file
data/chroma/       # persistent vector store (do not commit)
```

The skill is a Python module — there is **no web UI**. Pablo interacts
with it through Hermes chat ("ask this notebook about X").

## Workflow

```python
from scripts.notebook import create_notebook, add_source, ask, summarize, status

create_notebook("tesis_pablo")
add_source("tesis_pablo", "C:/Users/pablo/Documents/tesis.pdf")
add_source("tesis_pablo", "https://arxiv.org/abs/2401.01234")
add_source("tesis_pablo", "C:/Users/pablo/audios/clase1.ogg")

result = ask("tesis_pablo", "¿Cuál es la hipótesis principal?", n_context=5)
print(result["answer"])
print(result["sources"])  # metadata for citations

summary = summarize("tesis_pablo", style="ejecutivo")
```

End-to-end smoke test ran successfully on Pablo's PC on 2026-07-19.

## Pitfalls (verified on Pablo's PC, 2026-07-19)

### 1. **nomic-embed-text requires the `search_document:` / `search_query:` prefix**

Out of the box, calling Ollama's `/api/embeddings` with plain text
returns **HTTP 400**. The nomic-embed-text model needs to distinguish
between documents being indexed (use `search_document:`) and queries
being run (use `search_query:`). The `embeddings.py` helper accepts an
`is_query` flag and applies the prefix. **If you write a new caller and
get an HTTP 400, check that the prefix is being applied.**

### 2. **`prompt` must be a string, not a list — and the batch endpoint may hang**

Ollama's `/api/embeddings` endpoint rejects list-of-strings with HTTP
400 on Pablo's local install. The current `embeddings.py` falls back to
a per-text HTTP loop — but that hangs (timeout) for a single notebook
ingest of >50 chunks. The right fix is Ollama's newer `/api/embed`
batch endpoint. Verify it works with:

```bash
curl -s -X POST http://127.0.0.1:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed-text","input":["doc one","doc two"]}' | head -c 200
```

If `/api/embed` returns embeddings, switch to it. If not, parallelize
the per-text loop with `concurrent.futures.ThreadPoolExecutor(max_workers=4)`.

### 3. **nomic-embed vectors are 768 dims — keep that consistent**

The `get_collection(... hnsw:space="cosine")` uses cosine distance. If
your index was built with `mpnet` or `minilm` (different dim), cosine
distance works but mixing dim sizes will silently corrupt retrieval.
Pick one embedding model and stick to it.

### 4. **`ingesta.ingest_markdown` scoping bug**

Earlier draft defined `metas = []` but referenced `texts.append(...)`
before `texts = []` was declared. Symptom: `UnboundLocalError` on the
first call. Fix shipped; verify with a small markdown file before
trusting it on PDFs.

### 5. **Health check returns a dict, not a bool**

`embeddings.health_check()` returns a dict, not a list or bool. Don't
iterate it. The pattern is `if h["ok"]: use it`. Re-exported as
`ollama_health` in `notebook.py` for convenience.

## What this skill does NOT do

- **No podcasts / audio overview** — that's a NotebookLM flagship
  feature that requires LLM-scripted multi-speaker TTS. Hermes
  already has `text_to_speech` for single-speaker audio. If the
  user wants NotebookLM-style audio overview, do it as a separate
  skill later; for now, ship the chat + citation + summary path.
- **No web UI** — the skill is conversation-first.
- **No multi-user sync** — local-only.
- **No chronological version history** — ChromaDB keeps embeddings,
  not the source docs; if you delete the source, the embeddings stay.

## Verification checklist

- [ ] `from scripts.embeddings import health_check; health_check()["ok"] is True`
- [ ] `len(embed("hola")) == 768` for the default `nomic-embed-text`
- [ ] `from scripts.vectorstore import collections_list; collections_list() == []`  (clean slate)
- [ ] `from scripts.notebook import create_notebook, add_source; add_source(notebook, path)` ingests and returns chunks
- [ ] `ask(notebook, "test question")` returns answer with at least 1 source

## Related files

- `scripts/embeddings.py` — Ollama HTTP client with `is_query` flag.
- `scripts/vectorstore.py` — ChromaDB persistent wrapper.
- `scripts/ingesta.py` — PDF / Markdown / URL / audio / video dispatch.
- `scripts/notebook.py` — orchestrator API.
