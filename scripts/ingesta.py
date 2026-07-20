"""
Ingesta de fuentes: PDFs, URLs, Markdown, audio, video.
Genera chunks de texto listos para embeddings.
"""
import os
import re
import hashlib
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Optional


CHUNK_SIZE = 1500      # caracteres por chunk
CHUNK_OVERLAP = 200    # solapamiento entre chunks


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = max(chunk.rfind(". "), chunk.rfind("\n"))
            if last_period > chunk_size * 0.5:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def ingest_pdf(path: str) -> Tuple[List[str], List[dict]]:
    """Lee un PDF y devuelve chunks + metadatos con numero de pagina."""
    import fitz  # pymupdf
    doc = fitz.open(path)
    texts, metas = [], []
    fname = Path(path).name
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if not page_text.strip():
            continue
        page_chunks = _chunk_text(page_text)
        for chunk in page_chunks:
            texts.append(chunk)
            metas.append({
                "source": fname,
                "path": path,
                "page": page_num,
                "type": "pdf",
                "hash": _file_hash(path),
            })
    doc.close()
    return texts, metas


def ingest_markdown(path: str) -> Tuple[List[str], List[dict]]:
    """Lee un markdown o texto plano."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    fname = Path(path).name
    texts, metas = [], []
    sections = re.split(r"\n(?=#+\s)", text)
    for i, section in enumerate(sections, start=1):
        if not section.strip():
            continue
        for chunk in _chunk_text(section):
            metas.append({
                "source": fname,
                "path": path,
                "section": i,
                "type": "markdown",
                "hash": _file_hash(path),
            })
            texts.append(chunk)
    if not texts:
        texts = _chunk_text(text)
        metas = [{"source": fname, "path": path, "type": "markdown", "hash": _file_hash(path)} for _ in texts]
    return texts, metas


def ingest_url(url: str) -> Tuple[List[str], List[dict]]:
    """Descarga una URL, extrae el texto principal con BS4."""
    from bs4 import BeautifulSoup
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Hermes-OpenNotebook/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError as e:
        raise RuntimeError(f"No se pudo descargar {url}: {e}")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else url
    paragraphs = []
    for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "article"]):
        text = p.get_text(" ", strip=True)
        if len(text) > 30:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs) or soup.get_text(" ", strip=True)
    metas = []
    texts = []
    for chunk in _chunk_text(text):
        metas.append({
            "source": title,
            "url": url,
            "type": "web",
            "hash": hashlib.sha256(url.encode()).hexdigest()[:16],
        })
        texts.append(chunk)
    return texts, metas


def ingest_audio(path: str, language: str = "es") -> Tuple[List[str], List[dict]]:
    """Transcribe un audio con faster-whisper local."""
    from tools.transcription_tools import transcribe_audio
    fname = Path(path).name
    result = transcribe_audio(path)
    if not result.get("success"):
        raise RuntimeError(f"Transcripcion fallida: {result.get('error')}")
    transcript = result["transcript"]
    metas = []
    texts = []
    for chunk in _chunk_text(transcript):
        metas.append({
            "source": fname,
            "path": path,
            "type": "audio_transcript",
            "language": language,
            "hash": _file_hash(path),
        })
        texts.append(chunk)
    return texts, metas


def ingest_file(path: str, language: str = "es") -> Tuple[List[str], List[dict]]:
    """Dispatcher: detecta el tipo de archivo y llama al ingest correcto."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(str(p))
    if suffix in (".md", ".markdown", ".txt", ".rst"):
        return ingest_markdown(str(p))
    if suffix in (".ogg", ".mp3", ".wav", ".m4a", ".opus", ".flac"):
        return ingest_audio(str(p), language=language)
    if suffix in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        # Extraer audio del video con yt-dlp
        from yt_dlp import YoutubeDL
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": tmp_path.replace(".wav", ".%(ext)s"),
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
                "quiet": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([str(p)])
            texts, metas = ingest_audio(tmp_path, language=language)
            for m in metas:
                m["original_video"] = str(p)
                m["type"] = "video_transcript"
            return texts, metas
        finally:
            for ext in [".wav", ".mp3", ".m4a", ".opus", ".webm"]:
                candidate = tmp_path.replace(".wav", ext)
                if os.path.exists(candidate):
                    try:
                        os.unlink(candidate)
                    except OSError:
                        pass
    raise ValueError(f"Tipo de archivo no soportado: {suffix}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python ingesta.py <archivo.pdf|md|txt|audio|video>")
    else:
        texts, metas = ingest_file(sys.argv[1])
        print(f"Chunks: {len(texts)}")
        if metas:
            print(f"Primer chunk: {texts[0][:200]}...")
            print(f"Meta: {metas[0]}")
