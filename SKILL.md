"""
SKILL: open-notebook-clone
Inspirado en NotebookLM (Google) y open-notebook (lfnovo).
Implementa un cuaderno digital con:
- Ingesta multimodal (PDF, markdown, URL, audio, video)
- Embeddings locales con Ollama (nomic-embed-text)
- Vector store con ChromaDB
- Chat con citas a fuentes
- Resumenes automaticos
- 100% local y gratuito

NOTA IMPORTANTE:
- Esta skill NO levanta la aplicacion web de open-notebook.
- Es una API Python + SKILL.md para que Hermes pueda actuar como
  cuaderno digital conversacional.
- Toda la computacion corre en tu PC con Ollama local.

WORKFLOW:
1. Crear notebook
2. Agregar fuentes (PDF, URL, audio, video)
3. Preguntar al notebook (respuestas con citas)
4. Generar resumenes

LIMITACIONES:
- Sin UI web propia (usa el chat de Hermes).
- Sin sincronizacion entre dispositivos.
- Sin podcasts automaticos (TTS ya esta en Hermes via skill elementos-maquinas).

DEPENDENCIAS:
- Ollama con nomic-embed-text y un modelo LLM (gemma3:4b por defecto)
- ChromaDB, pymupdf, yt-dlp, faster-whisper (ya instaladas)
"""

__version__ = "0.1.0"
