# MediLex India — Docker deployment.
#
# chroma_db/ is gitignored (see .gitignore) — the vector store is rebuilt
# here from the committed data/raw/*.txt statute files, not copied in.
# medilex.db (session/error logging) is also gitignored and starts fresh
# each deploy — acceptable for a demo/portfolio deployment; if long-term
# session history matters later, mount a persistent volume at /app and
# point config.DB_PATH there instead.
#
# PORT is read at runtime (not baked in) so this same image works on
# Hugging Face Spaces (expects 7860 by default), Render/Railway (inject
# their own PORT), or anywhere else — without touching main.py's existing
# `python main.py` local-dev entrypoint, which still hardcodes 8000.

FROM python:3.12-slim

WORKDIR /app

# build-essential: some transitive deps (e.g. tokenizers used by
# sentence-transformers) may need to compile if no prebuilt wheel matches
# this exact Python/platform combo. Removed after pip install to keep the
# image lean — only needed at build time, not runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Rebuild the vector store now, once, at image-build time — not on every
# container start. Requires data/raw/*.txt to be present (they are, per
# .gitignore). Downloads the MiniLM embedding model as a side effect too,
# so it's cached in the image layer instead of re-downloaded per container.
RUN python scripts/build_database.py

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
