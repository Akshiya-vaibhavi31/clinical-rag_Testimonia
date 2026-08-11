# python:3.11-slim chosen over a newer version for broad ML-package wheel
# compatibility (torch, sentence-transformers, chromadb) — newer Python
# versions sometimes lack prebuilt wheels for these on release day, which
# is exactly the kind of "works on my machine, not in Docker" problem this
# phase is meant to eliminate.
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffering stdout — makes
# container logs show up immediately rather than being buffered, which
# matters for actually seeing what's happening via `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies BEFORE copying the rest of the code. Docker caches
# each layer — as long as requirements.txt doesn't change, this layer is
# reused on every rebuild instead of re-installing everything (torch and
# sentence-transformers are large; this cache saves real time on every
# code-only change).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY src/ ./src/
COPY frontend/ ./frontend/

# NOTE: data/ is intentionally NOT copied into the image. It's large,
# regenerable, and environment-specific (your own corpus, vector DB, and
# SQLite database) — it's mounted in via docker-compose's volume instead,
# consistent with the same "secrets and data don't belong in the image"
# principle from Phase 20's security review.

EXPOSE 8000

# A container-native health check — Docker itself can now tell whether the
# app inside is actually responding, not just whether the process exists.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
