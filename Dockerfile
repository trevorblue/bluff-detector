# Multi-stage build: compilers and pip caches stay in the builder; only the venv ships.
# Default image = CPU serving with whatever backend the config names (dev_fake in CI).
# For a real LLM, build with: docker build --build-arg EXTRAS=llm -t bluff-detector:llm .

# ---------- stage 1: builder ----------
FROM python:3.11-slim AS builder
ARG EXTRAS=""
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PIP_DISABLE_PIP_VERSION_CHECK=1
# Lock file first, code last, so a code edit does not reinstall every dependency.
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . \
 && if [ -n "$EXTRAS" ]; then pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu ".[${EXTRAS}]"; fi

# ---------- stage 2: runtime ----------
FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    BLUFF_CONFIG=configs/variants/dev_fake.yaml
WORKDIR /srv
COPY app/ app/
COPY configs/ configs/
# The trained detector is baked in: the image digest is the release.
COPY artefacts/ artefacts/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
