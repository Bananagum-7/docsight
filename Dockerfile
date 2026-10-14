# Multi-stage build: the builder stage downloads docs and models and builds the
# index, so the runtime image starts instantly and needs no network at boot.
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install --no-cache-dir --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt
ENV PYTHONPATH=/install/lib/python3.11/site-packages HF_HOME=/app/models FASTEMBED_CACHE_PATH=/app/models/fastembed
COPY docsight ./docsight
COPY scripts ./scripts
RUN bash scripts/fetch_docs.sh && python scripts/build_index.py \
 && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /app/models /app/models
COPY --from=builder /app/data /app/data
COPY . .
ENV HF_HOME=/app/models FASTEMBED_CACHE_PATH=/app/models/fastembed HF_HUB_OFFLINE=1 PORT=7860
# Run as a non-root user (Hugging Face Spaces requires uid 1000)
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
# SERVICE=ui (default) runs Streamlit; SERVICE=api runs the FastAPI server
CMD ["bash", "-c", "if [ \"${SERVICE:-ui}\" = api ]; then uvicorn docsight.api:app --host 0.0.0.0 --port $PORT; else streamlit run app.py --server.port $PORT --server.address 0.0.0.0; fi"]
