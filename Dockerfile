# Indian Tax Advisor RAG — Streamlit app.
#   docker build -t tax-advisor .
#   docker run --rm -p 8501:8501 --env-file .env -v "$PWD/data/user:/app/data/user" tax-advisor
# The committed indexes are used as-is; the embedding model is downloaded at build time
# so the container starts without network access to Hugging Face.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache QUERY_LOG_PATH=""
WORKDIR /app

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY src/ src/
RUN python -c "from sentence_transformers import SentenceTransformer as S; from src.config import EMBEDDING_MODEL as M; S(M)"

COPY app/ app/
COPY indexes/ indexes/
COPY data/processed/ data/processed/
COPY data/sources.json data/sources.json

RUN useradd --create-home appuser && mkdir -p data/user && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
