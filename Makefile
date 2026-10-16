PY ?= venv/bin/python

.PHONY: install install-dev data index upgrade-chunks chunk test lint eval eval-gen app check

install:            ## Runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:        ## Runtime + test/lint tooling
	$(PY) -m pip install -r requirements-dev.txt

data:               ## Verify (and fetch direct-URL) raw sources against data/sources.json
	$(PY) -m src.ingest.download_sources --download

chunk: data         ## Re-chunk data/raw/ → data/processed/chunks.jsonl (needs the PDFs)
	$(PY) -m src.ingest.chunk

upgrade-chunks:     ## Re-apply normalisation/section refs/tags to existing chunks.jsonl (no PDFs needed)
	$(PY) -m src.ingest.chunk --upgrade-existing

index:              ## Build Chroma + BM25 indexes and manifest from chunks.jsonl
	$(PY) -m src.ingest.build_index

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

eval:               ## Retrieval eval (no API key needed) → eval/results/latest.{json,md}
	$(PY) -m eval.run_eval --mode retrieval --configs dense,bm25,hybrid --name latest

eval-gen:           ## Generation smoke eval (6 LLM calls, needs .env) → eval/results/gen_smoke.json
	$(PY) -m eval.run_generation --name gen_smoke

app:
	$(PY) -m streamlit run app/streamlit_app.py

check: lint test    ## What CI runs
