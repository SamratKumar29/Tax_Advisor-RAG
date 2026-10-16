"""
Retrieval evaluation harness.

Usage:
    python -m eval.run_eval --mode retrieval --configs dense,bm25,hybrid --name baseline_pre_changes

Relevance is a PROXY (see eval/README.md): a retrieved chunk is relevant when its
source_doc is in the item's `source_docs` AND its text contains any `must_contain_any`
keyword (case-insensitive). Only `answerable: true` items are scored.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD_PATH = ROOT / "eval" / "gold.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
K_MAX = 10


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(chunk: dict, item: dict) -> bool:
    rel = item["relevant"]
    if chunk["metadata"].get("source_doc") not in rel["source_docs"]:
        return False
    text = chunk["text"].lower()
    if not all(k.lower() in text for k in rel.get("must_contain_all", [])):
        return False
    return any(k.lower() in text for k in rel["must_contain_any"])


def set_dense_backend(backend: str) -> None:
    """Select the dense backend used by dense_retriever (and therefore hybrid) in this process."""
    from src.retrieval import dense_retriever

    dense_retriever.DENSE_BACKEND = backend


def get_search_fn(config: str):
    if config == "dense":
        from src.retrieval import dense_retriever
        return lambda q: dense_retriever.search(q, top_k=K_MAX)
    if config == "bm25":
        from src.retrieval import bm25_retriever
        return lambda q: bm25_retriever.search(q, top_k=K_MAX)
    if config == "hybrid":
        from src.retrieval import hybrid
        return lambda q: hybrid.search(q, top_n=K_MAX)
    if config == "hybrid_llm":  # LLM keyword rewrite for BM25: makes one LLM call per item (needs .env)
        from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
        from src.retrieval import hybrid
        from src.retrieval.llm_query import rewrite_query
        return lambda q: hybrid.search(
            q, top_n=K_MAX, bm25_extra=rewrite_query(q, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL))
    raise ValueError(f"Unknown config: {config}")


def score_item(results: list[dict], item: dict) -> dict:
    flags = [is_relevant(r, item) for r in results[:K_MAX]]
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    return {
        "hit@5": float(any(flags[:5])),
        "hit@10": float(any(flags[:10])),
        "mrr@10": 1.0 / first if first else 0.0,
        "first_relevant_rank": first,
        "top_ids": [r["chunk_id"] for r in results[:K_MAX]],
    }


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "recall@5": round(sum(r["hit@5"] for r in rows) / n, 4) if n else 0.0,
        "recall@10": round(sum(r["hit@10"] for r in rows) / n, 4) if n else 0.0,
        "mrr@10": round(sum(r["mrr@10"] for r in rows) / n, 4) if n else 0.0,
    }


def run_retrieval(configs: list[str], gold: list[dict]) -> dict:
    items = [g for g in gold if g.get("answerable", True)]
    out = {"mode": "retrieval", "n_items": len(items), "configs": {}}
    for cfg in configs:
        search = get_search_fn(cfg)
        search(items[0]["question"])  # warm-up (model/index load) excluded from latency
        per_item, latencies = {}, []
        for item in items:
            t0 = time.perf_counter()
            results = search(item["question"])
            latencies.append((time.perf_counter() - t0) * 1000)
            per_item[item["id"]] = {"category": item["category"], **score_item(results, item)}
        rows = list(per_item.values())
        by_cat = {}
        for cat in sorted({r["category"] for r in rows}):
            by_cat[cat] = _agg([r for r in rows if r["category"] == cat])
        out["configs"][cfg] = {
            "overall": _agg(rows),
            "by_category": by_cat,
            # Latency is environment-dependent and excluded from determinism checks.
            "latency_ms": {
                "p50": round(statistics.median(latencies), 1),
                "max": round(max(latencies), 1),
            },
            "per_item": per_item,
        }
    return out


def to_markdown(res: dict, name: str) -> str:
    lines = [f"# Retrieval eval: {name}", "",
             f"Items scored: {res['n_items']} (answerable only). Relevance = proxy (source + keyword). "
             f"Dense backend: {res.get('dense_backend', 'exact')}.", "",
             "| config | recall@5 | recall@10 | MRR@10 | p50 ms |", "|---|---|---|---|---|"]
    for cfg, r in res["configs"].items():
        o = r["overall"]
        lines.append(f"| {cfg} | {o['recall@5']:.3f} | {o['recall@10']:.3f} | {o['mrr@10']:.3f} "
                     f"| {r['latency_ms']['p50']} |")
    cats = sorted({c for r in res["configs"].values() for c in r["by_category"]})
    lines += ["", "## Recall@5 by category", "",
              "| category | n | " + " | ".join(res["configs"]) + " |",
              "|---|---|" + "---|" * len(res["configs"])]
    for cat in cats:
        first = next(iter(res["configs"].values()))
        vals = [f"{r['by_category'][cat]['recall@5']:.2f}" for r in res["configs"].values()]
        lines.append(f"| {cat} | {first['by_category'][cat]['n']} | " + " | ".join(vals) + " |")
    lines += ["", "## Misses at 10 (any config)", ""]
    for cfg, r in res["configs"].items():
        miss = [i for i, v in r["per_item"].items() if not v["hit@10"]]
        lines.append(f"- **{cfg}**: {', '.join(miss) if miss else 'none'}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="retrieval", choices=["retrieval"])
    p.add_argument("--configs", default="dense,bm25,hybrid")
    p.add_argument("--name", default="latest")
    p.add_argument("--gold", default=str(GOLD_PATH))
    p.add_argument("--dense-backend", default="exact", choices=["exact", "hnsw"],
                   help="exact = deterministic cosine over all vectors (app default); hnsw = Chroma ANN")
    args = p.parse_args(argv)
    set_dense_backend(args.dense_backend)
    # Measure the committed base corpus only, never a developer's uploaded documents.
    import src.config
    src.config.USER_DOCS_DIR = str(ROOT / "eval" / ".no_user_docs")

    res = run_retrieval([c.strip() for c in args.configs.split(",") if c.strip()], load_gold(Path(args.gold)))
    res["dense_backend"] = args.dense_backend
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(
        json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    (RESULTS_DIR / f"{args.name}.md").write_text(to_markdown(res, args.name), encoding="utf-8")
    for cfg, r in res["configs"].items():
        print(cfg, r["overall"])


if __name__ == "__main__":
    main()
