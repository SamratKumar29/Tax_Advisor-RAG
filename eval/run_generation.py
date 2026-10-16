"""
Generation smoke eval: run selected gold questions end-to-end through
src.generation.chain.answer with the configured LLM and record, per question,
the answer, model used, latency, token usage and the deterministic citation
check. Answer correctness is NOT auto-scored here; results are for manual
review against the criteria in tests/test_questions.md.

Costs one LLM call per question. Usage:
    python -m eval.run_generation --ids q1_slabs,q2_sec123 --name gen_smoke
"""

import argparse
import json
import time

from eval.run_eval import RESULTS_DIR, load_gold
from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from src.generation.chain import answer
from src.llm_client import LLMUnavailableError

DEFAULT_IDS = "q1_slabs,q2_sec123,q3_hra,q4_regime,q5_fy2425,q6_80c_map"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--name", default="gen_smoke")
    args = p.parse_args(argv)

    if not LLM_API_KEY or not LLM_MODEL:
        raise SystemExit("Set LLM_API_KEY and LLM_MODEL in .env to run the generation eval.")

    wanted = args.ids.split(",")
    gold = {g["id"]: g for g in load_gold()}

    rows = []
    for qid in wanted:
        q = gold[qid]["question"]
        t0 = time.perf_counter()
        try:
            r = answer(q, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)  # FALLBACK_MODELS apply by default
            row = {
                "id": qid, "question": q, "ok": True,
                "latency_s": round(time.perf_counter() - t0, 1),
                "model_used": r["model_used"], "usage": r["usage"],
                "answer": r["answer"], "citation_check": r["citation_check"],
                "retrieved": [f"{c['source_doc']} | {c['section_ref']}" for c in r["retrieved_chunks"]],
            }
        except LLMUnavailableError as e:
            row = {"id": qid, "question": q, "ok": False, "error": str(e)}
        rows.append(row)
        cc = row.get("citation_check", {})
        print(f"{qid}: ok={row['ok']} model={row.get('model_used')} "
              f"markers={cc.get('markers_used')} unsupported={len(cc.get('unsupported', []))}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
