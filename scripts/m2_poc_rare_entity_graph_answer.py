#!/usr/bin/env python3
"""Four-call answer check after automatic rare-entity graph candidate selection."""

import hashlib
import json
import sqlite3
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord


ROOT = Path(__file__).resolve().parents[1]
RERANK = ROOT / "results/analysis/m2_rare_entity_fused_rerank_20260927.json"
GRAPH = ROOT / "results/analysis/m2_rare_entity_graph_priority_20260927.json"
RETRIEVAL = ROOT / "results/analysis/m2_current_zh_retrieval_probe_20260926.json"
PRIOR = ROOT / "results/analysis/m2_current_zh_answer_poc_20260926.json"
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_rare_entity_graph_answer_20260927.json"
CASE = "locomo_conv-43_qa25"
ORDER = ("baseline", "rare_first", "rare_first", "baseline")
JOHN = "1ffbf4d6-dbe0-4dd9-8ace-54850bbd24cc"
TIM = "219b4c97-f2af-4734-a4f4-1d93ebe19f99"


def file_sha():
    return hashlib.sha256(DB.read_bytes()).hexdigest()


def metadata_for(ids):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        out = {}
        for mid in ids:
            row = con.execute("select id from embeddings where embedding_id=?", (mid,)).fetchone()
            assert row is not None, mid
            out[mid] = {k: v for k, v in con.execute(
                "select key,string_value from embedding_metadata where id=? and key in ('data','session_id','session_date','user_id')",
                (row[0],))}
        return out
    finally:
        con.close()


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    replay = json.loads(RERANK.read_text())
    rows = {(r["case_id"], r["order"], r["policy"]): r for r in replay["runs"]}
    contexts = {arm: rows[CASE, "reversed", arm] for arm in ("baseline", "rare_first")}
    assert not contexts["baseline"]["john_in_top15"]
    assert contexts["rare_first"]["john_in_top15"]
    ids = {m["id"] for r in contexts.values() for m in r["top15"]}
    meta = metadata_for(ids)
    frozen_graph = json.loads(GRAPH.read_text())["runs"]["reversed"][CASE]
    frozen_retrieval = next(c for c in json.loads(RETRIEVAL.read_text())["cases"]
                            if c["case_id"] == CASE)
    frozen_contents = {m["id"]: m["content"] for m in frozen_retrieval["reranked"]}
    frozen_contents.update({m["id"]: m["content"]
                            for arm in ("baseline_top30", "rare_first_top30")
                            for m in frozen_graph[arm]})
    assert all(meta[mid]["data"] == frozen_contents[mid] for mid in ids)
    records = {}
    for arm, r in contexts.items():
        records[arm] = [MemoryRecord(id=m["id"], content=meta[m["id"]]["data"],
                                      metadata={k: meta[m["id"]].get(k)
                                                for k in ("user_id", "session_id", "session_date")})
                        for m in r["top15"]]
    assert all(len(v) == 15 for v in records.values())
    assert all(rec.metadata["user_id"] == "zhfull:locomo:conv-43"
               for arm in records.values() for rec in arm)
    assert TIM in {r.id for r in records["baseline"]} & {r.id for r in records["rare_first"]}
    assert JOHN not in {r.id for r in records["baseline"]}
    assert JOHN in {r.id for r in records["rare_first"]}
    prior = json.loads(PRIOR.read_text())
    previous = next(c for c in prior["cases"] if c["case_id"] == CASE)
    question = contexts["baseline"]["question"]
    reference_date = previous["reference_date"]
    prompts = {arm: build_official_answer_prompt(question, recs, reference_date)
               for arm, recs in records.items()}
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    baseline_ids = {r.id for r in records["baseline"]}
    rare_ids = {r.id for r in records["rare_first"]}
    report = {"scope": "One frozen current Chinese two-person question; automatic cached graph rare-bridge candidates + current unified rerank, ABBA four answer calls max, no judge",
              "case_id": CASE, "question": question,
              "source_rerank": str(RERANK.relative_to(ROOT)),
              "rerank_sha256": hashlib.sha256(RERANK.read_bytes()).hexdigest(),
              "model": settings.llm.model, "temperature": 0.0,
              "order": ORDER, "max_answer_calls": 4, "answer_calls": 0,
              "reference_date": reference_date,
              "context_ids": {arm: [r.id for r in recs] for arm, recs in records.items()},
              "entered": sorted(rare_ids - baseline_ids), "exited": sorted(baseline_ids - rare_ids),
              "prompt_sha256": {arm: hashlib.sha256(prompt.encode()).hexdigest()
                                for arm, prompt in prompts.items()},
              "vector_db_sha256_before": file_sha(), "calls": [],
              "limits": ["One current Chinese case and two repetitions per arm cannot estimate Graph overall benefit",
                         "Both context membership and ordering follow current rerank; effects are not separately identified",
                         "In-process graph cache path and rerank do not verify real HugeGraph edge order"]}
    for index, arm in enumerate(ORDER, 1):
        try:
            answer, info = answerer.answer(question, records[arm], temperature=0.0,
                                           reference_date=reference_date, benchmark="locomo")
            report["answer_calls"] += 1
            report["calls"].append({"index": index, "arm": arm, "answer": answer,
                                    "usage": info.get("usage")})
        except Exception as exc:
            report["answer_calls"] += 1
            report["calls"].append({"index": index, "arm": arm,
                                    "error_type": type(exc).__name__})
            report["stopped_early"] = True
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if report.get("stopped_early"):
            print("stopped after", report["answer_calls"], "answer call(s)")
            return
        print(index, arm, answer[:170].replace("\n", " "), flush=True)
    report["vector_db_sha256_after"] = file_sha()
    report["vector_db_unchanged"] = report["vector_db_sha256_before"] == report["vector_db_sha256_after"]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("completed", report["answer_calls"], "calls; db unchanged", report["vector_db_unchanged"])


if __name__ == "__main__":
    main()
