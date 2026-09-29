"""Two answer-only calls on a frozen current Chinese 15-memory context."""

import hashlib
import json
from pathlib import Path

from schema_rsi.benchmarks.base import parse_session_date
from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client
from schema_rsi.memory.base import MemoryRecord


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL = ROOT / "results/analysis/m2_current_mother_game_relation_20260929.json"
DATA = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_current_mother_game_answer_20260929.json"
TARGET = "f1455645-db16-42df-a737-cb52619de275"


def save(report):
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite previous calls: {OUT}")
    retrieval = json.loads(RETRIEVAL.read_text())
    assert retrieval["target_final_rank"] == 2 and retrieval["logical_source_unchanged"]
    assert not retrieval.get("stopped_after_failure")
    context = [MemoryRecord(id=m["id"], content=m["content"], metadata={
        "session_id": m.get("session_id"), "session_date": m.get("session_date"),
        "user_id": retrieval["user_id"]}) for m in retrieval["rerank_top15"]]
    assert len(context) == 15 and len({m.id for m in context}) == 15
    assert context[1].id == TARGET
    source = next(s for s in json.loads(DATA.read_text()) if s["sample_id"] == "conv-48")
    last_session = max(int(k.split("_")[1]) for k in source["conversation"] if k.endswith("_date_time"))
    reference_date = parse_session_date(source["conversation"][f"session_{last_session}_date_time"])
    assert reference_date and retrieval["question"] == source["qa"][172]["question"]
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    assert settings.llm.model == "glm-5.3-flash"
    prompt = build_official_answer_prompt(retrieval["question"], context, reference_date)
    report = {
        "scope": "Current pure-Chinese fixed no-graph top15; two repeated answer calls, no judge, no retrieval or graph calls.",
        "retrieval_artifact": str(RETRIEVAL.relative_to(ROOT)),
        "retrieval_sha256": hashlib.sha256(RETRIEVAL.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "case_id": retrieval["case_id"], "question": retrieval["question"],
        "gold_for_posthoc_review_only": source["qa"][172]["answer"],
        "context_ids": [m.id for m in context], "target_rank": 2,
        "reference_date": reference_date, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": settings.llm.model, "temperature": 0.0,
        "max_answer_calls": 2, "max_completion_tokens_per_call": 1500,
        "calls_sent": 0, "calls": [],
        "limits": [
            "One source-selected question and two generations cannot estimate system accuracy or stable net benefit.",
            "Wii appears in source image caption but not the frozen final memory text; an answer can still copy it from the question.",
            "A correct game name does not prove all answer clauses have source support.",
        ],
    }
    save(report)
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    for index in (1, 2):
        report["calls_sent"] = index
        try:
            answer, info = answerer.answer(retrieval["question"], context,
                                           temperature=0.0, reference_date=reference_date,
                                           benchmark="locomo")
            if not answer:
                raise RuntimeError("empty answer")
            call = {"index": index, "answer": answer, "usage": info["usage"]}
        except Exception as exc:
            call = {"index": index, "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None)}
            report["calls"].append(call)
            report["stopped_after_failure"] = True
            save(report)
            print(call, flush=True)
            return
        report["calls"].append(call)
        save(report)
        print(index, answer[:250].replace("\n", " "), flush=True)
    report["stopped_after_failure"] = False
    save(report)


if __name__ == "__main__":
    main()
