#!/usr/bin/env python3
"""Small Chinese RSI verifier check for source-person attribution, without answer calls."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/schema_rsi")]

from schema_rsi.config import get_settings  # noqa: E402
from schema_rsi.llm.chat import make_chat_client  # noqa: E402
from schema_rsi.memory.base import MemoryRecord  # noqa: E402
from schema_rsi_lab.verify import verify_answer  # noqa: E402


DATASET = Path("/Users/xu/git/memory-prompt/eval-datasets/locomo-zh/locomo10_zh.json")
OUT = ROOT / "results/analysis/m2_zh_rsi_relation_verifier_poc_20260927.json"
ORDER = ("answer_only", "pair", "pair", "answer_only")
CASES = (
    ("locomo_conv-26_qa111", ("D8:5", "D8:6"), "梅尔和孩子们一起画画。", True),
    ("locomo_conv-48_qa233", ("D28:6", "D28:7"),
     "乔琳和妈妈的老朋友们一起回忆往事、翻看照片。", False),
    ("locomo_conv-26_qa157", ("D4:2", "D4:3"),
     "梅拉妮的项链象征爱、信念和力量。", False),
)


class LoggedClient:
    def __init__(self, client):
        self.client = client
        self.raw = None
        self.usage = None
        self.prompt_sha256 = None
        self.prompt_chars = None

    def complete(self, **kwargs):
        self.prompt_sha256 = hashlib.sha256(kwargs["user"].encode()).hexdigest()
        self.prompt_chars = len(kwargs["user"])
        self.raw, self.usage = self.client.complete(**kwargs)
        return self.raw, self.usage


def build():
    data = {s["sample_id"]: s for s in json.loads(DATASET.read_text())}
    cases = []
    for cid, ids, proposed, expected in CASES:
        conv, index = cid.removeprefix("locomo_").split("_qa")
        sample = data[conv]
        question = sample["qa"][int(index)]["question"]
        found = {}
        for sid, rows in sample["conversation"].items():
            if not sid.startswith("session_") or sid.endswith("_date_time"):
                continue
            for row in rows:
                if row["dia_id"] in ids:
                    found[row["dia_id"]] = (sid, row)
        assert set(found) == set(ids)
        records = []
        for dia in ids:
            sid, turn = found[dia]
            date = sample["conversation"][sid + "_date_time"]
            records.append(MemoryRecord(id=f"T:{conv}|{dia}",
                                        content=f"Source conversation on {date}: "
                                                f"{turn['speaker']}：{turn['text']}",
                                        metadata={"user_id": conv, "session_id": sid,
                                                  "dia_id": dia}))
        cases.append({"case_id": cid, "question": question,
                      "proposed_answer": proposed,
                      "expected_source_relation_supported": expected,
                      "question_turn_id": ids[0], "answer_turn_id": ids[1],
                      "contexts": {"answer_only": records[1:], "pair": records}})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cases = build()
    prepared = {
        "scope": "Three previously source-audited Chinese cases; RSI experimental verifier; answer turn alone vs question+answer source pair; no answerer or judge calls",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "order_each_case": list(ORDER), "max_verifier_calls": 12,
        "verifier_calls": 0,
        "cases": [{"case_id": c["case_id"], "question": c["question"],
                   "proposed_answer": c["proposed_answer"],
                   "expected_source_relation_supported": c["expected_source_relation_supported"],
                   "question_turn_id": c["question_turn_id"],
                   "answer_turn_id": c["answer_turn_id"],
                   "contexts": {arm: [{"id": r.id, "content": r.content}
                                      for r in records]
                                for arm, records in c["contexts"].items()},
                   "calls": []} for c in cases],
        "limits": ["These cases were chosen after manual source inspection and cannot estimate verifier accuracy.",
                   "Explicit speaker labels in source records are a favorable condition, not proof the production SourceGraph will surface the same records.",
                   "Two repeated verifier outputs per arm show local variability only; there is no answerer or downstream RSI acceptance run.",
                   "The expected relation labels are post hoc source judgments and are not sent to the verifier."],
    }
    if not args.run:
        if OUT.exists():
            raise SystemExit(f"refusing to overwrite: {OUT}")
        OUT.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n")
        print("prepared", len(cases), "cases; max calls", prepared["max_verifier_calls"])
        return
    if not OUT.exists():
        raise SystemExit("prepare first")
    result = json.loads(OUT.read_text())
    for key in ("dataset_sha256", "order_each_case", "max_verifier_calls"):
        assert result[key] == prepared[key]
    assert result["verifier_calls"] == 0
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    result["model"] = settings.llm.model
    for index, case in enumerate(cases):
        row = result["cases"][index]
        assert row["case_id"] == case["case_id"]
        for arm in ORDER:
            logged = LoggedClient(client)
            try:
                verdict = verify_answer(logged, case["question"],
                                        case["proposed_answer"], case["contexts"][arm])
                call = {"arm": arm, "status": verdict.status,
                        "reason": verdict.reason, "search_query": verdict.search_query,
                        "support_ids": list(verdict.support_ids),
                        "raw_response": logged.raw, "usage": logged.usage,
                        "prompt_sha256": logged.prompt_sha256,
                        "prompt_chars": logged.prompt_chars}
            except Exception as exc:
                call = {"arm": arm, "error_type": type(exc).__name__}
                result["stopped_early"] = True
            result["verifier_calls"] += 1
            row["calls"].append(call)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            if result.get("stopped_early"):
                print("stopped after", result["verifier_calls"], "call(s)")
                return
            print(case["case_id"], arm, verdict.status, list(verdict.support_ids), flush=True)
    result["completed"] = True
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
