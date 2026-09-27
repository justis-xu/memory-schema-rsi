#!/usr/bin/env python3
"""Post hoc two-call deletion-only check for the frozen nature-beauty case."""

import hashlib
import json
from pathlib import Path

from schema_rsi.config import get_settings
from schema_rsi.evaluation.answerer import Answerer
from schema_rsi.evaluation.prompts_official import build_official_answer_prompt
from schema_rsi.llm.chat import make_chat_client

from m2_poc_conv42_nature_beauty_answers import build


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis/m2_conv42_nature_beauty_drop_control_20260927.json"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    case, contexts, reference_date, _, _, _ = build()
    records = contexts["base"][:13]
    prompt = build_official_answer_prompt(case["question"], records, reference_date)
    settings = get_settings(str(ROOT / "config/locomo_zh.yaml"))
    client = make_chat_client(settings)
    client._client = client._client.with_options(max_retries=0)
    answerer = Answerer(settings, client)
    report = {"scope": "Post hoc deletion-only diagnostic after the six-call source/existing POC; one Chinese question, two answer calls, no judge",
              "case_id": case["case_id"], "question": case["question"],
              "arm": "base_first13_only", "context_ids": [r.id for r in records],
              "removed_ids": [r.id for r in contexts["base"][13:]],
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "prompt_chars": len(prompt), "model": settings.llm.model,
              "temperature": 0.0, "max_answer_calls": 2, "answer_calls": 0,
              "calls": [], "limits": ["Adaptive check performed after viewing the three-arm answers.",
                                      "Thirteen slots differ in length from all 15-slot arms.",
                                      "Two repeats do not characterize generation noise or broad effect."]}
    for index in range(1, 3):
        try:
            answer, info = answerer.answer(case["question"], records, temperature=0.0,
                                           reference_date=reference_date, benchmark="locomo")
            report["answer_calls"] += 1
            report["calls"].append({"index": index, "answer": answer,
                                    "usage": info.get("usage")})
        except Exception as exc:
            report["answer_calls"] += 1
            report["calls"].append({"index": index,
                                    "error_type": type(exc).__name__})
            report["stopped_early"] = True
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if report.get("stopped_early"):
            return
        print(index, answer[:180].replace("\n", " "), flush=True)
    report["completed"] = True
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
