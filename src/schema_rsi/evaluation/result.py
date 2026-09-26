"""EvaluationResult 结构与 JSONL 输出。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from schema_rsi.memory.base import MemoryRecord


def _memories_to_json(memories: list[MemoryRecord]) -> list[dict]:
    return [
        {"id": m.id, "content": m.content, "score": m.score, "metadata": m.metadata}
        for m in memories
    ]


@dataclass
class EvaluationResult:
    case_id: str
    benchmark: str
    question: str
    expected_answer: str
    predicted_answer: str

    retrieved_memories: list = field(default_factory=list)  # list[MemoryRecord]
    graph_memories: list = field(default_factory=list)      # list[dict]（图邻居内容）

    metrics: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)             # 各阶段秒数
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieved_memories"] = _memories_to_json(self.retrieved_memories)
        d["graph_memories"] = self.graph_memories
        return d

    def write_jsonl(self, path: str | Path) -> None:
        append_jsonl([self], path)


def append_jsonl(results: list[EvaluationResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")
    return path


def summarize(results: list[EvaluationResult]) -> dict:
    """对一组结果做平均汇总（评测骨架的最小汇总，不追求论文级指标）。

    - contains 是严格归一化子串（日期词序不同即 miss，偏保守）
    - f1_hit_rate 以 token_f1>=0.5 认定"答对"，更接近主观判卷
    """
    if not results:
        return {"cases": 0}
    n = len(results)
    avg_f1 = sum(r.metrics.get("token_f1", 0.0) for r in results) / n
    hit = sum(1 for r in results if r.metrics.get("contains")) / n
    f1_hit = sum(1 for r in results if r.metrics.get("token_f1", 0.0) >= 0.5) / n
    avg_answer_s = sum(r.latency.get("answer", 0.0) for r in results) / n
    out = {
        "cases": n,
        "avg_token_f1": round(avg_f1, 4),
        "f1_hit_rate": round(f1_hit, 4),
        "contains_rate": round(hit, 4),
        "avg_answer_latency_s": round(avg_answer_s, 3),
    }
    judged = [r for r in results if "judge_correct" in r.metrics]
    if judged:
        out["judge_accuracy"] = round(
            sum(1 for r in judged if r.metrics["judge_correct"]) / len(judged), 4
        )
        voted = [r for r in judged if r.metrics.get("judge_votes")]
        if voted:
            # pass@N：任一采样被 judge 判对（最终答案按最优样本选取时的口径）
            out["pass_at_n_rate"] = round(
                sum(1 for r in voted if any(r.metrics["judge_votes"])) / len(voted), 4
            )
    return out
