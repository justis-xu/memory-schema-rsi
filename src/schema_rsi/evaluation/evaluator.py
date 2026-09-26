"""基础确定性指标（Phase 0 不做 LLM 判题，JUDGE_* 槽位已预留）。

- contains: 标准答案（归一化后）是否作为子串出现在预测中 —— LoCoMo 官方评测常用的宽松命中
- token_f1: 词级 F1
- exact: 归一化后完全相等
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def token_set(text: str) -> list[str]:
    return normalize(text).split()


def token_f1(predicted: str, reference: str) -> float:
    pred, ref = token_set(predicted), token_set(reference)
    if not pred or not ref:
        return 1.0 if normalize(predicted) == normalize(reference) else 0.0
    common: set[str] = set(pred) & set(ref)
    if not common:
        return 0.0
    # 多重集合意义上的粗略 F1（骨架够用）
    precision = len(common) / len(set(pred))
    recall = len(common) / len(set(ref))
    return 2 * precision * recall / (precision + recall)


def evaluate_answers(expected: str, predicted: str) -> dict:
    n_exp, n_pred = normalize(expected), normalize(predicted)
    return {
        "token_f1": round(token_f1(predicted, expected), 4),
        "contains": bool(n_exp) and n_exp in n_pred,
        "exact": bool(n_exp) and n_exp == n_pred,
    }
