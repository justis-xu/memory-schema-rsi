#!/usr/bin/env python3
"""Inventory every Chinese m2 report in the two local research repositories."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/m2_cross_machine_report_catalog_zh.md"


def title(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace("|", "\\|")
    return "（无一级标题）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-repo", type=Path,
                        default=ROOT.parent / "memory-prompt")
    args = parser.parse_args()
    prompt = args.prompt_repo.resolve()
    if not (prompt / "docs").is_dir():
        parser.error(f"missing peer docs: {prompt / 'docs'}")
    sections = []
    total = 0
    for repo, base in (("memory-schema-rsi", ROOT), ("memory-prompt", prompt)):
        paths = sorted(p for p in (base / "docs").glob("m2_*_zh.md") if p != OUT)
        total += len(paths)
        rows = []
        for path in paths:
            if repo == "memory-schema-rsi":
                url = path.name
            else:
                url = "https://github.com/justis-xu/memory-prompt/blob/master/docs/" + path.name
            rows.append(f"| [{path.name}]({url}) | {title(path)} |")
        sections.append(f"## {repo}（{len(paths)} 份）\n\n| 报告 | 标题 |\n| --- | --- |\n" + "\n".join(rows))
    body = (
        "# 中文记忆复盘：两仓阶段报告完整目录\n\n"
        "> 2026-09-30 的文件目录快照；由 `scripts/m2_build_cross_machine_report_catalog.py` "
        "读取两仓 `docs/m2_*_zh.md` 生成。只列阶段报告，不修改或删除原文件；原始机器账与复算脚本请从各报告进入。"
        "清单含两仓当时全部匹配文件，未来新增报告须重生成。\n\n"
        f"**合计 {total} 份报告。** 想先看结论，请回到[跨机最终索引](m2_cross_machine_final_index_zh.md)。\n\n"
        + "\n\n".join(sections) + "\n"
    )
    OUT.write_text(body)
    print(f"wrote {OUT}: {total} reports")


if __name__ == "__main__":
    main()
