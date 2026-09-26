"""Build a source-turn inventory for the fixed 12 Chinese carryover pairs.

Annotations below are manual judgments, not an automatic entailment detector.
Every cited turn is resolved against the frozen source packet before writing.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/analysis/m2_cross_session_carryover_packet_20260926.json"
LABELS = ROOT / "results/analysis/m2_cross_session_carryover_labels_20260926.json"
OUTPUT = ROOT / "results/analysis/m2_carryover_fact_inventory_20260926.json"
MERGE_OUTPUT = ROOT / "results/analysis/m2_carryover_merge_prototype_20260926.json"

# (claim, source dia_ids, old text contains claim, new text contains claim,
#  support kind). "qualified" means the wording needs narrowing before use.
FACTS = {
    "conv-41_051a88fcd77c": [
        ("约翰分享家人照片，家人激励他为改变而努力", ["D22:5", "D22:7"], True, True, "direct_self"),
        ("照片来自女儿萨拉生日时的旅行", ["D22:7"], True, True, "direct_self"),
        ("照片显示家人在秋天铁轨上合影", ["D22:5#caption"], True, True, "caption"),
        ("困难时约翰会看照片提醒自己为何努力", ["D22:9"], False, True, "direct_self"),
    ],
    "conv-44_0c8afd94ba05": [
        ("奥黛丽用瓶盖、纽扣、坏首饰等回收材料制作首饰", ["D22:5"], True, True, "direct_self"),
        ("她认为这体现创造力、可持续发展，每件作品有故事", ["D22:5"], True, True, "direct_self"),
    ],
    "conv-44_9b37baddbb91": [
        ("奥黛丽想过再养狗，但认为当前四只已够", ["D15:3"], True, True, "direct_self"),
        ("四只狗让她很忙，她要照顾好每一只", ["D15:3"], True, True, "direct_self"),
    ],
    "conv-44_b514edf1ce25": [
        ("奥黛丽做首饰先是爱好，后来开始出售", ["D22:7"], True, True, "direct_self"),
        ("她把部分利润捐给动物收容所", ["D22:7", "D22:9"], True, True, "direct_self"),
        ("她不能亲临收容所现场，但仍寻找行善方式", ["D22:10", "D22:11"], True, False, "self_confirmed_after_other"),
        ("首饰制作使用回收材料", ["D22:5"], False, True, "direct_self"),
    ],
    "conv-43_de89be96dda6": [
        ("蒂姆正在写奇幻小说，写作进展顺利", ["D15:3"], True, True, "direct_self"),
        ("他对此有些紧张但很兴奋，觉得努力得到回报", ["D15:3"], True, True, "direct_self"),
        ("写作给他快乐，并让他创造全新世界", ["D15:3"], False, True, "direct_self"),
    ],
    "conv-43_269540f8e596": [
        ("蒂姆加入志趣相投的环球旅行者群体", ["D27:1"], True, True, "direct_self"),
        ("认识他们、听旅行经历使他高兴", ["D27:1"], True, True, "direct_self"),
    ],
    "conv-26_444350391e64": [
        ("梅拉妮上月受伤，暂停用来自我表达和获得平静的陶艺", ["D17:8"], True, True, "direct_self"),
        ("暂停陶艺让梅拉妮难受，但她说自己还好", ["D17:10"], False, True, "direct_self"),
        ("她当时通过读书和画画打发时间", ["D17:10"], True, False, "direct_self"),
    ],
    "conv-43_c95588df7268": [
        ("蒂姆的写作灵感来自书、电影和现实经历", ["D15:5"], True, True, "direct_self"),
        ("阅读英国城堡给了他写作灵感", ["D15:5"], True, True, "direct_self"),
        ("某些作者也是他的灵感来源", ["D15:5"], True, False, "direct_self"),
        ("蒂姆梦想参观英国城堡", ["D15:1", "D15:3"], False, True, "direct_self"),
    ],
    "conv-43_d9a2fb3f97f7": [
        ("蒂姆从未参加过运动队", ["D9:5"], True, True, "direct_self"),
        ("他更喜欢阅读奇幻小说并沉浸于魔法世界", ["D9:5"], True, True, "direct_self"),
        ("这种探索不同世界的喜好也是他喜欢旅行的原因之一", ["D9:5"], True, False, "direct_self"),
    ],
    "conv-41_e94a25c4f01c": [
        ("约翰上一年参加过一次公路旅行", ["D11:3"], True, True, "direct_self"),
        ("旅途探索太平洋西北部海岸线和多个国家公园", ["D11:5"], True, True, "direct_self"),
        ("自然景色令他赞叹、深思，带来平静和灵感", ["D11:5", "D11:7", "D11:9"], True, True, "direct_self"),
        ("同行者是约翰的家人", [], True, False, "unsupported"),
        ("同行者是玛丽亚", [], False, True, "unsupported"),
    ],
    "conv-47_abc3c4fb4edf": [
        ("约翰在本地比赛得了第二名", ["D12:4"], True, True, "direct_self"),
        ("虽未夺冠，他获得奖金和奖杯", ["D12:6", "D12:8"], True, True, "direct_self"),
        ("他为努力得到回报感到激动", ["D12:6"], True, True, "direct_self"),
    ],
    "conv-47_12946f1346c5": [
        ("詹姆斯带狗远足并探索林中小径", ["D7:2", "D7:4#caption"], True, True, "direct_self_and_caption"),
        ("两只狗喜欢小径和风景，詹姆斯觉得好玩", ["D7:4", "D7:4#caption"], True, False, "direct_self_and_caption"),
        ("他喜欢脚下树叶声，散步使他理清思绪、放松", ["D7:8"], False, True, "direct_self"),
    ],
}


def main() -> None:
    packet = json.loads(PACKET.read_text())
    labels = json.loads(LABELS.read_text())
    selected = {x["pair_id"] for x in labels["逐对"] if x["分类"] == "旧事实写入较晚场次_未见新来源"}
    assert selected == set(FACTS), "Annotation and fixed-sample IDs differ"
    output = []
    for pair in packet["pairs"]:
        if pair["pair_id"] not in selected:
            continue
        old, new = sorted((pair["left"], pair["right"]), key=lambda m: int(m["metadata"]["session_id"].split("_")[-1]))
        old_session, new_session = old["metadata"]["session_id"], new["metadata"]["session_id"]
        turns = {t["dia_id"]: t for t in pair["source_sessions"][old_session]["turns"]}
        fact_rows = []
        for index, (claim, refs, in_old, in_new, support) in enumerate(FACTS[pair["pair_id"]], 1):
            evidence = []
            for ref in refs:
                dia_id, _, caption = ref.partition("#")
                turn = turns[dia_id]
                quote = turn["blip_caption"] if caption == "caption" else turn["text"]
                assert quote, ref
                evidence.append({"session_id": old_session, "dia_id": dia_id, "speaker": turn["speaker"], "channel": caption or "text", "quote": quote})
            assert (support == "unsupported") == (not evidence)
            fact_rows.append({
                "fact_id": f"{pair['pair_id']}:{index}", "claim_zh": claim,
                "in_older_memory": in_old, "in_later_memory": in_new,
                "support": support, "source_turns": evidence,
                "decision": "retract" if support == "unsupported" else "retain_with_original_source",
            })
        output.append({
            "pair_id": pair["pair_id"],
            "older_memory": {"id": old["id"], "session_id": old_session, "content": old["content"]},
            "later_memory": {"id": new["id"], "session_id": new_session, "content": new["content"]},
            "later_session_turn_count": len(pair["source_sessions"][new_session]["turns"]),
            "facts": fact_rows,
        })
    assert len(output) == 12
    counts = {
        "pairs": len(output),
        "facts": sum(len(p["facts"]) for p in output),
        "supported": sum(f["support"] != "unsupported" for p in output for f in p["facts"]),
        "unsupported": sum(f["support"] == "unsupported" for p in output for f in p["facts"]),
        "supported_only_in_later_memory_from_older_source": sum(f["support"] != "unsupported" and f["in_later_memory"] and not f["in_older_memory"] for p in output for f in p["facts"]),
        "supported_only_in_older_memory": sum(f["support"] != "unsupported" and f["in_older_memory"] and not f["in_later_memory"] for p in output for f in p["facts"]),
    }
    payload = {"scope": "Fixed 12 carryover pairs; manual fact decomposition; source quotes resolved from frozen packet", "counts": counts, "pairs": output, "limits": ["事实划分和新旧文本覆盖由人工标注，不是自动蕴含判断", "源会话包含图片说明而非原图复核", "不能从记忆文本与源话反推历史 LLM 生成原因或上线效果"]}
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    # A deterministic, reviewable merge contract. This does not claim to solve
    # extraction automatically: the accepted facts are the annotated input.
    merged = []
    for pair in output:
        accepted = [f for f in pair["facts"] if f["support"] != "unsupported"]
        rejected = [f for f in pair["facts"] if f["support"] == "unsupported"]
        assert all(f["source_turns"] and f["decision"] == "retain_with_original_source" for f in accepted)
        assert all(not f["source_turns"] and f["decision"] == "retract" for f in rejected)
        assert all(e["session_id"] == pair["older_memory"]["session_id"] for f in accepted for e in f["source_turns"])
        assert {f["fact_id"] for f in accepted} == {f["fact_id"] for f in pair["facts"] if f["support"] != "unsupported"}
        merged.append({
            "canonical_memory_id": pair["older_memory"]["id"],
            "alias_memory_id_to_retire": pair["later_memory"]["id"],
            "origin_session_id": pair["older_memory"]["session_id"],
            "accepted_fact_ids": [f["fact_id"] for f in accepted],
            "retracted_fact_ids": [f["fact_id"] for f in rejected],
            "source_turn_ids": sorted({e["dia_id"] for f in accepted for e in f["source_turns"]}),
            "accepted_claims_zh": [f["claim_zh"] for f in accepted],
        })
    assert len({m["canonical_memory_id"] for m in merged}) == 12
    assert len({m["alias_memory_id_to_retire"] for m in merged}) == 12
    assert sum(len(m["accepted_fact_ids"]) for m in merged) == counts["supported"]
    assert sum(len(m["retracted_fact_ids"]) for m in merged) == counts["unsupported"]
    MERGE_OUTPUT.write_text(json.dumps({
        "scope": "Offline acceptance prototype from manually annotated 12-pair fact inventory; no vector DB write or model call",
        "checks": {"all_supported_facts_retained": True, "all_unsupported_facts_retracted": True,
                   "all_sources_stay_in_older_session": True, "one_canonical_memory_id_per_pair": True},
        "merges": merged,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
