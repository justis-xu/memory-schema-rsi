"""Inventory source provenance fields in current-ID Chinese S1 cache entries."""

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/graph_zh/structured_cache.json"
DB = ROOT / "data/vector_store_zh/chroma/chroma.sqlite3"
OUT = ROOT / "results/analysis/m2_cached_relation_provenance_20260929.json"
KINDS = ("entities", "events", "preferences", "relationships")
SOURCE_FIELDS = {"source_turn_ids", "source_ref", "source_channel", "support_type",
                 "support_strength", "source_date", "dia_id", "source_session_id"}


def source_rows():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = conn.execute("""SELECT e.embedding_id,
            MAX(CASE WHEN m.key='data' THEN m.string_value END),
            MAX(CASE WHEN m.key='user_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_id' THEN m.string_value END),
            MAX(CASE WHEN m.key='session_date' THEN m.string_value END)
            FROM embeddings e LEFT JOIN embedding_metadata m ON m.id=e.id
            GROUP BY e.id ORDER BY e.embedding_id""").fetchall()
    finally:
        conn.close()
    selected = [r for r in rows if str(r[2]).startswith("zhfull:locomo:")]
    digest = hashlib.sha256(json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return selected, digest


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    rows, before = source_rows()
    ids = {r[0] for r in rows}
    cache = json.loads(CACHE.read_text())
    hits = ids & set(cache)
    by_kind = Counter()
    keys_by_kind = {kind: Counter() for kind in KINDS}
    with_source_by_kind = Counter()
    with_source_memory = set()
    top_keys = Counter()
    malformed = []
    for mid in sorted(hits):
        extraction = cache[mid]
        if not isinstance(extraction, dict):
            malformed.append(mid)
            continue
        top_keys.update(extraction.keys())
        if set(extraction) & SOURCE_FIELDS:
            with_source_memory.add(mid)
        for kind in KINDS:
            value = extraction.get(kind, [])
            if not isinstance(value, list):
                malformed.append(mid + ":" + kind)
                continue
            for item in value:
                if not isinstance(item, dict):
                    malformed.append(mid + ":" + kind + ":item")
                    continue
                by_kind[kind] += 1
                keys_by_kind[kind].update(item.keys())
                if set(item) & SOURCE_FIELDS:
                    with_source_by_kind[kind] += 1
                    with_source_memory.add(mid)
    _, after = source_rows()
    result = {
        "scope": "Current-ID cache keys in ten local zhfull:locomo users; raw cached S1 objects, before normalization or graph projection; read-only.",
        "cache_path": str(CACHE.relative_to(ROOT)),
        "cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        "current_source_logical_sha256_before": before,
        "current_source_logical_sha256_after": after,
        "current_source_count": len(ids),
        "current_cache_hits": len(hits),
        "current_cache_misses": len(ids - hits),
        "source_fields_checked": sorted(SOURCE_FIELDS),
        "raw_top_level_key_counts": dict(sorted(top_keys.items())),
        "raw_item_key_counts_by_kind": {kind: dict(sorted(keys_by_kind[kind].items())) for kind in KINDS},
        "raw_item_counts_by_kind": {kind: by_kind[kind] for kind in KINDS},
        "raw_item_total": sum(by_kind.values()),
        "items_with_checked_source_fields_by_kind": {kind: with_source_by_kind[kind] for kind in KINDS},
        "memories_with_checked_source_fields": len(with_source_memory),
        "malformed_entry_refs": malformed,
        "source_unchanged_during_audit": before == after,
        "limits": [
            "Only listed source-field keys are checked; a semantically encoded source hidden in free text is not assessed.",
            "Field absence does not imply every structured item is factually wrong.",
            "Cache hit by ID does not prove extraction input, model, or prompt version matches current memory text.",
            "A human sidecar may contain source evidence, but it is not part of this raw cache or current S1 graph path.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("current_source_count", "current_cache_hits",
                                           "raw_item_total", "memories_with_checked_source_fields",
                                           "source_unchanged_during_audit")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
