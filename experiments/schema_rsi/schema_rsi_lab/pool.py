"""Build a frozen sparse edge pool and opaque hubs from memory vectors.

Gold evidence IDs are copied into the output but never inspected by the edge
generator. Large corpora must supply ANN neighbors; the local kNN fallback has
an explicit per-user size guard and computes similarities in blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("candidate-pool preparation requires numpy") from exc
    return np


def build_pool(
    memories: list[dict], cases: list[dict], *, ann_edges: list[dict] | None = None,
    knn_k: int = 8, max_local_user_memories: int = 2000,
    hub_bits: int = 4, hub_tables: int = 2, hub_cap: int = 64, seed: int = 17,
) -> dict:
    """Produce Dataset.load-compatible nodes and opaque r0/r1 candidate edges.

    r0 is an ANN-supplied or locally computed vector-neighbor channel.
    r1 connects memories to vector-hash buckets (H nodes). The labels are
    intentionally operational IDs, not domain predicates.
    """
    if not 1 <= knn_k <= 128 or max_local_user_memories < 2:
        raise ValueError("invalid local neighbor limit")
    if not 1 <= hub_bits <= 16 or not 0 <= hub_tables <= 16 or hub_cap < 2:
        raise ValueError("invalid latent hub parameters")
    np = _numpy()
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    dimension: int | None = None
    for item in memories:
        node_id, user_id = str(item["id"]), str(item["user_id"])
        vector = item["embedding"]
        if not node_id or not user_id or node_id in seen:
            raise ValueError("memory IDs and users must be nonempty and IDs unique")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"memory {node_id} lacks an embedding vector")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("embedding dimensions differ")
        seen.add(node_id)
        groups[user_id].append(item)

    nodes = [{"id": str(item["id"]), "user_id": user, "kind": "M"}
             for user, items in sorted(groups.items()) for item in items]
    edges: dict[tuple[str, str, str], float] = {}

    def add_edge(left: str, right: str, channel: str, score: float) -> None:
        if left == right:
            return
        a, b = sorted((left, right))
        key = (a, b, channel)
        edges[key] = max(edges.get(key, 0.0), float(score))

    for user_id, items in sorted(groups.items()):
        user_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{user_id}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(user_seed)
        ids = [str(item["id"]) for item in items]
        matrix = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("nonfinite embedding")
        lengths = np.linalg.norm(matrix, axis=1)
        if (lengths == 0).any():
            raise ValueError("zero embedding")
        matrix = matrix / lengths[:, None]
        count = len(items)

        if ann_edges is None and count > max_local_user_memories:
            raise ValueError(
                f"user {user_id} has {count} memories; supply precomputed ANN edges "
                f"instead of local all-pairs kNN (limit {max_local_user_memories})"
            )
        if ann_edges is None and count > 1:
            k = min(knn_k, count - 1)
            # Blocked matrix product bounds peak similarity memory to O(block * N).
            for start in range(0, count, 256):
                stop = min(start + 256, count)
                similarities = matrix[start:stop] @ matrix.T
                for row in range(stop - start):
                    source_index = start + row
                    similarities[row, source_index] = -math.inf
                    top = np.argpartition(-similarities[row], k - 1)[:k]
                    ranked = sorted(top.tolist(), key=lambda j: (-float(similarities[row, j]), ids[j]))
                    for target_index in ranked:
                        score = (float(similarities[row, target_index]) + 1.0) / 2.0
                        add_edge(ids[source_index], ids[target_index], "r0", max(0.0, min(1.0, score)))

        for table in range(hub_tables):
            planes = rng.standard_normal((dimension, hub_bits), dtype=np.float32)
            signs = matrix @ planes >= 0
            buckets: dict[str, list[int]] = defaultdict(list)
            for index, bits in enumerate(signs):
                signature = "".join("1" if bit else "0" for bit in bits)
                buckets[signature].append(index)
            for signature, indexes in sorted(buckets.items()):
                if len(indexes) < 2:
                    continue
                # Cap hub fanout by splitting a large bucket into stable chunks.
                ordered = sorted(indexes, key=lambda i: ids[i])
                for part, offset in enumerate(range(0, len(ordered), hub_cap)):
                    members = ordered[offset:offset + hub_cap]
                    if len(members) < 2:
                        continue
                    opaque = hashlib.sha256(f"{user_id}:{table}:{signature}:{part}".encode()).hexdigest()[:16]
                    hub_id = f"H:{opaque}"
                    if hub_id in seen:
                        raise ValueError("latent hub ID collision")
                    seen.add(hub_id)
                    nodes.append({"id": hub_id, "user_id": user_id, "kind": "H"})
                    centroid = matrix[members].mean(axis=0)
                    norm = float(np.linalg.norm(centroid))
                    if norm == 0:
                        continue
                    centroid /= norm
                    for index in members:
                        cosine = float(matrix[index] @ centroid)
                        add_edge(ids[index], hub_id, "r1", max(0.0, min(1.0, (cosine + 1) / 2)))

    if ann_edges is not None:
        owners = {str(item["id"]): str(item["user_id"]) for item in memories}
        for item in ann_edges:
            left, right = str(item["source"]), str(item["target"])
            if left not in owners or right not in owners or owners[left] != owners[right]:
                raise ValueError(f"invalid ANN edge: {left} -> {right}")
            score = float(item["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("ANN edge score must be in [0,1]")
            add_edge(left, right, "r0", score)

    return {
        "nodes": nodes,
        "edges": [
            {"source": left, "target": right, "channel": channel, "score": score}
            for (left, right, channel), score in sorted(edges.items())
        ],
        "cases": cases,
        "pool_build": {
            "method": "blocked_cosine_knn+random_projection_hubs" if ann_edges is None else "external_ann+random_projection_hubs",
            "knn_k": knn_k, "hub_bits": hub_bits, "hub_tables": hub_tables,
            "hub_cap": hub_cap, "seed": seed,
            "gold_used_for_edges": False,
        },
    }
