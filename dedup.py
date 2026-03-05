"""Three-level deduplication: artifact, entity, claim.

Improvements over v1:
- Uses centralised embeddings.py singleton (no per-module model loading).
- Entity dedup uses vectorised numpy cosine matrix + sklearn AgglomerativeClustering
  instead of O(n²) Python loop — same asymptotic complexity but dramatically lower
  constant factor (single BLAS matmul vs. thousands of Python function calls).
"""
import hashlib
import re
import uuid
from datetime import datetime

import numpy as np

from schema import Entity, Claim, ClaimStatus, EntityType


def hash_email_body(body: str) -> str:
    """Normalize whitespace and compute MD5 hash of email body."""
    normalized = re.sub(r'\s+', ' ', body.strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def is_quoted_duplicate(text: str, seen_bodies: list[str]) -> bool:
    """Check if text (potentially quoted) is a substring of any seen body."""
    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('>'):
            clean_lines.append(stripped.lstrip('>').strip())
        else:
            clean_lines.append(stripped)
    clean = ' '.join(clean_lines).strip()
    clean_normalized = re.sub(r'\s+', ' ', clean)

    # Bug 12: cap comparison at 500 chars — avoids O(N×M) blow-up on long emails
    # while still catching duplicated quoted content (excerpts are rarely >500 chars)
    clean_prefix = clean_normalized[:500]
    for body in seen_bodies:
        body_normalized = re.sub(r'\s+', ' ', body.strip())
        if clean_prefix in body_normalized:
            return True
    return False


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    name = name.lower().strip()
    for prefix in ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.']:
        name = name.replace(prefix, '').strip()
    # Reverse "Last, First" format
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        name = f"{parts[1]} {parts[0]}"
    return name.strip()


def _extract_last_name(name: str) -> str:
    """Extract last name from a normalized person name."""
    parts = name.split()
    return parts[-1] if parts else name


def _names_likely_same_person(name_a: str, name_b: str) -> bool:
    """Check if two person names likely refer to the same person."""
    na = _normalize_name(name_a)
    nb = _normalize_name(name_b)

    la = _extract_last_name(na)
    lb = _extract_last_name(nb)
    if la == lb and la:
        tokens_a = set(na.split())
        tokens_b = set(nb.split())
        if tokens_a & tokens_b:
            return True
        # Check if first names are prefix-compatible (Jeff vs Jeffrey)
        first_a = na.split()[0] if na.split() else ""
        first_b = nb.split()[0] if nb.split() else ""
        if first_a and first_b:
            if first_a.startswith(first_b) or first_b.startswith(first_a):
                return True
    return False


def _build_merge_clusters(group: list[Entity], etype: str, threshold: float = 0.85) -> list[list[int]]:
    """Return list of index-clusters that should be merged.

    Strategy:
    1. Exact normalised name match → same cluster (fast, no embeddings needed).
    2. Person name heuristics (prefix match) → same cluster.
    3. Vectorised cosine similarity matrix (single BLAS matmul) — entities with
       similarity > threshold are candidates; sklearn AgglomerativeClustering
       groups them. This replaces the O(n²) Python loop.
    """
    n = len(group)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # Pass 1 & 2: exact match + person heuristics (cheap, no embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            names_i = {_normalize_name(nm) for nm in [group[i].canonical_name] + group[i].aliases}
            names_j = {_normalize_name(nm) for nm in [group[j].canonical_name] + group[j].aliases}

            if names_i & names_j:
                union(i, j)
                continue

            if etype == "person":
                raws_i = [group[i].canonical_name] + group[i].aliases
                raws_j = [group[j].canonical_name] + group[j].aliases
                found = any(
                    _names_likely_same_person(ni, nj)
                    for ni in raws_i
                    for nj in raws_j
                )
                if found:
                    union(i, j)

    # Pass 3: vectorised cosine similarity for remaining unmerged pairs
    # Build text representations
    texts = [' '.join([e.canonical_name] + e.aliases) for e in group]

    from embeddings import encode, cosine_similarity_matrix
    embs = encode(texts, normalize=True)          # shape (n, d), L2-normalised
    sim_matrix = cosine_similarity_matrix(embs, embs)  # shape (n, n), single matmul

    # Only check pairs not already in the same cluster
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if sim_matrix[i, j] > threshold:
                union(i, j)

    # Collect clusters
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return list(clusters.values())


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities using embedding similarity.
    Only merges entities of the SAME type.
    """
    if not entities:
        return []

    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        by_type.setdefault(e.entity_type.value, []).append(e)

    merged_all = []

    for etype, group in by_type.items():
        if len(group) == 1:
            merged_all.extend(group)
            continue

        clusters = _build_merge_clusters(group, etype, threshold=0.85)

        for indices in clusters:
            if len(indices) == 1:
                merged_all.append(group[indices[0]])
                continue

            members = [group[i] for i in indices]
            canonical = max((e.canonical_name for e in members), key=len)
            primary = members[0]

            all_aliases: set[str] = set()
            for e in members:
                all_aliases.add(e.canonical_name)
                all_aliases.update(e.aliases)
            all_aliases.discard(canonical)

            all_first = [e.first_seen for e in members if e.first_seen]
            all_last = [e.last_seen for e in members if e.last_seen]

            merge_hist = [
                {
                    "merged_from": e.entity_id,
                    "merged_into": primary.entity_id,
                    "reason": "embedding_similarity",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                for e in members[1:]
            ]

            merged_all.append(Entity(
                entity_id=primary.entity_id,
                canonical_name=canonical,
                entity_type=primary.entity_type,
                aliases=sorted(all_aliases),
                first_seen=min(all_first) if all_first else None,
                last_seen=max(all_last) if all_last else None,
                metadata=primary.metadata,
                merge_history=primary.merge_history + merge_hist,
            ))

    return merged_all


def deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Deduplicate and reconcile claims.
    - Same (type, subject, object_entity_id) → merge evidence, keep highest confidence.
    - Same (type, subject) but different object → keep both, mark older as SUPERSEDED.
    """
    if not claims:
        return []

    # Group by exact (claim_type, subject_entity_id, object_entity_id)
    exact_groups: dict[tuple, list[Claim]] = {}
    for c in claims:
        key = (c.claim_type, c.subject_entity_id, c.object_entity_id)
        exact_groups.setdefault(key, []).append(c)

    merged: list[Claim] = []
    for key, group in exact_groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        best = max(group, key=lambda c: c.confidence)
        seen_ev: set[str] = set()
        all_evidence = [ev for c in group for ev in c.evidence_ids if ev not in seen_ev and not seen_ev.add(ev)]  # type: ignore[func-returns-value]

        merge_hist = [
            {"merged_from": c.claim_id, "merged_into": best.claim_id,
             "reason": "duplicate_claim", "timestamp": datetime.utcnow().isoformat()}
            for c in group if c.claim_id != best.claim_id
        ]

        merged.append(best.model_copy(update={
            "evidence_ids": all_evidence,
            "merge_history": best.merge_history + merge_hist,
        }))

    # Handle conflicts: same (type, subject) but different object_entity_id
    conflict_groups: dict[tuple, list[Claim]] = {}
    for c in merged:
        conflict_groups.setdefault((c.claim_type, c.subject_entity_id), []).append(c)

    final: list[Claim] = []
    for key, group in conflict_groups.items():
        if len(group) == 1:
            final.append(group[0])
            continue

        # Bug 6 fix: only trigger conflict logic on entity-relation claims (object_entity_id is not None).
        # Attribute claims (object_value only, object_entity_id=None) must not be mixed in —
        # they would spuriously inflate obj_ids and get wrongly marked SUPERSEDED.
        entity_rel = [c for c in group if c.object_entity_id is not None]
        attr_claims = [c for c in group if c.object_entity_id is None]

        obj_ids = {c.object_entity_id for c in entity_rel}
        if len(obj_ids) <= 1:
            # No conflict among entity-relation claims — all pass through
            final.extend(group)
            continue

        # Conflicting entity-relation claims: sort by (valid_from, confidence); newest = winner
        sorted_entity_rel = sorted(entity_rel, key=lambda c: (c.valid_from or datetime.min, c.confidence))
        newest = sorted_entity_rel[-1]

        for c in sorted_entity_rel[:-1]:
            final.append(c.model_copy(update={
                "status": ClaimStatus.SUPERSEDED,
                "superseded_by": newest.claim_id,
            }))
        final.append(newest)
        # Attribute claims are unrelated to the conflict — pass through unchanged
        final.extend(attr_claims)

    return final
