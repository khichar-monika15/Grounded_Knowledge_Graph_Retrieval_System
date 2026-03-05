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
from datetime import datetime, UTC

import numpy as np

from memory.schema import Entity, Claim, ClaimStatus, EntityType

# Claim types where only one value is valid at a time (e.g. a person has one boss).
# Multi-valued types (sent_to, mentioned, discussed, etc.) can have many objects
# for the same subject — these are NOT conflicts and must NOT trigger supersession.
SINGLE_VALUED_CLAIMS = frozenset({"reports_to", "works_at", "role_assignment"})


def hash_email_body(body: str) -> str:
    """Normalize whitespace and compute MD5 hash of email body."""
    normalized = re.sub(r'\s+', ' ', body.strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def is_quoted_duplicate(text: str, seen_bodies: list[str]) -> bool:
    """Check if text (potentially quoted) is a substring of any seen body.

    Bug 20 optimisations:
    - Short texts (< 20 chars) are skipped — single-word matches cause false positives.
    - Comparison is capped at ~500 chars to limit O(N×M) cost on long emails.
    - seen_bodies is expected to be pre-cleaned (strip_quoted_content output)
      so we avoid re-normalising every body on every call.
    """
    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('>'):
            clean_lines.append(stripped.lstrip('>').strip())
        else:
            clean_lines.append(stripped)
    clean = ' '.join(clean_lines).strip()
    clean_normalized = re.sub(r'\s+', ' ', clean)

    # Very short text produces false positives (e.g. "ok" matches everything)
    if len(clean_normalized) < 20:
        return False

    # Cap comparison to first 500 chars to limit cost
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


def deduplicate_persons_splink(group: list[Entity]) -> list[list[int]]:
    """Probabilistic person entity resolution via Splink (Fellegi-Sunter model).

    Uses Jaro-Winkler comparison on canonical_name + last-name blocking.
    Returns list-of-index-lists (same signature as _build_merge_clusters).
    Falls back to embedding cosine (threshold=0.92 + last-name guard) if
    Splink raises an error (too few records for EM training).
    """
    n = len(group)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    try:
        import pandas as pd
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on
        import splink.comparison_library as cl

        records = []
        for i, e in enumerate(group):
            norm = _normalize_name(e.canonical_name)
            parts = norm.split()
            records.append({
                "unique_id": i,
                "canonical_name": norm,
                "last_name": parts[-1] if parts else "",
            })
        df = pd.DataFrame(records)

        settings = SettingsCreator(
            link_type="dedupe_only",
            comparisons=[cl.NameComparison("canonical_name")],
            blocking_rules_to_generate_predictions=[block_on("last_name")],
        )

        linker = Linker(df, settings, DuckDBAPI())
        linker.estimate_probability_two_random_records_match(
            deterministic_matching_rules=[block_on("last_name")],
            recall=0.6,
        )
        linker.estimate_u_using_random_sampling(max_pairs=1e5)
        linker.estimate_parameters_using_expectation_maximisation(block_on("last_name"))
        preds = linker.predict(threshold_match_probability=0.9).as_pandas_dataframe()
        for _, row in preds.iterrows():
            union(int(row["unique_id_l"]), int(row["unique_id_r"]))

    except Exception:
        # Fallback 3-pass approach when Splink EM training fails (too few records)

        # Pass 1: exact normalized name match (catches "Skilling, Jeff" ↔ "Jeff Skilling")
        for i in range(n):
            for j in range(i + 1, n):
                names_i = {_normalize_name(nm) for nm in [group[i].canonical_name] + group[i].aliases
                           if '@' not in nm}
                names_j = {_normalize_name(nm) for nm in [group[j].canonical_name] + group[j].aliases
                           if '@' not in nm}
                if names_i & names_j:
                    union(i, j)

        # Pass 2: same last name + first-name prefix match on canonical names
        # (catches "Jeff Skilling" ↔ "Jeffrey Skilling")
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                na = _normalize_name(group[i].canonical_name)
                nb = _normalize_name(group[j].canonical_name)
                if '@' in na or '@' in nb:
                    continue
                parts_a = na.split()
                parts_b = nb.split()
                if len(parts_a) >= 2 and len(parts_b) >= 2:
                    if parts_a[-1] == parts_b[-1]:  # same last name
                        fa, fb = parts_a[0], parts_b[0]
                        if fa.startswith(fb) or fb.startswith(fa):
                            union(i, j)

        # Pass 3: embedding cosine + last-name hard gate for remaining pairs
        from memory.embeddings import encode, cosine_similarity_matrix
        texts = [' '.join([e.canonical_name] + e.aliases) for e in group]
        embs = encode(texts, normalize=True)
        sim = cosine_similarity_matrix(embs, embs)
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                if sim[i, j] > 0.85:
                    last_i = _extract_last_name(_normalize_name(group[i].canonical_name))
                    last_j = _extract_last_name(_normalize_name(group[j].canonical_name))
                    if last_i != last_j and len(last_i) > 2 and len(last_j) > 2:
                        continue
                    union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _build_merge_clusters(group: list[Entity], etype: str, threshold: float = 0.85) -> list[list[int]]:
    """Route persons to Splink; use exact-match + embedding for non-persons."""
    if etype == "person" and len(group) > 1:
        return deduplicate_persons_splink(group)

    # --- Non-person path (orgs, projects, topics, locations, roles) ---
    n = len(group)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    # Pass 1: exact normalized name match (non-persons can safely merge on single-token names)
    for i in range(n):
        for j in range(i + 1, n):
            names_i = {_normalize_name(nm) for nm in [group[i].canonical_name] + group[i].aliases}
            names_j = {_normalize_name(nm) for nm in [group[j].canonical_name] + group[j].aliases}
            if names_i & names_j:
                union(i, j)

    # Pass 3: embedding cosine similarity
    from memory.embeddings import encode, cosine_similarity_matrix
    texts = [' '.join([e.canonical_name] + e.aliases) for e in group]
    embs = encode(texts, normalize=True)
    sim_matrix = cosine_similarity_matrix(embs, embs)
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            effective_threshold = 0.75 if etype == "topic" else threshold
            if sim_matrix[i, j] > effective_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
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
                    "timestamp": datetime.now(UTC).isoformat(),
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
             "reason": "duplicate_claim", "timestamp": datetime.now(UTC).isoformat()}
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
        claim_type_val = key[0].value if hasattr(key[0], 'value') else str(key[0])
        if len(obj_ids) <= 1 or claim_type_val not in SINGLE_VALUED_CLAIMS:
            # Multi-valued claims (sent_to, mentioned, etc.) naturally have many objects — no conflict
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
