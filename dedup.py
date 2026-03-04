"""Three-level deduplication: artifact, entity, claim."""
import hashlib
import re
import uuid
from datetime import datetime
from typing import Optional

from schema import Entity, Claim, ClaimStatus, EntityType


def hash_email_body(body: str) -> str:
    """Normalize whitespace and compute MD5 hash of email body."""
    normalized = re.sub(r'\s+', ' ', body.strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def is_quoted_duplicate(text: str, seen_bodies: list[str]) -> bool:
    """Check if text (potentially quoted) is a substring of any seen body."""
    # Strip leading '>' from each line
    clean_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('>'):
            clean_lines.append(stripped.lstrip('>').strip())
        else:
            clean_lines.append(stripped)
    clean = ' '.join(clean_lines).strip()
    clean_normalized = re.sub(r'\s+', ' ', clean)

    for body in seen_bodies:
        body_normalized = re.sub(r'\s+', ' ', body.strip())
        if clean_normalized in body_normalized:
            return True
    return False


_DEDUP_MODEL = None


def _get_embeddings(texts: list[str]):
    """Get embeddings for a list of texts (model loaded once)."""
    global _DEDUP_MODEL
    if _DEDUP_MODEL is None:
        import logging, warnings, os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        from sentence_transformers import SentenceTransformer
        _DEDUP_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _DEDUP_MODEL.encode(texts, convert_to_numpy=True)


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    name = name.lower().strip()
    # Remove common prefixes
    for prefix in ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.']:
        name = name.replace(prefix, '').strip()
    # Reverse "Last, First" format
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        name = f"{parts[1]} {parts[0]}"
    return name.strip()


def _extract_last_name(name: str) -> str:
    """Extract last name from a normalized person name."""
    # After normalization, assume last token is last name
    parts = name.split()
    return parts[-1] if parts else name


def _names_likely_same_person(name_a: str, name_b: str) -> bool:
    """Check if two person names likely refer to the same person."""
    na = _normalize_name(name_a)
    nb = _normalize_name(name_b)

    # Check if last names match and at least one token overlaps
    la = _extract_last_name(na)
    lb = _extract_last_name(nb)
    if la == lb and la:
        # Same last name — check if first names are compatible
        tokens_a = set(na.split())
        tokens_b = set(nb.split())
        # At least one shared token (last name already matches)
        if tokens_a & tokens_b:
            return True
        # Check if one first name is prefix of the other (Jeff vs Jeffrey)
        first_a = na.split()[0] if na.split() else ""
        first_b = nb.split()[0] if nb.split() else ""
        if first_a and first_b:
            if first_a.startswith(first_b) or first_b.startswith(first_a):
                return True

    return False


def deduplicate_entities(entities: list[Entity]) -> list[Entity]:
    """
    Deduplicate entities using embedding similarity.
    Only merges entities of the SAME type.
    """
    if not entities:
        return []

    # Group by entity_type
    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        key = e.entity_type.value
        by_type.setdefault(key, []).append(e)

    merged_all = []

    for etype, group in by_type.items():
        if len(group) == 1:
            merged_all.extend(group)
            continue

        # Build text representations for embedding
        texts = []
        for e in group:
            all_names = [e.canonical_name] + e.aliases
            texts.append(' '.join(all_names))

        embeddings = _get_embeddings(texts)

        # Union-Find for grouping
        parent = list(range(len(group)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        # Check pairwise similarity
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # Check all aliases too
                all_names_i = {_normalize_name(n) for n in [group[i].canonical_name] + group[i].aliases}
                all_names_j = {_normalize_name(n) for n in [group[j].canonical_name] + group[j].aliases}

                # Exact normalized name match
                if all_names_i & all_names_j:
                    union(i, j)
                    continue

                # For person entities: check if names likely refer to same person
                if etype == "person":
                    all_raw_i = [group[i].canonical_name] + group[i].aliases
                    all_raw_j = [group[j].canonical_name] + group[j].aliases
                    found_match = False
                    for ni in all_raw_i:
                        for nj in all_raw_j:
                            if _names_likely_same_person(ni, nj):
                                union(i, j)
                                found_match = True
                                break
                        if found_match:
                            break
                    if found_match:
                        continue

                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim > 0.85:
                    union(i, j)

        # Build merged entities
        groups: dict[int, list[int]] = {}
        for i in range(len(group)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        for root, indices in groups.items():
            if len(indices) == 1:
                merged_all.append(group[indices[0]])
                continue

            # Merge all into one
            members = [group[i] for i in indices]
            # Pick canonical_name = longest name
            canonical = max((e.canonical_name for e in members), key=len)
            primary = members[0]

            all_aliases = set()
            for e in members:
                all_aliases.add(e.canonical_name)
                all_aliases.update(e.aliases)
            all_aliases.discard(canonical)

            # Merge timestamps
            all_first = [e.first_seen for e in members if e.first_seen]
            all_last = [e.last_seen for e in members if e.last_seen]
            first_seen = min(all_first) if all_first else None
            last_seen = max(all_last) if all_last else None

            # Build merge history
            merge_hist = []
            for e in members[1:]:
                merge_hist.append({
                    "merged_from": e.entity_id,
                    "merged_into": primary.entity_id,
                    "reason": "embedding_similarity",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            merged_entity = Entity(
                entity_id=primary.entity_id,
                canonical_name=canonical,
                entity_type=primary.entity_type,
                aliases=sorted(all_aliases),
                first_seen=first_seen,
                last_seen=last_seen,
                metadata=primary.metadata,
                merge_history=primary.merge_history + merge_hist,
            )
            merged_all.append(merged_entity)

    return merged_all


def deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """
    Deduplicate and reconcile claims.
    - Same (type, subject, object_entity_id) → merge evidence, keep highest confidence
    - Same (type, subject) but different object → keep both, mark older as SUPERSEDED
    """
    if not claims:
        return []

    from difflib import SequenceMatcher

    # Group by (claim_type, subject_entity_id, object_entity_id)
    exact_groups: dict[tuple, list[Claim]] = {}
    for c in claims:
        key = (c.claim_type, c.subject_entity_id, c.object_entity_id)
        exact_groups.setdefault(key, []).append(c)

    merged: list[Claim] = []

    for key, group in exact_groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Merge: keep highest confidence, union evidence_ids
        best = max(group, key=lambda c: c.confidence)
        all_evidence = []
        seen_ev = set()
        for c in group:
            for ev in c.evidence_ids:
                if ev not in seen_ev:
                    all_evidence.append(ev)
                    seen_ev.add(ev)

        merge_hist = []
        for c in group:
            if c.claim_id != best.claim_id:
                merge_hist.append({
                    "merged_from": c.claim_id,
                    "merged_into": best.claim_id,
                    "reason": "duplicate_claim",
                    "timestamp": datetime.utcnow().isoformat(),
                })

        merged_claim = best.model_copy(update={
            "evidence_ids": all_evidence,
            "merge_history": best.merge_history + merge_hist,
        })
        merged.append(merged_claim)

    # Now handle conflicts: same (type, subject) but different object_entity_id
    # Group by (claim_type, subject_entity_id)
    conflict_groups: dict[tuple, list[Claim]] = {}
    for c in merged:
        key = (c.claim_type, c.subject_entity_id)
        conflict_groups.setdefault(key, []).append(c)

    final: list[Claim] = []
    for key, group in conflict_groups.items():
        if len(group) == 1:
            final.append(group[0])
            continue

        # Check if they have different object_entity_ids (conflicting)
        obj_ids = {c.object_entity_id for c in group}
        if len(obj_ids) <= 1:
            final.extend(group)
            continue

        # Conflicting claims: sort by (valid_from, confidence) descending
        def sort_key(c: Claim):
            ts = c.valid_from or datetime.min
            return (ts, c.confidence)

        sorted_group = sorted(group, key=sort_key)
        newest = sorted_group[-1]

        updated = []
        for c in sorted_group[:-1]:
            updated.append(c.model_copy(update={
                "status": ClaimStatus.SUPERSEDED,
                "superseded_by": newest.claim_id,
            }))
        updated.append(newest)
        final.extend(updated)

    return final
