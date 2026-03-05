"""Retrieval and context pack generation.

Improvements over v1:
- Uses centralised embeddings.py singleton instead of per-module model loading.
- Integrates FAISS vector index (VectorStore) for <10ms entity/claim search.
- Replaces weighted scoring formula with Reciprocal Rank Fusion (RRF):
    score(d) = 1/(k + rank_BM25(d)) + 1/(k + rank_FAISS(d))
  RRF is parameter-free (k=60) and robust to heterogeneous score distributions.
- BM25 provides keyword recall; FAISS provides semantic recall; RRF merges them.
- Date range cached as module-level variable to avoid recomputing on every call.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np

from memory.schema import Entity, Claim, Evidence, ClaimStatus

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Optional FAISS store — injected by run_pipeline / app.py when available
_VECTOR_STORE = None

# Optional Kùzu store — injected by run_pipeline / app.py when available
_KUZU_STORE = None

# Cached date range for recency normalisation
_EARLIEST: Optional[datetime] = None
_LATEST: Optional[datetime] = None

# RRF constant (standard value; works well across many domains)
_RRF_K = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model():
    from memory.embeddings import get_model
    return get_model()


def _encode(texts: list[str]) -> np.ndarray:
    from memory.embeddings import encode
    return encode(texts, normalize=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    from memory.embeddings import cosine_similarity
    return cosine_similarity(a, b)


def _rrf_score(ranks: list[int], k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score from a list of per-signal ranks (1-indexed)."""
    return sum(1.0 / (k + r) for r in ranks)


def _bm25_search(query: str, corpus: list[str], k: int = 20) -> list[tuple[int, float]]:
    """BM25 keyword ranking over a corpus. Returns (index, score) sorted desc."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_matching_entities(query: str, entities: list[Entity],
                            threshold: float = 0.3) -> list[Entity]:
    """Find entities matching the query.

    If a VectorStore is available, uses FAISS for ANN search.
    Falls back to brute-force cosine similarity otherwise.
    Returns list of Entity objects with similarity > threshold.
    """
    if not entities:
        return []

    query_emb = _encode([query])[0]

    if _VECTOR_STORE is not None:
        hits = _VECTOR_STORE.search_entities(query_emb, k=min(20, len(entities)))
        id_to_score = {eid: score for eid, score in hits}
        results = [
            (e, id_to_score[e.entity_id])
            for e in entities
            if e.entity_id in id_to_score and id_to_score[e.entity_id] >= threshold
        ]
    else:
        # Brute-force fallback
        results = []
        for entity in entities:
            all_names = [entity.canonical_name] + entity.aliases
            for name in all_names:
                if query.lower() in name.lower() or name.lower() in query.lower():
                    results.append((entity, 1.0))
                    break
            else:
                text_emb = _encode([' '.join(all_names)])[0]
                sim = _cosine_sim(query_emb, text_emb)
                if sim >= threshold:
                    results.append((entity, sim))

    # Also add exact/substring matches not caught by FAISS
    faiss_ids = {e.entity_id for e, _ in results}
    for entity in entities:
        if entity.entity_id in faiss_ids:
            continue
        for name in [entity.canonical_name] + entity.aliases:
            if query.lower() in name.lower() or name.lower() in query.lower():
                results.append((entity, 1.0))
                break

    results.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in results]


def _find_matching_entities_with_scores(query: str, entities: list[Entity],
                                         threshold: float = 0.3) -> list[tuple[Entity, float]]:
    """Return (entity, score) tuples sorted by score descending."""
    if not entities:
        return []

    query_emb = _encode([query])[0]

    if _VECTOR_STORE is not None:
        hits = _VECTOR_STORE.search_entities(query_emb, k=min(20, len(entities)))
        id_to_score = {eid: score for eid, score in hits}
        results = [
            (e, id_to_score[e.entity_id])
            for e in entities
            if e.entity_id in id_to_score
        ]
    else:
        results = []
        for entity in entities:
            all_names = [entity.canonical_name] + entity.aliases
            matched_exact = False
            for name in all_names:
                if query.lower() in name.lower() or name.lower() in query.lower():
                    results.append((entity, 1.0))
                    matched_exact = True
                    break
            if not matched_exact:
                text_emb = _encode([' '.join(all_names)])[0]
                sim = _cosine_sim(query_emb, text_emb)
                results.append((entity, sim))

    # Add exact matches not in FAISS results
    found_ids = {e.entity_id for e, _ in results}
    for entity in entities:
        if entity.entity_id in found_ids:
            continue
        for name in [entity.canonical_name] + entity.aliases:
            if query.lower() in name.lower() or name.lower() in query.lower():
                results.append((entity, 1.0))
                break

    return sorted(results, key=lambda x: x[1], reverse=True)


def get_claims_for_entity(entity_id: str, claims: list[Claim]) -> list[Claim]:
    """Get all ACTIVE claims where entity is subject OR object."""
    return [c for c in claims
            if (c.subject_entity_id == entity_id or c.object_entity_id == entity_id)
            and c.status == ClaimStatus.ACTIVE]


def search_claims_by_text(query: str, claims: list[Claim],
                           threshold: float = 0.3,
                           entity_by_id: dict | None = None) -> list[Claim]:
    """Search claims by semantic similarity (FAISS if available, else brute-force).

    entity_by_id maps entity_id → canonical_name; when provided, claim texts use
    human-readable names instead of UUID strings for meaningful semantic matching.
    """
    if not claims:
        return []

    _eid = entity_by_id or {}
    claim_texts = []
    for c in claims:
        parts = [c.claim_type.value]
        parts.append(_eid.get(c.subject_entity_id, c.subject_entity_id))
        if c.object_entity_id:
            parts.append(_eid.get(c.object_entity_id, c.object_entity_id))
        if c.object_value:
            parts.append(c.object_value)
        claim_texts.append(' '.join(parts))

    query_emb = _encode([query])[0]

    if _VECTOR_STORE is not None:
        hits = _VECTOR_STORE.search_claims(query_emb, k=min(20, len(claims)))
        id_to_score = {cid: score for cid, score in hits}
        results = [
            (c, id_to_score[c.claim_id])
            for c in claims
            if c.claim_id in id_to_score and id_to_score[c.claim_id] >= threshold
        ]
        return [c for c, _ in sorted(results, key=lambda x: x[1], reverse=True)]
    else:
        embs = _encode(claim_texts)
        results = []
        for c, emb in zip(claims, embs):
            sim = _cosine_sim(query_emb, emb)  # computed once per claim (Bug 9)
            if sim >= threshold:
                results.append((c, sim))
        return [c for c, _ in sorted(results, key=lambda x: x[1], reverse=True)]


def compute_recency_score(timestamp: datetime, earliest: datetime,
                           latest: datetime) -> float:
    """Normalize timestamp to [0, 1] range."""
    total = (latest - earliest).total_seconds()
    if total == 0:
        return 1.0
    return max(0.0, min(1.0, (timestamp - earliest).total_seconds() / total))


def _cache_date_range(evidence: list[Evidence]) -> None:
    """Cache earliest/latest timestamps as module-level variables."""
    global _EARLIEST, _LATEST
    dates = [ev.timestamp for ev in evidence if ev.timestamp]
    if dates:
        _EARLIEST = min(dates)
        _LATEST = max(dates)
    else:
        _EARLIEST = datetime(2000, 1, 1)
        _LATEST = datetime(2002, 12, 31)


def build_context_pack(question: str, entities: list[Entity],
                        claims: list[Claim], evidence: list[Evidence]) -> dict:
    """Build a context pack answering a natural language question.

    Uses Reciprocal Rank Fusion (RRF) to merge:
      - FAISS semantic ranking (entity similarity + claim similarity)
      - BM25 keyword ranking over claim text
    Returns top-10 grounded claims with full evidence.
    """
    global _EARLIEST, _LATEST

    ev_by_id = {ev.evidence_id: ev for ev in evidence}
    entity_by_id = {e.entity_id: e for e in entities}

    # Cache date range once
    if _EARLIEST is None or _LATEST is None:
        _cache_date_range(evidence)

    # --- Signal 1: entity-based claim retrieval (FAISS/brute-force) ---
    entity_scores = _find_matching_entities_with_scores(question, entities, threshold=0.2)
    top_entities = entity_scores[:5]
    matched_entity_ids = {e.entity_id for e, _ in top_entities}

    entity_claims: list[Claim] = []
    for e, _ in top_entities:
        entity_claims.extend(get_claims_for_entity(e.entity_id, claims))

    # --- Signal 2: BM25 keyword ranking over claim text ---
    active_claims = [c for c in claims if c.status == ClaimStatus.ACTIVE]
    claim_texts_all = []
    for c in active_claims:
        parts = [c.claim_type.value]
        subj = entity_by_id.get(c.subject_entity_id)
        if subj:
            parts.append(subj.canonical_name)
        if c.object_entity_id and c.object_entity_id in entity_by_id:
            parts.append(entity_by_id[c.object_entity_id].canonical_name)
        if c.object_value:
            parts.append(c.object_value)
        claim_texts_all.append(' '.join(parts))

    bm25_ranked = _bm25_search(question, claim_texts_all, k=20)
    bm25_claim_ids = [active_claims[idx].claim_id for idx, _ in bm25_ranked]

    # --- Signal 3: FAISS/semantic claim search ---
    eid_to_name = {e.entity_id: e.canonical_name for e in entities}
    semantic_matches = search_claims_by_text(question, active_claims, threshold=0.2,
                                              entity_by_id=eid_to_name)
    semantic_claim_ids = [c.claim_id for c in semantic_matches[:20]]

    # --- Signal 4: Kùzu multi-hop expansion (2 hops) ---
    kuzu_claim_ids: list[str] = []
    if _KUZU_STORE is not None:
        for ent, _score in top_entities[:3]:
            hop_ids = _KUZU_STORE.neighborhood(ent.entity_id, depth=2)
            for cid in hop_ids:
                if cid not in kuzu_claim_ids:
                    kuzu_claim_ids.append(cid)

    # --- RRF fusion ---
    # Collect all candidate claim IDs
    all_candidate_ids: set[str] = set()
    all_candidate_ids.update(c.claim_id for c in entity_claims)
    all_candidate_ids.update(bm25_claim_ids)
    all_candidate_ids.update(semantic_claim_ids)
    all_candidate_ids.update(kuzu_claim_ids)

    claim_by_id = {c.claim_id: c for c in claims}

    def _rank_in(claim_id: str, ranked_list: list[str]) -> int:
        """1-indexed rank of claim_id in ranked_list, or len+1 if absent."""
        try:
            return ranked_list.index(claim_id) + 1
        except ValueError:
            return len(ranked_list) + 1

    entity_claim_ids = [c.claim_id for c in entity_claims]

    rrf_scores: dict[str, float] = {}
    for cid in all_candidate_ids:
        r_entity = _rank_in(cid, entity_claim_ids)
        r_bm25 = _rank_in(cid, bm25_claim_ids)
        r_semantic = _rank_in(cid, semantic_claim_ids)
        ranks = [r_entity, r_bm25, r_semantic]
        if kuzu_claim_ids:
            ranks.append(_rank_in(cid, kuzu_claim_ids))
        rrf_scores[cid] = _rrf_score(ranks)

    ranked_claim_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # --- Build output ---
    active_out: list[dict] = []
    conflict_out: list[dict] = []

    for cid in ranked_claim_ids[:15]:
        claim = claim_by_id.get(cid)
        if claim is None:
            continue

        claim_evs = [ev_by_id[eid] for eid in claim.evidence_ids if eid in ev_by_id]

        subj = entity_by_id.get(claim.subject_entity_id)
        subj_name = subj.canonical_name if subj else claim.subject_entity_id
        claim_str = f"{subj_name} {claim.claim_type.value}"
        if claim.object_entity_id and claim.object_entity_id in entity_by_id:
            claim_str += f" {entity_by_id[claim.object_entity_id].canonical_name}"
        elif claim.object_value:
            claim_str += f" {claim.object_value}"

        ev_list = [
            {
                "excerpt": ev.excerpt,
                "source_id": ev.source_id,
                "date": ev.timestamp.isoformat() if ev.timestamp else None,
                "sender": ev.sender,
            }
            for ev in claim_evs
        ]

        # Skip claims with no evidence (should not happen but guard anyway)
        if not ev_list:
            continue

        entry = {
            "claim": claim_str,
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
            "rrf_score": rrf_scores[cid],
            "evidence": ev_list,
        }

        if claim.status == ClaimStatus.ACTIVE:
            active_out.append(entry)
        else:
            conflict_out.append(entry)

        if len(active_out) >= 10:
            break

    return {
        "question": question,
        "matched_entities": [
            {
                "entity_id": e.entity_id,
                "canonical_name": e.canonical_name,
                "entity_type": e.entity_type.value,
                "score": s,
            }
            for e, s in top_entities
        ],
        "claims": active_out,
        "conflicts": conflict_out,
    }
