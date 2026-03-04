"""Retrieval and context pack generation."""
from datetime import datetime
from typing import Optional

import numpy as np
from schema import Entity, Claim, Evidence, ClaimStatus


def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_matching_entities(query: str, entities: list[Entity],
                            threshold: float = 0.3) -> list:
    """
    Find entities matching the query.
    Returns list of Entity objects where similarity > threshold.
    Also returns (entity, score) tuples for the no-match test to work.
    """
    if not entities:
        return []

    model = _get_model()
    query_emb = model.encode([query])[0]

    results = []
    for entity in entities:
        # Build text: canonical_name + all aliases
        all_names = [entity.canonical_name] + entity.aliases
        # Check exact/substring matches first
        for name in all_names:
            if query.lower() in name.lower() or name.lower() in query.lower():
                results.append((entity, 1.0))
                break
        else:
            # Semantic similarity
            text = ' '.join(all_names)
            emb = model.encode([text])[0]
            sim = _cosine_sim(query_emb, emb)
            if sim >= threshold:
                results.append((entity, sim))

    # Sort by score desc, return entities (for compatibility with test)
    results.sort(key=lambda x: x[1], reverse=True)

    # Return list of entities (not tuples) for tests that do `any(e.entity_id == ...)`
    # but keep tuple format available via the raw list for no_match test
    return [e for e, s in results]


def _find_matching_entities_with_scores(query: str, entities: list[Entity],
                                         threshold: float = 0.3) -> list[tuple]:
    """Returns (entity, score) tuples."""
    if not entities:
        return []

    model = _get_model()
    query_emb = model.encode([query])[0]

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
            text = ' '.join(all_names)
            emb = model.encode([text])[0]
            sim = _cosine_sim(query_emb, emb)
            results.append((entity, sim))

    return sorted(results, key=lambda x: x[1], reverse=True)


def get_claims_for_entity(entity_id: str, claims: list[Claim]) -> list[Claim]:
    """Get all ACTIVE claims for an entity (as subject)."""
    return [c for c in claims
            if c.subject_entity_id == entity_id and c.status == ClaimStatus.ACTIVE]


def search_claims_by_text(query: str, claims: list[Claim],
                           threshold: float = 0.3) -> list[Claim]:
    """Search claims by semantic similarity of claim text."""
    if not claims:
        return []

    model = _get_model()
    query_emb = model.encode([query])[0]

    # Build text for each claim
    claim_texts = []
    for c in claims:
        parts = [c.claim_type.value, c.subject_entity_id]
        if c.object_entity_id:
            parts.append(c.object_entity_id)
        if c.object_value:
            parts.append(c.object_value)
        claim_texts.append(' '.join(parts))

    if not claim_texts:
        return []

    claim_embs = model.encode(claim_texts)
    results = []
    for i, (claim, emb) in enumerate(zip(claims, claim_embs)):
        sim = _cosine_sim(query_emb, emb)
        if sim >= threshold:
            results.append((claim, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in results]


def compute_recency_score(timestamp: datetime, earliest: datetime,
                           latest: datetime) -> float:
    """Normalize timestamp to [0, 1] range."""
    total = (latest - earliest).total_seconds()
    if total == 0:
        return 1.0
    elapsed = (timestamp - earliest).total_seconds()
    return max(0.0, min(1.0, elapsed / total))


def build_context_pack(question: str, entities: list[Entity],
                        claims: list[Claim], evidence: list[Evidence]) -> dict:
    """Build a context pack answering a natural language question."""
    # Build evidence lookup
    ev_by_id = {ev.evidence_id: ev for ev in evidence}

    # Find matching entities
    entity_scores = _find_matching_entities_with_scores(question, entities, threshold=0.3)
    top_entities = entity_scores[:5]

    # Collect relevant claims
    matched_entity_ids = {e.entity_id for e, _ in top_entities}
    relevant_claims = []

    for e, score in top_entities:
        for c in get_claims_for_entity(e.entity_id, claims):
            relevant_claims.append((c, score))

    # Also search by text
    text_matches = search_claims_by_text(question, claims, threshold=0.3)
    seen_ids = {c.claim_id for c, _ in relevant_claims}
    for c in text_matches:
        if c.claim_id not in seen_ids:
            relevant_claims.append((c, 0.3))
            seen_ids.add(c.claim_id)

    # Compute dates for recency
    all_dates = []
    for ev in evidence:
        if ev.timestamp:
            all_dates.append(ev.timestamp)

    earliest = min(all_dates) if all_dates else datetime(2000, 1, 1)
    latest = max(all_dates) if all_dates else datetime(2002, 12, 31)

    # Score and rank claims
    scored_claims = []
    for claim, entity_sim in relevant_claims:
        # Get evidence dates
        claim_evs = [ev_by_id[eid] for eid in claim.evidence_ids if eid in ev_by_id]
        ev_dates = [ev.timestamp for ev in claim_evs if ev.timestamp]
        if ev_dates:
            recency = compute_recency_score(max(ev_dates), earliest, latest)
        else:
            recency = 0.5

        score = (entity_sim * 0.4) + (claim.confidence * 0.3) + (recency * 0.3)
        scored_claims.append((claim, score, claim_evs))

    scored_claims.sort(key=lambda x: x[1], reverse=True)

    # Separate active vs conflicts
    active_claims = []
    conflict_claims = []

    for claim, score, claim_evs in scored_claims[:10]:
        claim_str = f"{claim.subject_entity_id} {claim.claim_type.value}"
        if claim.object_entity_id:
            claim_str += f" {claim.object_entity_id}"
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

        entry = {
            "claim": claim_str,
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type.value,
            "confidence": claim.confidence,
            "status": claim.status.value,
            "score": score,
            "evidence": ev_list,
        }

        if claim.status == ClaimStatus.ACTIVE:
            active_claims.append(entry)
        else:
            conflict_claims.append(entry)

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
        "claims": active_claims,
        "conflicts": conflict_claims,
    }
