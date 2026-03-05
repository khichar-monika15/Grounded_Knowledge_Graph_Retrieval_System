"""FAISS vector index for pre-computed entity and claim embeddings.

Builds and persists two FAISS IndexFlatIP indices (inner product on L2-normalised
vectors = cosine similarity) so retrieval queries hit a pre-built ANN index in
<10ms rather than encoding all entities/claims at search time.

Usage in pipeline:
    build_and_save_index(entities, claims, "outputs/faiss_index.npz")

Usage at query time:
    index = VectorStore.load("outputs/faiss_index.npz")
    entity_hits = index.search_entities(query_emb, k=5)
    claim_hits  = index.search_claims(query_emb, k=10)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from schema import Entity, Claim


class VectorStore:
    """In-memory FAISS index wrapping entity and claim embeddings."""

    def __init__(
        self,
        entity_ids: list[str],
        entity_embs: np.ndarray,
        claim_ids: list[str],
        claim_texts: list[str],
        claim_embs: np.ndarray,
    ):
        import faiss

        self.entity_ids = entity_ids
        self.claim_ids = claim_ids
        self.claim_texts = claim_texts

        dim = entity_embs.shape[1] if len(entity_embs) > 0 else claim_embs.shape[1]

        # Inner-product index on L2-normalised vectors == cosine similarity search
        self._entity_index = faiss.IndexFlatIP(dim)
        if len(entity_embs) > 0:
            self._entity_index.add(entity_embs.astype(np.float32))

        self._claim_index = faiss.IndexFlatIP(dim)
        if len(claim_embs) > 0:
            self._claim_index.add(claim_embs.astype(np.float32))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_entities(self, query_emb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return (entity_id, score) pairs for top-k nearest entities."""
        if self._entity_index.ntotal == 0:
            return []
        q = query_emb.reshape(1, -1).astype(np.float32)
        k = min(k, self._entity_index.ntotal)
        scores, indices = self._entity_index.search(q, k)
        return [
            (self.entity_ids[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

    def search_claims(self, query_emb: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Return (claim_id, score) pairs for top-k nearest claims."""
        if self._claim_index.ntotal == 0:
            return []
        q = query_emb.reshape(1, -1).astype(np.float32)
        k = min(k, self._claim_index.ntotal)
        scores, indices = self._claim_index.search(q, k)
        return [
            (self.claim_ids[idx], float(scores[0][rank]))
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save embeddings and metadata to a .npz archive."""
        import faiss

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Reconstruct raw float arrays from FAISS
        entity_embs = np.zeros((self._entity_index.ntotal, self._entity_index.d), dtype=np.float32)
        if self._entity_index.ntotal > 0:
            faiss.extract_index_ivf  # noqa: B018 (import check)
            entity_embs = self._entity_index.reconstruct_n(0, self._entity_index.ntotal)

        claim_embs = np.zeros((self._claim_index.ntotal, self._claim_index.d), dtype=np.float32)
        if self._claim_index.ntotal > 0:
            claim_embs = self._claim_index.reconstruct_n(0, self._claim_index.ntotal)

        np.savez_compressed(
            str(p),
            entity_embs=entity_embs,
            claim_embs=claim_embs,
            entity_ids=np.array(self.entity_ids, dtype=object),
            claim_ids=np.array(self.claim_ids, dtype=object),
            claim_texts=np.array(self.claim_texts, dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load from a .npz archive created by save()."""
        data = np.load(path, allow_pickle=True)
        return cls(
            entity_ids=data["entity_ids"].tolist(),
            entity_embs=data["entity_embs"],
            claim_ids=data["claim_ids"].tolist(),
            claim_texts=data["claim_texts"].tolist(),
            claim_embs=data["claim_embs"],
        )


# ------------------------------------------------------------------
# Build helper called from run_pipeline.py
# ------------------------------------------------------------------

def build_and_save_index(
    entities: list[Entity],
    claims: list[Claim],
    index_path: str,
) -> VectorStore:
    """Pre-compute embeddings for all entities and claims; build FAISS index; persist.

    Called once at the end of run_pipeline.py. The resulting .npz file is loaded
    by the Streamlit app at startup so retrieval never re-encodes the corpus.
    Claim texts use entity canonical names (not UUIDs) for meaningful semantic search.
    """
    from embeddings import encode

    print(f"Building FAISS index for {len(entities)} entities and {len(claims)} claims...")

    entity_by_id = {e.entity_id: e.canonical_name for e in entities}

    def _claim_text(claim: Claim) -> str:
        """Build a text representation using canonical entity names, not UUIDs."""
        parts = [claim.claim_type.value]
        parts.append(entity_by_id.get(claim.subject_entity_id, claim.subject_entity_id))
        if claim.object_entity_id:
            parts.append(entity_by_id.get(claim.object_entity_id, claim.object_entity_id))
        if claim.object_value:
            parts.append(claim.object_value)
        return ' '.join(parts)

    # Entity embeddings: canonical_name + aliases
    entity_ids = [e.entity_id for e in entities]
    entity_texts = [' '.join([e.canonical_name] + e.aliases) for e in entities]
    entity_embs = encode(entity_texts, normalize=True) if entity_texts else np.zeros((0, 384), dtype=np.float32)

    # Claim embeddings: claim_type + subject_name + object_name/value
    claim_ids = [c.claim_id for c in claims]
    claim_texts = [_claim_text(c) for c in claims]
    claim_embs = encode(claim_texts, normalize=True) if claim_texts else np.zeros((0, 384), dtype=np.float32)

    store = VectorStore(
        entity_ids=entity_ids,
        entity_embs=entity_embs.astype(np.float32),
        claim_ids=claim_ids,
        claim_texts=claim_texts,
        claim_embs=claim_embs.astype(np.float32),
    )
    store.save(index_path)
    print(f"FAISS index saved to {index_path}")
    return store
