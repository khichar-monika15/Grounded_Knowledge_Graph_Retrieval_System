"""Tests for retrieval and context pack generation."""
import pytest


class TestEntityMatching:
    def test_find_entity_by_exact_name(self, sample_graph_data):
        from retrieval import find_matching_entities
        matches = find_matching_entities("Jeff Skilling", sample_graph_data["entities"])
        assert any(e.entity_id == "e1" for e in matches)

    def test_find_entity_by_alias(self, sample_graph_data):
        from retrieval import find_matching_entities
        matches = find_matching_entities("Skilling", sample_graph_data["entities"])
        assert any(e.entity_id == "e1" for e in matches)

    def test_find_entity_by_semantic_similarity(self, sample_graph_data):
        from retrieval import find_matching_entities
        matches = find_matching_entities("Kenneth Lay chairman", sample_graph_data["entities"])
        assert any(e.entity_id == "e2" for e in matches)

    def test_no_match_returns_empty(self, sample_graph_data):
        from retrieval import find_matching_entities
        matches = find_matching_entities("completely unrelated xyz123", sample_graph_data["entities"])
        assert len(matches) == 0 or all(m[1] < 0.3 for m in matches)


class TestClaimRetrieval:
    def test_get_claims_for_entity(self, sample_graph_data):
        from retrieval import get_claims_for_entity
        claims = get_claims_for_entity("e1", sample_graph_data["claims"])
        assert len(claims) >= 2

    def test_only_active_claims_returned(self, sample_graph_data):
        from retrieval import get_claims_for_entity
        claims = get_claims_for_entity("e1", sample_graph_data["claims"])
        from schema import ClaimStatus
        assert all(c.status == ClaimStatus.ACTIVE for c in claims)

    def test_get_claims_for_entity_as_object(self, sample_graph_data):
        """Bug 18: claims where entity is object_entity_id must also be returned.
        e3 (Enron) is only an object in sample_graph_data (c1: e1 works_at e3)."""
        from retrieval import get_claims_for_entity
        claims = get_claims_for_entity("e3", sample_graph_data["claims"])
        assert len(claims) >= 1
        assert any(c.object_entity_id == "e3" for c in claims)


class TestClaimTextSearch:
    def test_fallback_search_finds_concept(self, sample_graph_data):
        """Searching 'California' should match claim about California situation."""
        from retrieval import search_claims_by_text
        matches = search_claims_by_text("California energy", sample_graph_data["claims"])
        assert len(matches) > 0

    def test_search_claims_uses_entity_names(self, sample_graph_data):
        """Bug 19: claim search must use entity canonical names not UUIDs.
        Searching 'Jeff Skilling Enron' should match the works_at claim."""
        from retrieval import search_claims_by_text
        entity_by_id = {e.entity_id: e.canonical_name
                        for e in sample_graph_data["entities"]}
        matches = search_claims_by_text(
            "Jeff Skilling Enron",
            sample_graph_data["claims"],
            entity_by_id=entity_by_id,
        )
        assert len(matches) > 0
        assert any(c.claim_type.value == "works_at" for c in matches)


class TestContextPack:
    def test_context_pack_structure(self, sample_graph_data):
        from retrieval import build_context_pack
        pack = build_context_pack("Who is Jeff Skilling?",
                                   sample_graph_data["entities"],
                                   sample_graph_data["claims"],
                                   sample_graph_data["evidence"])
        assert "question" in pack
        assert "matched_entities" in pack
        assert "claims" in pack
        assert "conflicts" in pack

    def test_context_pack_evidence_is_grounded(self, sample_graph_data):
        from retrieval import build_context_pack
        pack = build_context_pack("Jeff Skilling",
                                   sample_graph_data["entities"],
                                   sample_graph_data["claims"],
                                   sample_graph_data["evidence"])
        for claim in pack["claims"]:
            assert len(claim["evidence"]) > 0
            for ev in claim["evidence"]:
                assert "excerpt" in ev
                assert ev["excerpt"] != ""

    def test_recency_score_normalized(self):
        from retrieval import compute_recency_score
        from datetime import datetime
        score = compute_recency_score(
            datetime(2001, 6, 1),
            earliest=datetime(2001, 1, 1),
            latest=datetime(2001, 12, 31))
        assert 0.0 <= score <= 1.0
