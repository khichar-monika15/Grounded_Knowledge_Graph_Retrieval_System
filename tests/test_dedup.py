"""Tests for deduplication at all three levels."""
import pytest
import hashlib


class TestArtifactDedup:
    def test_identical_emails_detected(self):
        from dedup import hash_email_body
        h1 = hash_email_body("Hello world this is a test")
        h2 = hash_email_body("Hello world this is a test")
        assert h1 == h2

    def test_whitespace_normalized(self):
        from dedup import hash_email_body
        h1 = hash_email_body("Hello   world")
        h2 = hash_email_body("Hello world")
        assert h1 == h2

    def test_different_emails_not_equal(self):
        from dedup import hash_email_body
        h1 = hash_email_body("Email one")
        h2 = hash_email_body("Email two")
        assert h1 != h2

    def test_quoted_content_substring_detection(self):
        from dedup import is_quoted_duplicate
        original = "we need to discuss the California situation"
        quoted = "> we need to discuss the California situation"
        assert is_quoted_duplicate(quoted, [original]) is True


class TestEntityDedup:
    def test_same_person_different_names_merged(self, duplicate_entities):
        from dedup import deduplicate_entities
        merged = deduplicate_entities(duplicate_entities)
        person_names = [e.canonical_name for e in merged if e.entity_type.value == "person"]
        assert len([n for n in person_names if "skilling" in n.lower()]) == 1
        assert len([n for n in person_names if "lay" in n.lower()]) == 1

    def test_merge_preserves_all_aliases(self, duplicate_entities):
        from dedup import deduplicate_entities
        merged = deduplicate_entities(duplicate_entities)
        skilling = [e for e in merged if "skilling" in e.canonical_name.lower()][0]
        assert len(skilling.aliases) >= 3

    def test_merge_history_recorded(self, duplicate_entities):
        from dedup import deduplicate_entities
        merged = deduplicate_entities(duplicate_entities)
        skilling = [e for e in merged if "skilling" in e.canonical_name.lower()][0]
        assert len(skilling.merge_history) > 0
        assert "merged_from" in skilling.merge_history[0]

    def test_different_types_not_merged(self):
        from schema import Entity, EntityType
        from dedup import deduplicate_entities
        entities = [
            Entity(entity_id="e1", canonical_name="Raptor", entity_type=EntityType.PROJECT),
            Entity(entity_id="e2", canonical_name="Raptor", entity_type=EntityType.ORGANIZATION),
        ]
        merged = deduplicate_entities(entities)
        assert len(merged) == 2


class TestClaimDedup:
    def test_identical_claims_merged(self, duplicate_claims):
        from dedup import deduplicate_claims
        merged = deduplicate_claims(duplicate_claims)
        assert len(merged) == 1
        assert "ev1" in merged[0].evidence_ids
        assert "ev2" in merged[0].evidence_ids

    def test_merged_claim_keeps_highest_confidence(self, duplicate_claims):
        from dedup import deduplicate_claims
        merged = deduplicate_claims(duplicate_claims)
        assert merged[0].confidence == 0.9

    def test_claim_merge_history_recorded(self, duplicate_claims):
        from dedup import deduplicate_claims
        merged = deduplicate_claims(duplicate_claims)
        assert len(merged[0].merge_history) > 0

    def test_conflicting_claims_not_merged(self, conflicting_claims):
        from dedup import deduplicate_claims
        merged = deduplicate_claims(conflicting_claims)
        assert len(merged) == 2

    def test_conflicting_claims_older_superseded(self, conflicting_claims):
        from dedup import deduplicate_claims
        from schema import ClaimStatus
        merged = deduplicate_claims(conflicting_claims)
        older = [c for c in merged if c.claim_id == "c1"][0]
        newer = [c for c in merged if c.claim_id == "c2"][0]
        assert older.status == ClaimStatus.SUPERSEDED
        assert older.superseded_by == "c2"
        assert newer.status == ClaimStatus.ACTIVE
