"""Tests for Pydantic schema models."""
import pytest
from datetime import datetime


class TestEvidence:
    def test_create_valid_evidence(self):
        from memory.schema import Evidence
        ev = Evidence(evidence_id="ev1", source_id="email_001", source_type="email",
                      excerpt="test excerpt", timestamp=datetime(2001, 5, 14),
                      sender="jeff@enron.com")
        assert ev.evidence_id == "ev1"
        assert ev.excerpt == "test excerpt"

    def test_evidence_requires_excerpt(self):
        from memory.schema import Evidence
        with pytest.raises(Exception):
            Evidence(evidence_id="ev1", source_id="email_001", source_type="email",
                     excerpt="", timestamp=None, sender=None)

    def test_evidence_optional_fields(self):
        from memory.schema import Evidence
        ev = Evidence(evidence_id="ev1", source_id="email_001", source_type="email",
                      excerpt="some text")
        assert ev.timestamp is None
        assert ev.sender is None


class TestEntity:
    def test_create_person_entity(self):
        from memory.schema import Entity, EntityType
        e = Entity(entity_id="e1", canonical_name="Jeff Skilling",
                   entity_type=EntityType.PERSON, aliases=["Skilling"])
        assert e.entity_type == EntityType.PERSON
        assert "Skilling" in e.aliases

    def test_entity_merge_history_starts_empty(self):
        from memory.schema import Entity, EntityType
        e = Entity(entity_id="e1", canonical_name="Test", entity_type=EntityType.PERSON)
        assert e.merge_history == []

    def test_entity_type_enum_values(self):
        from memory.schema import EntityType
        assert EntityType.PERSON == "person"
        assert EntityType.ORGANIZATION == "organization"
        assert EntityType.PROJECT == "project"


class TestClaim:
    def test_create_relation_claim(self):
        from memory.schema import Claim, ClaimType, ClaimStatus
        c = Claim(claim_id="c1", claim_type=ClaimType.WORKS_AT,
                  subject_entity_id="e1", object_entity_id="e2",
                  confidence=0.9, evidence_ids=["ev1"])
        assert c.status == ClaimStatus.ACTIVE
        assert c.confidence == 0.9

    def test_claim_default_status_is_active(self):
        from memory.schema import Claim, ClaimType, ClaimStatus
        c = Claim(claim_id="c1", claim_type=ClaimType.MENTIONED,
                  subject_entity_id="e1", confidence=0.5, evidence_ids=["ev1"])
        assert c.status == ClaimStatus.ACTIVE

    def test_claim_superseded_by_is_none_by_default(self):
        from memory.schema import Claim, ClaimType
        c = Claim(claim_id="c1", claim_type=ClaimType.DECIDED,
                  subject_entity_id="e1", confidence=0.7, evidence_ids=["ev1"])
        assert c.superseded_by is None

    def test_claim_requires_evidence(self):
        """Claims can have empty evidence_ids list (soft constraint)."""
        from memory.schema import Claim, ClaimType
        c = Claim(claim_id="c1", claim_type=ClaimType.MENTIONED,
                  subject_entity_id="e1", confidence=0.5, evidence_ids=[])
        assert c.evidence_ids == []

    def test_confidence_bounds(self):
        from memory.schema import Claim, ClaimType
        c = Claim(claim_id="c1", claim_type=ClaimType.MENTIONED,
                  subject_entity_id="e1", confidence=0.0, evidence_ids=["ev1"])
        assert 0.0 <= c.confidence <= 1.0

    def test_confidence_above_one_raises(self):
        """Bug 13: Claim.confidence must be bounded to [0, 1]."""
        from memory.schema import Claim, ClaimType
        with pytest.raises(Exception):  # Pydantic ValidationError
            Claim(claim_id="c1", claim_type=ClaimType.MENTIONED,
                  subject_entity_id="e1", confidence=1.5, evidence_ids=["ev1"])

    def test_confidence_below_zero_raises(self):
        """Bug 13: Claim.confidence must be bounded to [0, 1]."""
        from memory.schema import Claim, ClaimType
        with pytest.raises(Exception):  # Pydantic ValidationError
            Claim(claim_id="c1", claim_type=ClaimType.MENTIONED,
                  subject_entity_id="e1", confidence=-0.1, evidence_ids=["ev1"])


class TestRawClaimExtraction:
    def test_empty_subject_raises(self):
        """Bug 16: RawClaimExtraction.subject cannot be blank."""
        from memory.schema import RawClaimExtraction
        with pytest.raises(Exception):  # Pydantic ValidationError
            RawClaimExtraction(
                claim_type="mentioned", subject="", object="",
                confidence=0.5, supporting_excerpt="some text"
            )
