"""Tests for time-based confidence decay."""
import pytest
from datetime import datetime


def _make_claim(claim_id, confidence=0.8, valid_from=None, evidence_ids=None, status=None):
    from memory.schema import Claim, ClaimType, ClaimStatus
    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.MENTIONED,
        subject_entity_id="e1",
        confidence=confidence,
        evidence_ids=evidence_ids or ["ev1"],
        status=status or ClaimStatus.ACTIVE,
        valid_from=valid_from,
    )


class TestConfidenceDecay:
    def test_recent_claim_not_decayed(self):
        """A claim with valid_from == reference_date has age=0 → factor=1.0 → no change."""
        from memory.decay import apply_confidence_decay

        ref = datetime(2001, 12, 31)
        claim = _make_claim("c1", confidence=0.8, valid_from=ref)
        result = apply_confidence_decay([claim], reference_date=ref)
        assert result[0].confidence == 0.8

    def test_old_claim_decays(self):
        """A claim 365 days old should have lower confidence than the original."""
        from memory.decay import apply_confidence_decay

        ref = datetime(2001, 12, 31)
        old_date = datetime(2001, 1, 1)  # 364 days before ref
        claim = _make_claim("c1", confidence=0.8, valid_from=old_date)
        result = apply_confidence_decay([claim], reference_date=ref, half_life_days=180)
        assert result[0].confidence < 0.8

    def test_multi_evidence_decays_slower(self):
        """Same age, multi-evidence claim should decay less than single-evidence claim."""
        from memory.decay import apply_confidence_decay

        ref = datetime(2001, 12, 31)
        old_date = datetime(2001, 1, 1)

        single = _make_claim("c1", confidence=0.8, valid_from=old_date,
                              evidence_ids=["ev1"])
        multi = _make_claim("c2", confidence=0.8, valid_from=old_date,
                             evidence_ids=["ev1", "ev2", "ev3", "ev4", "ev5"])

        results_single = apply_confidence_decay([single], reference_date=ref)
        results_multi = apply_confidence_decay([multi], reference_date=ref)

        assert results_multi[0].confidence > results_single[0].confidence

    def test_below_threshold_marked_uncertain(self):
        """A very old claim should drop below 0.3 and be marked UNCERTAIN."""
        from memory.decay import apply_confidence_decay
        from memory.schema import ClaimStatus

        ref = datetime(2001, 12, 31)
        very_old = datetime(1999, 1, 1)  # ~3 years old
        claim = _make_claim("c1", confidence=0.5, valid_from=very_old)
        result = apply_confidence_decay([claim], reference_date=ref, half_life_days=180)
        assert result[0].status == ClaimStatus.UNCERTAIN
        assert result[0].confidence < 0.3

    def test_superseded_claim_untouched(self):
        """Superseded claims should be returned unchanged."""
        from memory.decay import apply_confidence_decay
        from memory.schema import ClaimStatus

        ref = datetime(2001, 12, 31)
        very_old = datetime(1999, 1, 1)
        claim = _make_claim("c1", confidence=0.9, valid_from=very_old,
                             status=ClaimStatus.SUPERSEDED)
        result = apply_confidence_decay([claim], reference_date=ref, half_life_days=180)
        # Status and confidence must be unchanged
        assert result[0].status == ClaimStatus.SUPERSEDED
        assert result[0].confidence == 0.9
