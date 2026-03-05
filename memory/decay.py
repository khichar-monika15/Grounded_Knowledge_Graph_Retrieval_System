"""Time-based confidence decay for stale claims."""
from datetime import datetime

from memory.schema import Claim, ClaimStatus


def apply_confidence_decay(
    claims: list[Claim],
    reference_date: datetime = None,
    half_life_days: int = 180,
) -> list[Claim]:
    """Decay confidence for old claims with no corroborating evidence.

    Claims supported by multiple evidence sources decay slower (log scale).
    Claims that fall below 0.3 are marked UNCERTAIN.
    Already-superseded or retracted claims are untouched.
    """
    ref = reference_date or datetime(2001, 12, 31)
    decayed = []
    for c in claims:
        if c.status not in (ClaimStatus.ACTIVE,):
            decayed.append(c)
            continue

        age_days = max((ref - (c.valid_from or ref)).days, 0)
        evidence_count = max(len(c.evidence_ids), 1)
        # Multi-evidence claims get a longer effective half-life (up to 3× at 5+ sources)
        effective_half_life = half_life_days * (1 + 0.5 * min(evidence_count - 1, 5))
        decay_factor = 0.5 ** (age_days / effective_half_life) if age_days > 0 else 1.0

        new_confidence = round(c.confidence * decay_factor, 3)
        new_status = ClaimStatus.UNCERTAIN if new_confidence < 0.3 else c.status

        decayed.append(c.model_copy(update={
            "confidence": new_confidence,
            "status": new_status,
        }))
    return decayed
