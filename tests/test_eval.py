"""Tests for gold-standard evaluation functions."""
import pytest


def _make_entity(entity_id, canonical_name, entity_type="person", aliases=None):
    from memory.schema import Entity, EntityType
    return Entity(
        entity_id=entity_id,
        canonical_name=canonical_name,
        entity_type=EntityType(entity_type),
        aliases=aliases or [],
    )


class TestGoldStandardEval:
    def test_perfect_merge_precision_recall(self):
        """A perfect entity set where all gold aliases are present → recall > 0."""
        from eval.gold_standard import evaluate_entity_dedup

        entities = [
            _make_entity("e1", "Jeffrey Skilling",
                         aliases=["Jeff Skilling", "Skilling", "jeff.skilling@enron.com"]),
            _make_entity("e2", "Kenneth Lay",
                         aliases=["Ken Lay", "ken.lay@enron.com", "Lay"]),
        ]
        result = evaluate_entity_dedup(entities)
        # No false merges → fp=0 → precision=1.0
        assert result["precision"] == 1.0
        # We covered multiple aliases → tp > 0
        assert result["tp"] > 0

    def test_false_merge_reduces_precision(self):
        """Merging Kenneth Lay with Andrew Fastow triggers fp+=1."""
        from eval.gold_standard import evaluate_entity_dedup

        entities = [
            # Wrongly merged: Kenneth Lay + Andrew Fastow in same entity
            _make_entity("e1", "Kenneth Lay",
                         aliases=["Ken Lay", "ken.lay@enron.com", "Lay",
                                  "Andrew Fastow", "Andy Fastow"]),
        ]
        result = evaluate_entity_dedup(entities)
        assert result["fp"] >= 1

    def test_missing_alias_reduces_recall(self):
        """Entity for Jeffrey Skilling missing 'Jeff Skilling' alias → fn increases."""
        from eval.gold_standard import evaluate_entity_dedup

        # Only canonical name, none of the expected aliases
        entities = [
            _make_entity("e1", "Jeffrey Skilling", aliases=[]),
        ]
        result_no_alias = evaluate_entity_dedup(entities)

        entities_with_alias = [
            _make_entity("e1", "Jeffrey Skilling",
                         aliases=["Jeff Skilling", "Skilling", "jeff.skilling@enron.com"]),
        ]
        result_with_alias = evaluate_entity_dedup(entities_with_alias)

        assert result_with_alias["recall"] > result_no_alias["recall"]

    def test_normalize_name_handles_formats(self):
        """Various name formats normalize to the same canonical form."""
        from eval.gold_standard import _normalize_name

        assert _normalize_name("Lay, Kenneth") == "kenneth lay"
        assert _normalize_name("Ken Lay") == "ken lay"
        assert _normalize_name("ken lay") == "ken lay"
        assert _normalize_name("Mr. Ken Lay") == "ken lay"
