"""End-to-end integration tests. Run after all unit tests pass."""
import pytest
import os
import json


class TestFullPipeline:
    def test_extraction_to_graph_pipeline(self, sample_email_raw, sample_valid_llm_response, mocker, tmp_path):
        """Full flow: email → extraction → dedup → graph → SQLite."""
        from extraction import extract_email
        from dedup import deduplicate_entities, deduplicate_claims
        from graph_builder import build_graph, save_to_sqlite
        from schema import Entity, Claim, Evidence, EntityType, ClaimType
        import uuid

        mocker.patch("extraction.call_llm", return_value=sample_valid_llm_response)

        result = extract_email(sample_email_raw)
        assert result is not None

        entities = []
        for e in result["entities"]:
            entities.append(Entity(
                entity_id=str(uuid.uuid4()),
                canonical_name=e["name"],
                entity_type=EntityType(e["type"]),
                aliases=e.get("aliases", [])
            ))

        deduped_entities = deduplicate_entities(entities)
        assert len(deduped_entities) <= len(entities)

        G = build_graph(deduped_entities, [])
        assert G.number_of_nodes() > 0

        db_path = str(tmp_path / "integration_test.db")
        save_to_sqlite(deduped_entities, [], [], db_path)
        assert os.path.exists(db_path)

    def test_retrieval_returns_grounded_results(self, sample_graph_data):
        """Every returned claim must have evidence with non-empty excerpts."""
        from retrieval import build_context_pack
        pack = build_context_pack("Raptor project",
                                   sample_graph_data["entities"],
                                   sample_graph_data["claims"],
                                   sample_graph_data["evidence"])
        for claim in pack["claims"]:
            for ev in claim["evidence"]:
                assert ev["excerpt"].strip() != ""
                assert ev["source_id"] != ""

    def test_object_entity_auto_created(self):
        """Bug 10: object entities not listed under 'entities' must be auto-created
        so the claim becomes an entity-relation edge, not a degraded attribute claim."""
        from run_pipeline import emails_to_schema_objects
        import uuid

        extraction = {
            "_source_id": "test_bug10",
            "_email_meta": {"date": "2001-05-14", "sender": "a@enron.com", "recipient": "b@enron.com"},
            "entities": [
                {"name": "Jeff Skilling", "type": "person", "aliases": []},
                # "Enron" intentionally absent from entities list — LLM omission
            ],
            "claims": [{
                "claim_type": "works_at",
                "subject": "Jeff Skilling",
                "object": "Enron",
                "confidence": 0.9,
                "supporting_excerpt": "Jeff Skilling works at Enron",
            }],
        }
        email = {
            "source_id": "test_bug10",
            "date": "2001-05-14",
            "sender": "a@enron.com",
            "recipient": "b@enron.com",
            "body": "Jeff Skilling works at Enron",
        }

        entities, claims, _ = emails_to_schema_objects([extraction], [email])

        entity_names = {e.canonical_name for e in entities}
        assert "Enron" in entity_names, "Object entity 'Enron' must be auto-created"

        works_at = [c for c in claims if c.claim_type.value == "works_at"]
        assert len(works_at) == 1
        assert works_at[0].object_entity_id is not None, (
            "Claim should be an entity-relation (edge), not an attribute claim"
        )
        assert works_at[0].object_value is None, (
            "object_value should be None when object_entity_id is set"
        )

    def test_context_pack_serializable(self, sample_graph_data):
        """Context packs must be JSON-serializable for output files."""
        from retrieval import build_context_pack
        pack = build_context_pack("Jeff Skilling",
                                   sample_graph_data["entities"],
                                   sample_graph_data["claims"],
                                   sample_graph_data["evidence"])
        json_str = json.dumps(pack, default=str)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["question"] == "Jeff Skilling"
