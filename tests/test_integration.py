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
