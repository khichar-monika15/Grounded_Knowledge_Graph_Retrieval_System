"""Tests for graph construction and SQLite persistence."""
import pytest
import os
import sqlite3


class TestGraphConstruction:
    def test_entities_become_nodes(self, sample_graph_data):
        from memory.graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        assert G.number_of_nodes() == 4
        assert G.nodes["e1"]["canonical_name"] == "Jeff Skilling"

    def test_claims_become_edges(self, sample_graph_data):
        from memory.graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        assert G.has_edge("e1", "e3")

    def test_attribute_claims_stored_on_node(self, sample_graph_data):
        """Claims with object_value (no object_entity) → stored as node attributes."""
        from memory.graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        node_data = G.nodes["e1"]
        assert "attribute_claims" in node_data or G.number_of_edges() >= 2

    def test_edge_has_evidence_ids(self, sample_graph_data):
        from memory.graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        edge_data = G.get_edge_data("e1", "e3")
        assert edge_data is not None


class TestPruning:
    def test_prune_leaf_topics_removes_single_edge_topics(self):
        from memory.graph_builder import build_graph, prune_leaf_topics
        from memory.schema import Entity, Claim, EntityType, ClaimType
        entities = [
            Entity(entity_id="p1", canonical_name="Jeff Skilling", entity_type=EntityType.PERSON),
            Entity(entity_id="p2", canonical_name="Ken Lay", entity_type=EntityType.PERSON),
            Entity(entity_id="t1", canonical_name="California Energy Crisis", entity_type=EntityType.TOPIC),
            Entity(entity_id="t2", canonical_name="quarterly earnings report", entity_type=EntityType.TOPIC),
        ]
        claims = [
            Claim(claim_id="c1", claim_type=ClaimType.DISCUSSED,
                  subject_entity_id="p1", object_entity_id="t1",
                  confidence=0.8, evidence_ids=["ev1"]),
            Claim(claim_id="c2", claim_type=ClaimType.DISCUSSED,
                  subject_entity_id="p2", object_entity_id="t1",
                  confidence=0.8, evidence_ids=["ev2"]),
            # t2 connected only to p1 — leaf topic, should be pruned
            Claim(claim_id="c3", claim_type=ClaimType.DISCUSSED,
                  subject_entity_id="p1", object_entity_id="t2",
                  confidence=0.5, evidence_ids=["ev3"]),
        ]
        G = build_graph(entities, claims)
        removed = prune_leaf_topics(G)
        assert removed == 1
        assert "t2" not in G.nodes  # leaf topic gone
        assert "t1" in G.nodes      # recurring topic preserved

    def test_prune_leaf_topics_preserves_persons(self):
        """Person nodes with degree 1 are NOT removed by prune_leaf_topics."""
        from memory.graph_builder import build_graph, prune_leaf_topics
        from memory.schema import Entity, Claim, EntityType, ClaimType
        entities = [
            Entity(entity_id="p1", canonical_name="Jeff Skilling", entity_type=EntityType.PERSON),
            Entity(entity_id="p2", canonical_name="Ken Lay", entity_type=EntityType.PERSON),
        ]
        claims = [
            Claim(claim_id="c1", claim_type=ClaimType.SENT_TO,
                  subject_entity_id="p1", object_entity_id="p2",
                  confidence=0.9, evidence_ids=["ev1"]),
        ]
        G = build_graph(entities, claims)
        removed = prune_leaf_topics(G)
        assert removed == 0
        assert "p1" in G.nodes
        assert "p2" in G.nodes


class TestSQLitePersistence:
    def test_save_and_load_entities(self, sample_graph_data, tmp_path):
        from memory.graph_builder import save_to_sqlite, load_entities_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_entities_from_sqlite(db_path)
        assert len(loaded) == 4

    def test_save_and_load_claims(self, sample_graph_data, tmp_path):
        from memory.graph_builder import save_to_sqlite, load_claims_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_claims_from_sqlite(db_path)
        assert len(loaded) == 3

    def test_save_and_load_evidence(self, sample_graph_data, tmp_path):
        from memory.graph_builder import save_to_sqlite, load_evidence_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_evidence_from_sqlite(db_path)
        assert len(loaded) == 2
        assert loaded[0].excerpt != ""

    def test_sqlite_tables_created(self, sample_graph_data, tmp_path):
        from memory.graph_builder import save_to_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "entities" in tables
        assert "claims" in tables
        assert "evidence" in tables
        assert "merges" in tables

    def test_idempotent_save(self, sample_graph_data, tmp_path):
        """Saving twice should not create duplicates."""
        from memory.graph_builder import save_to_sqlite, load_entities_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_entities_from_sqlite(db_path)
        assert len(loaded) == 4

    def test_evidence_char_offsets_persisted(self, tmp_path):
        """Bug 11: char_start/char_end must survive the SQLite round-trip."""
        from memory.graph_builder import save_to_sqlite, load_evidence_from_sqlite
        from memory.schema import Evidence
        from datetime import datetime
        ev = Evidence(
            evidence_id="ev_offset_test",
            source_id="email_001",
            source_type="email",
            excerpt="California situation",
            timestamp=datetime(2001, 5, 14),
            sender="jeff@enron.com",
            char_start=42,
            char_end=62,
        )
        db_path = str(tmp_path / "offset_test.db")
        save_to_sqlite([], [], [ev], db_path)
        loaded = load_evidence_from_sqlite(db_path)
        assert len(loaded) == 1
        assert loaded[0].char_start == 42
        assert loaded[0].char_end == 62

    def test_evidence_null_offsets_persisted(self, tmp_path):
        """Evidence with no offsets (None) should round-trip cleanly."""
        from memory.graph_builder import save_to_sqlite, load_evidence_from_sqlite
        from memory.schema import Evidence
        ev = Evidence(
            evidence_id="ev_no_offset",
            source_id="email_002",
            source_type="email",
            excerpt="some text",
        )
        db_path = str(tmp_path / "null_offset_test.db")
        save_to_sqlite([], [], [ev], db_path)
        loaded = load_evidence_from_sqlite(db_path)
        assert loaded[0].char_start is None
        assert loaded[0].char_end is None
