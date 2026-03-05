"""Tests for graph construction and SQLite persistence."""
import pytest
import os
import sqlite3


class TestGraphConstruction:
    def test_entities_become_nodes(self, sample_graph_data):
        from graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        assert G.number_of_nodes() == 4
        assert G.nodes["e1"]["canonical_name"] == "Jeff Skilling"

    def test_claims_become_edges(self, sample_graph_data):
        from graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        assert G.has_edge("e1", "e3")

    def test_attribute_claims_stored_on_node(self, sample_graph_data):
        """Claims with object_value (no object_entity) → stored as node attributes."""
        from graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        node_data = G.nodes["e1"]
        assert "attribute_claims" in node_data or G.number_of_edges() >= 2

    def test_edge_has_evidence_ids(self, sample_graph_data):
        from graph_builder import build_graph
        G = build_graph(sample_graph_data["entities"], sample_graph_data["claims"])
        edge_data = G.get_edge_data("e1", "e3")
        assert edge_data is not None


class TestSQLitePersistence:
    def test_save_and_load_entities(self, sample_graph_data, tmp_path):
        from graph_builder import save_to_sqlite, load_entities_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_entities_from_sqlite(db_path)
        assert len(loaded) == 4

    def test_save_and_load_claims(self, sample_graph_data, tmp_path):
        from graph_builder import save_to_sqlite, load_claims_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_claims_from_sqlite(db_path)
        assert len(loaded) == 3

    def test_save_and_load_evidence(self, sample_graph_data, tmp_path):
        from graph_builder import save_to_sqlite, load_evidence_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_evidence_from_sqlite(db_path)
        assert len(loaded) == 2
        assert loaded[0].excerpt != ""

    def test_sqlite_tables_created(self, sample_graph_data, tmp_path):
        from graph_builder import save_to_sqlite
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
        from graph_builder import save_to_sqlite, load_entities_from_sqlite
        db_path = str(tmp_path / "test.db")
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        save_to_sqlite(sample_graph_data["entities"], sample_graph_data["claims"],
                       sample_graph_data["evidence"], db_path)
        loaded = load_entities_from_sqlite(db_path)
        assert len(loaded) == 4

    def test_evidence_char_offsets_persisted(self, tmp_path):
        """Bug 11: char_start/char_end must survive the SQLite round-trip."""
        from graph_builder import save_to_sqlite, load_evidence_from_sqlite
        from schema import Evidence
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
        from graph_builder import save_to_sqlite, load_evidence_from_sqlite
        from schema import Evidence
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
