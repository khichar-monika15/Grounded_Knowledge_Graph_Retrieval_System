"""TDD tests for KuzuGraphStore — multi-hop graph reasoning.

9 tests covering: schema creation, idempotent load, 1-hop/2-hop/3-hop
neighborhood expansion, unknown entity, raw Cypher, and graceful degradation
when _KUZU_STORE is None.
"""
import pytest
import tempfile
import os


# ── Local fixture for multi-hop chain ────────────────────────────────────────

@pytest.fixture
def kuzu_chain_data():
    """e1→e3 (c1, works_at) and e3→e4 (c_bridge, mentioned) — 2-hop from e1 to e4."""
    from memory.schema import Entity, Claim, EntityType, ClaimType
    entities = [
        Entity(entity_id="e1", canonical_name="Jeff Skilling", entity_type=EntityType.PERSON),
        Entity(entity_id="e3", canonical_name="Enron", entity_type=EntityType.ORGANIZATION),
        Entity(entity_id="e4", canonical_name="Raptor", entity_type=EntityType.PROJECT),
    ]
    claims = [
        Claim(claim_id="c1", claim_type=ClaimType.WORKS_AT, subject_entity_id="e1",
              object_entity_id="e3", confidence=0.95, evidence_ids=["ev1"]),
        Claim(claim_id="c_bridge", claim_type=ClaimType.MENTIONED, subject_entity_id="e3",
              object_entity_id="e4", confidence=0.8, evidence_ids=["ev2"]),
    ]
    return {"entities": entities, "claims": claims}


# ── Schema ────────────────────────────────────────────────────────────────────

class TestKuzuStoreSchema:
    def test_schema_creates_tables(self, tmp_path):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        count = store.execute_cypher("MATCH (e:Entity) RETURN count(e) AS n")
        assert count[0]["n"] == 0  # empty but schema exists

    def test_schema_idempotent(self, tmp_path):
        """Calling KuzuGraphStore twice on the same path must not raise."""
        from memory.kuzu_store import KuzuGraphStore
        KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store2 = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        assert store2 is not None


# ── Load ──────────────────────────────────────────────────────────────────────

class TestKuzuStoreLoad:
    def test_load_entities(self, tmp_path, sample_graph_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        rows = store.execute_cypher("MATCH (e:Entity) RETURN count(e) AS n")
        assert rows[0]["n"] == 4  # e1, e2, e3, e4

    def test_load_claims_entity_to_entity_only(self, tmp_path, sample_graph_data):
        """Attribute claims (no object_entity_id) are skipped; c1 and c3 load."""
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        rows = store.execute_cypher("MATCH ()-[c:Claim]->() RETURN count(c) AS n")
        assert rows[0]["n"] == 2  # c1 (e1→e3) and c3 (e1→e4); c2 has no object_entity_id

    def test_idempotent_load(self, tmp_path, sample_graph_data):
        """Loading twice must not duplicate rows."""
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        rows = store.execute_cypher("MATCH (e:Entity) RETURN count(e) AS n")
        assert rows[0]["n"] == 4


# ── Multi-hop ─────────────────────────────────────────────────────────────────

class TestKuzuStoreMultiHop:
    def test_neighborhood_1hop_returns_direct_claim_ids(self, tmp_path, sample_graph_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        claim_ids = store.neighborhood("e1", depth=1)
        assert "c1" in claim_ids  # e1→e3 (works_at)
        assert "c3" in claim_ids  # e1→e4 (participated_in)

    def test_neighborhood_2hop_returns_indirect_claims(self, tmp_path, kuzu_chain_data):
        """e1→e3 (c1) then e3→e4 (c_bridge): 2-hop from e1 must include c_bridge."""
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(kuzu_chain_data["entities"], kuzu_chain_data["claims"])
        claim_ids = store.neighborhood("e1", depth=2)
        assert "c1" in claim_ids        # 1-hop
        assert "c_bridge" in claim_ids  # 2-hop

    def test_neighborhood_depth_3_valid(self, tmp_path, kuzu_chain_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(kuzu_chain_data["entities"], kuzu_chain_data["claims"])
        claim_ids = store.neighborhood("e1", depth=3)
        assert isinstance(claim_ids, list)

    def test_neighborhood_unknown_entity_returns_empty(self, tmp_path, sample_graph_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        claim_ids = store.neighborhood("nonexistent_id", depth=2)
        assert claim_ids == []


# ── Raw Cypher ────────────────────────────────────────────────────────────────

class TestKuzuRawCypher:
    def test_execute_returns_rows(self, tmp_path, sample_graph_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        rows = store.execute_cypher(
            "MATCH (e:Entity) WHERE e.entity_type = $t RETURN e.name AS name",
            params={"t": "person"}
        )
        names = {r["name"] for r in rows}
        assert "Jeff Skilling" in names
        assert "Ken Lay" in names

    def test_execute_empty_result_is_list(self, tmp_path, sample_graph_data):
        from memory.kuzu_store import KuzuGraphStore
        store = KuzuGraphStore(str(tmp_path / "test.kuzu"))
        store.load(sample_graph_data["entities"], sample_graph_data["claims"])
        rows = store.execute_cypher(
            "MATCH (e:Entity) WHERE e.name = 'Nobody' RETURN e.name"
        )
        assert rows == []


# ── Graceful degradation ──────────────────────────────────────────────────────

class TestKuzuResilience:
    def test_retrieval_works_without_kuzu_store(self, sample_graph_data):
        """build_context_pack must succeed when _KUZU_STORE is None."""
        import memory.retrieval as ret
        original = ret._KUZU_STORE
        ret._KUZU_STORE = None
        try:
            from memory.retrieval import build_context_pack
            pack = build_context_pack(
                "Jeff Skilling",
                sample_graph_data["entities"],
                sample_graph_data["claims"],
                sample_graph_data["evidence"]
            )
            assert "claims" in pack
        finally:
            ret._KUZU_STORE = original
