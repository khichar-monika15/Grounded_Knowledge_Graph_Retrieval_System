"""Kùzu embedded graph store for multi-hop Cypher traversal.

Kùzu runs embedded (no server process), like SQLite.  The graph
mirrors the entity/claim data already in SQLite — it is NOT the source
of truth (SQLite is).  It is rebuilt at pipeline end and queried at
retrieval time for multi-hop expansion.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

NODE_DDL = """
CREATE NODE TABLE IF NOT EXISTS Entity(
    entity_id   STRING,
    name        STRING,
    entity_type STRING,
    PRIMARY KEY(entity_id)
)
"""

REL_DDL = """
CREATE REL TABLE IF NOT EXISTS Claim(
    FROM Entity TO Entity,
    claim_id     STRING,
    claim_type   STRING,
    confidence   DOUBLE,
    status       STRING,
    evidence_ids STRING
)
"""


class KuzuGraphStore:
    """Embedded Kùzu graph store wrapping entity/claim schema."""

    def __init__(self, db_path: str):
        import kuzu
        self.db_path = db_path
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(NODE_DDL)
        self._conn.execute(REL_DDL)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, entities, claims, clear_existing: bool = True) -> None:
        """Populate Kùzu from Pydantic Entity/Claim lists.

        clear_existing=True (default) drops and recreates tables — idempotent.
        Attribute claims (no object_entity_id) are silently skipped.
        """
        if clear_existing:
            try:
                self._conn.execute("DROP TABLE Claim")
                self._conn.execute("DROP TABLE Entity")
            except Exception:
                pass  # tables may not exist on first run
            self._create_schema()

        # Insert entities
        for e in entities:
            self._conn.execute(
                "CREATE (:Entity {entity_id: $id, name: $name, entity_type: $type})",
                parameters={
                    "id": e.entity_id,
                    "name": e.canonical_name,
                    "type": e.entity_type.value,
                }
            )

        # Insert relational claims only (skip attribute claims)
        rel_count = 0
        for c in claims:
            if not c.object_entity_id:
                continue
            try:
                self._conn.execute(
                    """MATCH (a:Entity {entity_id: $src}), (b:Entity {entity_id: $dst})
                       CREATE (a)-[:Claim {
                           claim_id: $cid, claim_type: $ct,
                           confidence: $conf, status: $st,
                           evidence_ids: $ev
                       }]->(b)""",
                    parameters={
                        "src": c.subject_entity_id,
                        "dst": c.object_entity_id,
                        "cid": c.claim_id,
                        "ct": c.claim_type.value,
                        "conf": float(c.confidence),
                        "st": c.status.value,
                        "ev": json.dumps(c.evidence_ids),
                    }
                )
                rel_count += 1
            except Exception as exc:
                logger.warning(f"Skipping claim {c.claim_id}: {exc}")

        logger.info(
            f"Kuzu loaded {len(entities)} entities, {rel_count} relational claims"
        )

    # ── Multi-hop ─────────────────────────────────────────────────────────────

    def neighborhood(self, entity_id: str, depth: int = 2) -> list[str]:
        """Return claim_ids reachable within `depth` hops from entity_id (undirected).

        Uses iterative 1-hop queries to avoid variable-length path API differences
        across kuzu versions.  Returns an empty list if entity is not found.
        """
        depth = max(1, min(depth, 5))

        visited_entities: set[str] = {entity_id}
        collected_claim_ids: list[str] = []

        frontier: set[str] = {entity_id}

        for _ in range(depth):
            if not frontier:
                break
            new_frontier: set[str] = set()
            for eid in frontier:
                # Outgoing edges
                try:
                    rows = self.execute_cypher(
                        "MATCH (src:Entity {entity_id: $id})-[c:Claim]->(dst:Entity) "
                        "RETURN c.claim_id AS cid, dst.entity_id AS dst_id",
                        params={"id": eid},
                    )
                    for row in rows:
                        cid = row.get("cid")
                        dst_id = row.get("dst_id")
                        if cid and cid not in collected_claim_ids:
                            collected_claim_ids.append(cid)
                        if dst_id and dst_id not in visited_entities:
                            new_frontier.add(dst_id)
                            visited_entities.add(dst_id)
                except Exception as exc:
                    logger.warning(f"neighborhood outgoing query failed for {eid}: {exc}")

                # Incoming edges
                try:
                    rows = self.execute_cypher(
                        "MATCH (src:Entity)-[c:Claim]->(dst:Entity {entity_id: $id}) "
                        "RETURN c.claim_id AS cid, src.entity_id AS src_id",
                        params={"id": eid},
                    )
                    for row in rows:
                        cid = row.get("cid")
                        src_id = row.get("src_id")
                        if cid and cid not in collected_claim_ids:
                            collected_claim_ids.append(cid)
                        if src_id and src_id not in visited_entities:
                            new_frontier.add(src_id)
                            visited_entities.add(src_id)
                except Exception as exc:
                    logger.warning(f"neighborhood incoming query failed for {eid}: {exc}")

            frontier = new_frontier

        return collected_claim_ids

    # ── Raw Cypher ────────────────────────────────────────────────────────────

    def execute_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return rows as a list of dicts.

        Does not use pandas — uses Kùzu's built-in column iterator.
        """
        result = self._conn.execute(query, parameters=params or {})
        cols = result.get_column_names()
        rows: list[dict] = []
        while result.has_next():
            rows.append(dict(zip(cols, result.get_next())))
        return rows

    def close(self) -> None:
        try:
            self._conn.destroy()
        except Exception:
            pass
        try:
            self._db.close()
        except Exception:
            pass
