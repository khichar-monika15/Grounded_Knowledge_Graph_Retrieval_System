"""Build NetworkX memory graph and persist to SQLite."""
import json
import sqlite3
import uuid
from datetime import datetime, UTC
from typing import Optional

import networkx as nx

from memory.schema import Entity, Claim, Evidence, EntityType, ClaimType, ClaimStatus


def build_graph(entities: list[Entity], claims: list[Claim]) -> nx.MultiDiGraph:
    """Build a directed multigraph from entities and claims."""
    G = nx.MultiDiGraph()

    # Add entity nodes
    for entity in entities:
        G.add_node(
            entity.entity_id,
            canonical_name=entity.canonical_name,
            entity_type=entity.entity_type.value,
            aliases=entity.aliases,
            first_seen=entity.first_seen.isoformat() if entity.first_seen else None,
            last_seen=entity.last_seen.isoformat() if entity.last_seen else None,
            attribute_claims=[],
        )

    # Add claim edges or node attributes
    for claim in claims:
        if claim.object_entity_id is not None:
            # Relation claim → edge
            if G.has_node(claim.subject_entity_id) and G.has_node(claim.object_entity_id):
                G.add_edge(
                    claim.subject_entity_id,
                    claim.object_entity_id,
                    claim_id=claim.claim_id,
                    claim_type=claim.claim_type.value,
                    confidence=claim.confidence,
                    status=claim.status.value,
                    valid_from=claim.valid_from.isoformat() if claim.valid_from else None,
                    valid_until=claim.valid_until.isoformat() if claim.valid_until else None,
                    evidence_ids=claim.evidence_ids,
                )
        elif claim.object_value is not None:
            # Attribute claim → node attribute
            if G.has_node(claim.subject_entity_id):
                node = G.nodes[claim.subject_entity_id]
                node["attribute_claims"].append({
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type.value,
                    "value": claim.object_value,
                    "confidence": claim.confidence,
                    "status": claim.status.value,
                    "evidence_ids": claim.evidence_ids,
                })

    return G


def prune_leaf_topics(G: nx.MultiDiGraph) -> int:
    """Remove topic nodes with degree <= 1 (appear in only one email, add noise).

    Topic nodes with a single edge are one-off sentence fragments that never
    recur. Keeping only topics with degree >= 2 preserves recurring themes
    (e.g. "California Energy Crisis" linked to many people) while eliminating
    the ~794 dead-end topic nodes that turn the graph into a hub-spoke mess.

    Returns the number of nodes removed.
    """
    to_remove = [
        n for n in G.nodes()
        if G.nodes[n].get("entity_type") == "topic" and G.degree(n) <= 1
    ]
    G.remove_nodes_from(to_remove)
    return len(to_remove)


def prune_orphan_nodes(G: nx.MultiDiGraph, keep_types: set[str] | None = None) -> int:
    """Remove nodes with degree=0 (no edges). Returns number removed.

    keep_types: entity types to preserve even if isolated (e.g. {"organization"}).
    """
    keep_types = keep_types or set()
    orphans = [
        n for n in G.nodes()
        if G.degree(n) == 0 and G.nodes[n].get("entity_type") not in keep_types
    ]
    G.remove_nodes_from(orphans)
    return len(orphans)


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            canonical_name TEXT,
            entity_type TEXT,
            aliases TEXT,
            first_seen TEXT,
            last_seen TEXT,
            metadata TEXT,
            merge_history TEXT
        );
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            claim_type TEXT,
            subject_entity_id TEXT REFERENCES entities(entity_id),
            object_entity_id TEXT REFERENCES entities(entity_id),
            object_value TEXT,
            confidence REAL,
            status TEXT DEFAULT 'active',
            valid_from TEXT,
            valid_until TEXT,
            evidence_ids TEXT,
            superseded_by TEXT,
            extraction_version TEXT,
            merge_history TEXT
        );
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            source_id TEXT,
            source_type TEXT,
            excerpt TEXT,
            timestamp TEXT,
            subject TEXT,
            sender TEXT,
            recipients TEXT,
            char_start INTEGER,
            char_end INTEGER
        );
        CREATE TABLE IF NOT EXISTS merges (
            merge_id TEXT PRIMARY KEY,
            merge_type TEXT,
            merged_from TEXT,
            merged_into TEXT,
            reason TEXT,
            timestamp TEXT,
            reversible INTEGER DEFAULT 1
        );
    """)
    # Bug 11 migration: add char_start/char_end if absent (for pre-existing databases)
    for col_def in ("char_start INTEGER", "char_end INTEGER"):
        try:
            conn.execute(f"ALTER TABLE evidence ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def save_to_sqlite(entities: list[Entity], claims: list[Claim],
                   evidence: list[Evidence], db_path: str):
    """Persist entities, claims, and evidence to SQLite.

    Uses executemany() for bulk inserts — 10-50× faster than per-row execute()
    for batches of 1000+ rows. INSERT OR REPLACE ensures idempotency.
    """
    conn = sqlite3.connect(db_path)
    _init_db(conn)

    # Batch insert entities
    entity_rows = [
        (
            e.entity_id,
            e.canonical_name,
            e.entity_type.value,
            json.dumps(e.aliases),
            e.first_seen.isoformat() if e.first_seen else None,
            e.last_seen.isoformat() if e.last_seen else None,
            json.dumps(e.metadata),
            json.dumps(e.merge_history),
        )
        for e in entities
    ]
    conn.executemany("INSERT OR REPLACE INTO entities VALUES (?,?,?,?,?,?,?,?)", entity_rows)

    # Batch insert claims
    claim_rows = [
        (
            c.claim_id,
            c.claim_type.value,
            c.subject_entity_id,
            c.object_entity_id,
            c.object_value,
            c.confidence,
            c.status.value,
            c.valid_from.isoformat() if c.valid_from else None,
            c.valid_until.isoformat() if c.valid_until else None,
            json.dumps(c.evidence_ids),
            c.superseded_by,
            c.extraction_version,
            json.dumps(c.merge_history),
        )
        for c in claims
    ]
    conn.executemany("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", claim_rows)

    # Batch insert evidence (Bug 11: include char_start/char_end)
    evidence_rows = [
        (
            ev.evidence_id,
            ev.source_id,
            ev.source_type,
            ev.excerpt,
            ev.timestamp.isoformat() if ev.timestamp else None,
            ev.subject,
            ev.sender,
            json.dumps(ev.recipients) if ev.recipients else None,
            ev.char_start,
            ev.char_end,
        )
        for ev in evidence
    ]
    conn.executemany("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?)", evidence_rows)

    # Batch insert merge audit records
    merge_rows = [
        (
            str(uuid.uuid4()),
            "entity",
            mh.get("merged_from", ""),
            mh.get("merged_into", ""),
            mh.get("reason", ""),
            mh.get("timestamp", datetime.now(UTC).isoformat()),
            1,
        )
        for e in entities
        for mh in e.merge_history
    ]
    if merge_rows:
        conn.executemany("INSERT OR IGNORE INTO merges VALUES (?,?,?,?,?,?,?)", merge_rows)

    conn.commit()
    conn.close()


def load_entities_from_sqlite(db_path: str) -> list[Entity]:
    """Load all entities from SQLite."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM entities").fetchall()
    conn.close()

    entities = []
    for row in rows:
        (entity_id, canonical_name, entity_type, aliases_json,
         first_seen, last_seen, metadata_json, merge_history_json) = row
        entities.append(Entity(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=EntityType(entity_type),
            aliases=json.loads(aliases_json) if aliases_json else [],
            first_seen=datetime.fromisoformat(first_seen) if first_seen else None,
            last_seen=datetime.fromisoformat(last_seen) if last_seen else None,
            metadata=json.loads(metadata_json) if metadata_json else {},
            merge_history=json.loads(merge_history_json) if merge_history_json else [],
        ))
    return entities


def load_claims_from_sqlite(db_path: str) -> list[Claim]:
    """Load all claims from SQLite."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM claims").fetchall()
    conn.close()

    claims = []
    for row in rows:
        (claim_id, claim_type, subject_entity_id, object_entity_id,
         object_value, confidence, status, valid_from, valid_until,
         evidence_ids_json, superseded_by, extraction_version, merge_history_json) = row
        claims.append(Claim(
            claim_id=claim_id,
            claim_type=ClaimType(claim_type),
            subject_entity_id=subject_entity_id,
            object_entity_id=object_entity_id,
            object_value=object_value,
            confidence=confidence,
            status=ClaimStatus(status),
            valid_from=datetime.fromisoformat(valid_from) if valid_from else None,
            valid_until=datetime.fromisoformat(valid_until) if valid_until else None,
            evidence_ids=json.loads(evidence_ids_json) if evidence_ids_json else [],
            superseded_by=superseded_by,
            extraction_version=extraction_version or "v1",
            merge_history=json.loads(merge_history_json) if merge_history_json else [],
        ))
    return claims


def load_evidence_from_sqlite(db_path: str) -> list[Evidence]:
    """Load all evidence from SQLite.

    Runs _init_db first so any older database (missing char_start/char_end)
    gets the ALTER TABLE migration applied before SELECT *, preventing the
    'not enough values to unpack' crash when the app loads a pre-Bug11 DB.
    """
    conn = sqlite3.connect(db_path)
    _init_db(conn)   # ensures char_start/char_end columns exist (idempotent)
    rows = conn.execute("SELECT * FROM evidence").fetchall()
    conn.close()

    evidence = []
    for row in rows:
        (evidence_id, source_id, source_type, excerpt, timestamp,
         subject, sender, recipients_json, char_start, char_end) = row
        evidence.append(Evidence(
            evidence_id=evidence_id,
            source_id=source_id,
            source_type=source_type or "email",
            excerpt=excerpt,
            timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
            subject=subject,
            sender=sender,
            recipients=json.loads(recipients_json) if recipients_json else None,
            char_start=char_start,
            char_end=char_end,
        ))
    return evidence


def save_graph_json(G: nx.MultiDiGraph, path: str):
    """Export graph to JSON format."""
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
