"""Streamlit visualization app for the Enron memory graph."""
import json
import os
import sys
import sqlite3
import warnings

# Ensure project root is on sys.path so `config` / `memory` are importable
# regardless of how Streamlit invokes this file.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import networkx as nx

# Suppress CUDA / HuggingFace noise before any torch/transformers imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from config import DB_PATH, GRAPH_JSON_PATH, KUZU_DB_PATH

FAISS_INDEX_PATH = "outputs/faiss_index.npz"

st.set_page_config(page_title="Enron Memory Graph", layout="wide")

ENTITY_COLORS = {
    "person": "#4e79a7",
    "organization": "#59a14f",
    "project": "#f28e2b",
    "topic": "#e15759",
    "location": "#76b7b2",
    "role": "#edc948",
}
ENTITY_ICONS = {
    "person": "person",
    "organization": "business",
    "project": "work",
    "topic": "label",
    "location": "place",
    "role": "badge",
}


@st.cache_resource
def get_embedding_model():
    """Return the shared embedding model singleton — delegates to memory/embeddings.py (Bug 8)."""
    from memory.embeddings import get_model
    return get_model()


@st.cache_resource
def load_data():
    """Load entities, claims, evidence from SQLite."""
    if not os.path.exists(DB_PATH):
        return [], [], []
    from memory.graph_builder import load_entities_from_sqlite, load_claims_from_sqlite, load_evidence_from_sqlite
    entities = load_entities_from_sqlite(DB_PATH)
    claims = load_claims_from_sqlite(DB_PATH)
    evidence = load_evidence_from_sqlite(DB_PATH)
    return entities, claims, evidence


@st.cache_resource
def load_graph(_entities, _claims):
    """Build NetworkX graph from entities and claims."""
    from memory.graph_builder import build_graph
    return build_graph(list(_entities), list(_claims))


def build_graph_elements(entities, claims, entity_type_filter, confidence_threshold, status_filter):
    """Build st-link-analysis elements (nodes + edges) with NodeStyle/EdgeStyle lists."""
    from st_link_analysis import NodeStyle, EdgeStyle
    from memory.schema import ClaimType

    visible_ids = {
        e.entity_id for e in entities
        if not entity_type_filter or e.entity_type.value in entity_type_filter
    }

    nodes = [
        {"data": {
            "id": e.entity_id,
            "label": e.entity_type.value.upper(),
            "name": e.canonical_name,
            "type": e.entity_type.value,
            "aliases": ", ".join(e.aliases[:3]),
        }}
        for e in entities if e.entity_id in visible_ids
    ]

    edges = []
    for c in claims:
        if not c.object_entity_id:
            continue
        if c.subject_entity_id not in visible_ids or c.object_entity_id not in visible_ids:
            continue
        if c.confidence < confidence_threshold:
            continue
        if status_filter != "all" and c.status.value != status_filter:
            continue
        edges.append({"data": {
            "id": c.claim_id,
            "label": c.claim_type.value.upper(),
            "source": c.subject_entity_id,
            "target": c.object_entity_id,
            "confidence": round(c.confidence, 2),
            "status": c.status.value,
        }})

    node_styles = [
        NodeStyle(etype.upper(), ENTITY_COLORS.get(etype, "#b07aa1"), "name", ENTITY_ICONS.get(etype, "circle"))
        for etype in ENTITY_COLORS
    ]
    edge_styles = [
        EdgeStyle(ct.value.upper(), caption="label", directed=True)
        for ct in ClaimType
    ]

    return {"nodes": nodes, "edges": edges}, node_styles, edge_styles


def main():
    st.title("Enron Memory Graph")
    st.markdown("Grounded long-term memory extracted from Enron emails.")

    entities, claims, evidence = load_data()

    if not entities:
        st.warning("No data found. Run `python run_pipeline.py` first to populate the database.")
        st.code("python run_pipeline.py --sample-size 200")
        return

    ev_by_id = {ev.evidence_id: ev for ev in evidence}
    entity_by_id = {e.entity_id: e for e in entities}

    # Pre-warm embedding model into the shared embeddings singleton (Bug 4)
    import memory.embeddings as embeddings
    embeddings._MODEL = get_embedding_model()

    # Load pre-built FAISS index into retrieval module (Bug 5)
    if os.path.exists(FAISS_INDEX_PATH):
        from memory.vector_store import VectorStore
        import memory.retrieval as retrieval
        retrieval._VECTOR_STORE = VectorStore.load(FAISS_INDEX_PATH)

    # Load Kùzu graph for multi-hop retrieval
    import memory.retrieval as _ret_module
    if os.path.exists(KUZU_DB_PATH):
        try:
            from memory.kuzu_store import KuzuGraphStore
            _ret_module._KUZU_STORE = KuzuGraphStore(KUZU_DB_PATH)
        except Exception:
            pass  # graceful degradation — multi-hop just disabled

    # Sidebar filters (global — affect Tab 1 graph)
    st.sidebar.header("Filters")
    all_etypes = sorted({e.entity_type.value for e in entities})
    selected_etypes = st.sidebar.multiselect("Entity Types", all_etypes, default=all_etypes)
    confidence_threshold = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
    status_filter = st.sidebar.selectbox("Claim Status", ["active", "all", "superseded"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Entities:** {len(entities)}")
    st.sidebar.markdown(f"**Claims:** {len(claims)}")
    st.sidebar.markdown(f"**Evidence:** {len(evidence)}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Graph Explorer", "Merge History", "Search", "Cypher Query"
    ])

    # ------------------------------------------------------------------
    # Tab 1: Graph Explorer
    # ------------------------------------------------------------------
    with tab1:
        from st_link_analysis import st_link_analysis

        elements, node_styles, edge_styles = build_graph_elements(
            entities, claims, selected_etypes, confidence_threshold, status_filter
        )

        st.caption(
            f"Showing {len(elements['nodes'])} nodes · {len(elements['edges'])} edges. "
            "Click a node or edge for details."
        )

        selected = st_link_analysis(
            elements, "cose", node_styles, edge_styles, height=580, key="main_graph"
        )

        # Detail panel driven by graph selection
        if selected and selected.get("data"):
            data = selected["data"]
            st.markdown("---")
            if data.get("type"):
                # Node selected — show entity detail
                entity = entity_by_id.get(data["id"])
                if entity:
                    st.markdown(f"### {entity.canonical_name}")
                    st.markdown(f"**Type:** `{entity.entity_type.value}`")
                    if entity.aliases:
                        st.markdown(f"**Aliases:** {', '.join(entity.aliases[:8])}")
                    if entity.merge_history:
                        st.caption(f"{len(entity.merge_history)} entities merged here")

                    st.markdown("**Claims:**")
                    entity_claims = [c for c in claims if c.subject_entity_id == entity.entity_id]
                    for c in entity_claims[:10]:
                        icon = "✓" if c.status.value == "active" else "⚠"
                        target = entity_by_id.get(c.object_entity_id)
                        tname = target.canonical_name if target else (c.object_value or "?")
                        st.markdown(f"{icon} **{c.claim_type.value}** → {tname} (conf={c.confidence:.2f})")
                        for ev_id in c.evidence_ids[:1]:
                            ev = ev_by_id.get(ev_id)
                            if ev:
                                st.caption(
                                    f'"{ev.excerpt[:100]}..." — {ev.sender or "?"}, '
                                    f'{ev.timestamp.date() if ev.timestamp else "?"}'
                                )
            else:
                # Edge selected — show claim detail
                st.markdown(f"### Claim: {data.get('label', '')}")
                st.markdown(
                    f"**Confidence:** {data.get('confidence', '?')} | "
                    f"**Status:** {data.get('status', '?')}"
                )
                src = entity_by_id.get(data.get("source", ""))
                tgt = entity_by_id.get(data.get("target", ""))
                if src and tgt:
                    st.markdown(f"**{src.canonical_name}** → **{tgt.canonical_name}**")

    # ------------------------------------------------------------------
    # Tab 2: Merge History
    # ------------------------------------------------------------------
    with tab2:
        import pandas as pd

        st.subheader("Entity Merge History")
        rows = []
        for e in entities:
            for mh in e.merge_history:
                rows.append({
                    "Canonical Name": e.canonical_name,
                    "Type": e.entity_type.value,
                    "Merged From ID": mh.get("merged_from", ""),
                    "Reason": mh.get("reason", ""),
                    "Timestamp": mh.get("timestamp", ""),
                    "Current Aliases": ", ".join(e.aliases[:5]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(f"{len(rows)} entity merge(s)")
        else:
            st.info("No entity merge history found. Run the pipeline first.")

        st.markdown("---")
        st.subheader("Full Merge Audit Log (SQLite)")
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            try:
                df = pd.read_sql_query(
                    "SELECT merge_type, merged_from, merged_into, reason, timestamp, reversible "
                    "FROM merges ORDER BY timestamp DESC",
                    conn,
                )
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} total merge record(s)")
            except Exception as exc:
                st.warning(f"Could not read merges table: {exc}")
            finally:
                conn.close()
        else:
            st.info("Database not found.")

    # ------------------------------------------------------------------
    # Tab 3: Search
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Question Retrieval")
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            question = st.text_input(
                "Ask a question about Enron",
                placeholder="Who did Jeff Skilling report to?"
            )
        with col_btn:
            search = st.button("Search", use_container_width=True)

        if search and question:
            import memory.embeddings as _emb
            _emb._MODEL = get_embedding_model()
            from memory.retrieval import build_context_pack
            with st.spinner("Searching..."):
                pack = build_context_pack(question, entities, claims, evidence)

            st.markdown(
                f"**Matched entities:** "
                f"{', '.join(e['canonical_name'] for e in pack['matched_entities'][:5])}"
            )

            if pack["claims"]:
                for claim_entry in pack["claims"][:5]:
                    with st.expander(f"{claim_entry['claim']} (conf={claim_entry['confidence']:.2f})"):
                        for ev in claim_entry["evidence"]:
                            st.markdown(f"> {ev['excerpt']}")
                            st.caption(
                                f"Source: {ev['source_id']} | "
                                f"From: {ev['sender'] or 'unknown'} | "
                                f"Date: {ev['date'] or 'unknown'}"
                            )
            else:
                st.info("No matching claims found.")

            if pack["conflicts"]:
                st.markdown("**Conflicting/Superseded claims:**")
                for c in pack["conflicts"][:3]:
                    st.warning(f"{c['claim']} (status: {c['status']})")

    # ------------------------------------------------------------------
    # Tab 4: Cypher Query
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Graph Query (Cypher)")
        if _ret_module._KUZU_STORE is None:
            st.info("Kùzu graph not loaded. Run the pipeline first.")
        else:
            TEMPLATES = {
                "Custom Cypher": "",
                "All claims of a type": (
                    "MATCH (a:Entity)-[c:Claim]->(b:Entity)\n"
                    "WHERE c.claim_type = 'works_at'\n"
                    "RETURN a.name, c.claim_type, b.name, c.confidence\n"
                    "ORDER BY c.confidence DESC LIMIT 20"
                ),
                "2-hop neighborhood": (
                    "MATCH (src:Entity {name: 'Jeff Skilling'})-[c:Claim*1..2]-(dst:Entity)\n"
                    "RETURN dst.name, dst.entity_type LIMIT 20"
                ),
            }
            template = st.selectbox("Template", list(TEMPLATES.keys()))
            cypher_query = st.text_area("Cypher query", value=TEMPLATES[template], height=120)
            if st.button("Run Query"):
                try:
                    rows = _ret_module._KUZU_STORE.execute_cypher(cypher_query)
                    if rows:
                        st.dataframe(rows)
                        st.caption(f"{len(rows)} row(s) returned")
                    else:
                        st.info("Query returned no results.")
                except Exception as exc:
                    st.error(f"Query error: {exc}")


if __name__ == "__main__":
    main()
