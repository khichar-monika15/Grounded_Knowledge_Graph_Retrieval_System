"""Streamlit visualization app for the Enron memory graph."""
import json
import os
import sqlite3
import warnings

import streamlit as st
import networkx as nx

# Suppress CUDA / HuggingFace noise before any torch/transformers imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from config import DB_PATH, GRAPH_JSON_PATH

st.set_page_config(page_title="Enron Memory Graph", layout="wide")


@st.cache_resource
def get_embedding_model():
    """Load sentence-transformers model once for the lifetime of the app."""
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_data():
    """Load entities, claims, evidence from SQLite."""
    if not os.path.exists(DB_PATH):
        return [], [], []
    from graph_builder import load_entities_from_sqlite, load_claims_from_sqlite, load_evidence_from_sqlite
    entities = load_entities_from_sqlite(DB_PATH)
    claims = load_claims_from_sqlite(DB_PATH)
    evidence = load_evidence_from_sqlite(DB_PATH)
    return entities, claims, evidence


@st.cache_resource
def load_graph(_entities, _claims):
    """Build NetworkX graph from entities and claims."""
    from graph_builder import build_graph
    return build_graph(list(_entities), list(_claims))


def entity_type_color(etype: str) -> str:
    colors = {
        "person": "#4e79a7",
        "organization": "#59a14f",
        "project": "#f28e2b",
        "topic": "#e15759",
        "location": "#76b7b2",
        "role": "#edc948",
    }
    return colors.get(etype, "#b07aa1")


def render_graph_pyvis(G: nx.MultiDiGraph, entity_type_filter: list, confidence_threshold: float,
                        status_filter: str):
    """Render graph using pyvis."""
    from pyvis.network import Network
    import tempfile

    net = Network(height="500px", width="100%", directed=True, notebook=False)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      },
      "edges": {"arrows": {"to": {"enabled": true}}}
    }
    """)

    # Add nodes
    for node_id, data in G.nodes(data=True):
        etype = data.get("entity_type", "topic")
        if entity_type_filter and etype not in entity_type_filter:
            continue
        color = entity_type_color(etype)
        label = data.get("canonical_name", node_id)
        net.add_node(node_id, label=label, color=color, title=f"{etype}: {label}")

    # Add edges
    for u, v, data in G.edges(data=True):
        conf = data.get("confidence", 0.5)
        status = data.get("status", "active")
        if conf < confidence_threshold:
            continue
        if status_filter != "all" and status != status_filter:
            continue
        if not (net.get_node(u) and net.get_node(v)):
            continue
        claim_type = data.get("claim_type", "")
        net.add_edge(u, v, title=f"{claim_type} (conf={conf:.2f})", width=conf * 3)

    # Save to temp HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return f.name


def render_graph_agraph(G: nx.MultiDiGraph, entity_type_filter: list, confidence_threshold: float,
                         status_filter: str):
    """Render graph using streamlit-agraph."""
    from streamlit_agraph import agraph, Node, Edge, Config

    nodes = []
    edges = []

    for node_id, data in G.nodes(data=True):
        etype = data.get("entity_type", "topic")
        if entity_type_filter and etype not in entity_type_filter:
            continue
        color = entity_type_color(etype)
        label = data.get("canonical_name", node_id)
        nodes.append(Node(id=node_id, label=label, color=color, size=20))

    node_ids = {n.id for n in nodes}
    for u, v, data in G.edges(data=True):
        if u not in node_ids or v not in node_ids:
            continue
        conf = data.get("confidence", 0.5)
        status = data.get("status", "active")
        if conf < confidence_threshold:
            continue
        if status_filter != "all" and status != status_filter:
            continue
        claim_type = data.get("claim_type", "")
        edges.append(Edge(source=u, target=v, label=claim_type))

    config = Config(width=700, height=500, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)


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

    # Pre-warm embedding model (cached, loads once)
    import retrieval
    retrieval._MODEL = get_embedding_model()

    # Sidebar filters
    st.sidebar.header("Filters")
    all_etypes = sorted({e.entity_type.value for e in entities})
    selected_etypes = st.sidebar.multiselect("Entity Types", all_etypes, default=all_etypes)
    confidence_threshold = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
    status_filter = st.sidebar.selectbox("Claim Status", ["active", "all", "superseded"])

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Entities:** {len(entities)}")
    st.sidebar.markdown(f"**Claims:** {len(claims)}")
    st.sidebar.markdown(f"**Evidence:** {len(evidence)}")

    G = load_graph(tuple(entities), tuple(claims))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Memory Graph")
        html_path = render_graph_pyvis(G, selected_etypes, confidence_threshold, status_filter)
        with open(html_path) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=520, scrolling=True)
        selected_node = None

    with col2:
        st.subheader("Entity Browser")
        entity_names = [f"{e.canonical_name} ({e.entity_type.value})" for e in entities]
        selected_idx = st.selectbox("Select Entity", range(len(entity_names)),
                                     format_func=lambda i: entity_names[i])

        if selected_idx is not None:
            entity = entities[selected_idx]
            st.markdown(f"**{entity.canonical_name}**")
            st.markdown(f"Type: `{entity.entity_type.value}`")
            if entity.aliases:
                st.markdown(f"Aliases: {', '.join(entity.aliases[:5])}")
            if entity.merge_history:
                st.markdown(f"Merges: {len(entity.merge_history)} entities merged")

            st.markdown("**Claims:**")
            entity_claims = [c for c in claims if c.subject_entity_id == entity.entity_id]
            for c in entity_claims[:10]:
                status_emoji = "✓" if c.status.value == "active" else "⚠"
                target = entity_by_id.get(c.object_entity_id, {})
                target_name = target.canonical_name if hasattr(target, 'canonical_name') else c.object_value or "?"
                st.markdown(f"{status_emoji} **{c.claim_type.value}** → {target_name} (conf={c.confidence:.2f})")

                # Show evidence
                for ev_id in c.evidence_ids[:1]:
                    ev = ev_by_id.get(ev_id)
                    if ev:
                        st.caption(f'"{ev.excerpt[:100]}..." — {ev.sender or "unknown"}, {ev.timestamp.date() if ev.timestamp else "?"}')

    # Retrieval panel
    st.markdown("---")
    st.subheader("Question Retrieval")
    col_q, col_btn = st.columns([4, 1])
    with col_q:
        question = st.text_input("Ask a question about Enron", placeholder="Who did Jeff Skilling report to?")
    with col_btn:
        search = st.button("Search", use_container_width=True)

    if search and question:
        import retrieval
        retrieval._MODEL = get_embedding_model()  # reuse cached model
        from retrieval import build_context_pack
        with st.spinner("Searching..."):
            pack = build_context_pack(question, entities, claims, evidence)

        st.markdown(f"**Matched entities:** {', '.join(e['canonical_name'] for e in pack['matched_entities'][:5])}")

        if pack["claims"]:
            for claim_entry in pack["claims"][:5]:
                with st.expander(f"{claim_entry['claim']} (conf={claim_entry['confidence']:.2f})"):
                    for ev in claim_entry["evidence"]:
                        st.markdown(f"> {ev['excerpt']}")
                        st.caption(f"Source: {ev['source_id']} | From: {ev['sender'] or 'unknown'} | Date: {ev['date'] or 'unknown'}")
        else:
            st.info("No matching claims found.")

        if pack["conflicts"]:
            st.markdown("**Conflicting/Superseded claims:**")
            for c in pack["conflicts"][:3]:
                st.warning(f"{c['claim']} (status: {c['status']})")


if __name__ == "__main__":
    main()
