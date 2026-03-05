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


@st.cache_resource
def get_embedding_model():
    """Return the shared embedding model singleton — delegates to memory/embeddings.py (Bug 8).
    Avoids loading a second ~90MB model copy into RAM."""
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
                        status_filter: str, focus_node_id: str = None):
    """Render graph using pyvis with zoom controls and entity focus."""
    from pyvis.network import Network
    import tempfile

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150},
        "barnesHut": {
          "gravitationalConstant": -3000,
          "springLength": 120,
          "springConstant": 0.04
        }
      },
      "edges": {"arrows": {"to": {"enabled": true}}, "smooth": {"type": "continuous"}},
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": false,
        "keyboard": {"enabled": true}
      }
    }
    """)

    # Add nodes — highlight focused node
    for node_id, data in G.nodes(data=True):
        etype = data.get("entity_type", "topic")
        if entity_type_filter and etype not in entity_type_filter:
            continue
        color = entity_type_color(etype)
        label = data.get("canonical_name", node_id)
        is_focused = (focus_node_id and node_id == focus_node_id)
        node_opts = {
            "label": label,
            "color": {
                "background": color,
                "border": "#FFD700" if is_focused else color,
                "highlight": {"background": color, "border": "#FFD700"},
            },
            "title": f"{etype}: {label}",
            "size": 25 if is_focused else 15,
            "borderWidth": 4 if is_focused else 1,
            "borderWidthSelected": 4,
        }
        net.add_node(node_id, **node_opts)

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
        is_connected = focus_node_id and (u == focus_node_id or v == focus_node_id)
        edge_color = "#FFD700" if is_connected else "#666666"
        net.add_edge(u, v, title=f"{claim_type} (conf={conf:.2f})",
                     width=(conf * 4) if is_connected else (conf * 2),
                     color=edge_color)

    # Save base HTML
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)

    # Inject zoom controls + focus JS into the generated HTML
    focus_js = ""
    if focus_node_id:
        safe_id = focus_node_id.replace("'", "\\'")
        focus_js = f"""
        // Auto-focus on selected entity after stabilization
        network.once('stabilized', function() {{
            try {{
                network.focus('{safe_id}', {{
                    scale: 1.8,
                    animation: {{duration: 800, easingFunction: 'easeInOutQuad'}}
                }});
                network.selectNodes(['{safe_id}']);
            }} catch(e) {{}}
        }});
        """

    controls_html = """
    <style>
      #graph-controls {
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      #graph-controls button {
        width: 36px;
        height: 36px;
        border: 1px solid #555;
        border-radius: 6px;
        background: rgba(30, 30, 30, 0.85);
        color: #fff;
        font-size: 18px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(4px);
        transition: background 0.15s, transform 0.1s;
      }
      #graph-controls button:hover {
        background: rgba(60, 60, 60, 0.95);
        transform: scale(1.08);
      }
      #graph-controls button:active {
        transform: scale(0.95);
      }
      #graph-controls button.active-btn {
        background: rgba(255, 215, 0, 0.3);
        border-color: #FFD700;
      }
      #graph-controls .separator {
        height: 1px;
        background: #555;
        margin: 2px 4px;
      }
    </style>
    <div id="graph-controls">
      <button onclick="zoomIn()" title="Zoom In">+</button>
      <button onclick="zoomOut()" title="Zoom Out">−</button>
      <button onclick="fitAll()" title="Fit All">⊡</button>
      <div class="separator"></div>
      <button id="physicsBtn" onclick="togglePhysics()" title="Toggle Physics">⚡</button>
    </div>
    <script>
      var physicsOn = true;

      function zoomIn() {
        var scale = network.getScale();
        network.moveTo({scale: scale * 1.4, animation: {duration: 300, easingFunction: 'easeInOutQuad'}});
      }
      function zoomOut() {
        var scale = network.getScale();
        network.moveTo({scale: scale / 1.4, animation: {duration: 300, easingFunction: 'easeInOutQuad'}});
      }
      function fitAll() {
        network.fit({animation: {duration: 500, easingFunction: 'easeInOutQuad'}});
      }
      function togglePhysics() {
        physicsOn = !physicsOn;
        network.setOptions({physics: {enabled: physicsOn}});
        var btn = document.getElementById('physicsBtn');
        btn.classList.toggle('active-btn', physicsOn);
        btn.title = physicsOn ? 'Physics ON (click to freeze)' : 'Physics OFF (click to unfreeze)';
      }

      // Mark physics button as active initially
      document.getElementById('physicsBtn').classList.add('active-btn');

      """ + focus_js + """
    </script>
    """

    # Inject controls before closing </body>
    with open(f.name, "r") as fh:
        html = fh.read()
    html = html.replace("</body>", controls_html + "</body>")
    with open(f.name, "w") as fh:
        fh.write(html)

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

    # --- Entity Browser (col2, rendered first so we know the focus target) ---
    with col2:
        st.subheader("Entity Browser 🔗")
        entity_names = [f"{e.canonical_name} ({e.entity_type.value})" for e in entities]
        selected_idx = st.selectbox("Select Entity", range(len(entity_names)),
                                     format_func=lambda i: entity_names[i])

    # Determine focus node
    focus_entity_id = None
    if selected_idx is not None:
        focus_entity_id = entities[selected_idx].entity_id

    # --- Graph View (col1, uses focus_entity_id) ---
    with col1:
        st.subheader("Memory Graph")
        st.caption("🔍 Use +/− buttons or scroll to zoom · Click an entity in the browser to focus")
        html_path = render_graph_pyvis(G, selected_etypes, confidence_threshold, status_filter,
                                       focus_node_id=focus_entity_id)
        with open(html_path) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=640, scrolling=False)

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
        import memory.embeddings as embeddings
        embeddings._MODEL = get_embedding_model()  # reuse cached model (Bug 4)
        from memory.retrieval import build_context_pack
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

    # Advanced Cypher panel — hidden by default
    st.markdown("---")
    import memory.retrieval as _ret_module
    with st.expander("Advanced: Graph Query (Cypher)", expanded=False):
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
