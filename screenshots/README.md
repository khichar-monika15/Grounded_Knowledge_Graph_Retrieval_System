# Screenshots

To regenerate screenshots:
1. `uv run python main.py --sample-size 200` (or use existing `outputs/`)
2. `uv run streamlit run app/app.py`
3. Take 4 screenshots and save to this directory:
   - `graph_explorer.png`  — Tab 1, graph visible with colored nodes
   - `evidence_panel.png`  — Tab 1, click a node, evidence accordion open
   - `retrieval.png`       — Tab 3, question answered with claim cards
   - `merge_inspector.png` — Tab 2, merge history table visible
