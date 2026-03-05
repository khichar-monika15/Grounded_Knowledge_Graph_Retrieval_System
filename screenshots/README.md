# Screenshots

To regenerate screenshots:
1. `uv run python main.py --sample-size 200` (or use existing `outputs/`)
2. `uv run streamlit run app/app.py`
3. Take 5 screenshots and save to this directory:
   - `1_graph_explorer.png`       — Tab 1, graph visible with colored nodes
   - `2_merge_history_top.png`    — Tab 2, entity merge history table
   - `3_merge_history_bottom.png` — Tab 2, full SQLite audit log scrolled down
   - `4_search_retrieval.png`     — Tab 3, question answered with claim cards
   - `5_advanced_query_cypher.png`— Tab 4, cypher query results
