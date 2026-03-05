# Project Timeline — Grounded Long-Term Memory System

A chronological account of decisions, errors, debugging, and improvements made while building the Layer10 take-home project.

---

## Day 1 — Project Setup & Schema

### Environment Bootstrap
**Decision:** Use the existing `uv` virtual environment rather than creating a new one. The user had already set up `.venv` and installed some packages.

**Action:** Added all required packages to `pyproject.toml` and ran `uv sync`. Discovered that `pydantic` was missing from dependencies despite being imported in schema code — added it explicitly.

**Directory structure created:**
```
tests/   data/   outputs/extractions/   outputs/context_packs/
```

**Files created:** `.gitignore`, `requirements.txt`, `config.py`

### Initial Config Mistake
**Error:** First `config.py` pointed `BASE_URL` to `https://llm-gateway.truefoundry.ai/api/inference/openai` — a non-existent subdomain.

**Discovery:** DNS resolution failure when running the first real extraction. The user provided the correct TrueFoundry API snippet directly:
```python
from openai import OpenAI
client = OpenAI(api_key='****', base_url='https://gateway.truefoundry.ai')
```

**Fix:** Updated `BASE_URL = "https://gateway.truefoundry.ai"`.

---

## Phase 1 — Schema (TDD)

### TDD Flow
Wrote `tests/test_schema.py` → ran tests (all RED) → implemented `schema.py` → tests GREEN.

**Key design decisions:**
- `Evidence.excerpt` gets a Pydantic `@field_validator` enforcing non-empty — grounding is a hard schema constraint, not just a convention.
- `ClaimStatus.SUPERSEDED` enables temporal claim evolution without deletion.
- `merge_history: list[dict]` on both `Entity` and `Claim` creates a reversible audit trail.
- `extraction_version: str = "v1"` on `Claim` for backfill queries when prompts change.

**Result:** 11/11 schema tests passing.

---

## Phase 2 — Extraction (TDD)

### LLM Provider Setup
**Initial approach:** Used `openai.OpenAI` client directly, pointed at TrueFoundry gateway.

**Model ID discovery:** The model ID `tfy-ai-bedrock/anthropic.claude-haiku-4-5-20251001-v1-0` returned HTTP 400 — "on-demand throughput not supported". Fixed by listing available models via `client.models.list()` and selecting `anthropic/claude-haiku-4-5-20251001`.

### Prompt Design
**First iteration:** Simple flat JSON prompt asking for entities + claims in one step.

**Problem:** LLM would sometimes invent entity names in claims that weren't in the entity list, causing orphaned references.

**Solution (later iteration):** Hybrid chain-of-thought prompt — STEP 1 forces the model to list all entities first, STEP 2 tells it to reference only those entities in claims. This reduces hallucination without a second API call.

### Response Parsing
**Problem:** LLM sometimes wraps JSON in `` ```json ``` `` markdown fences, which breaks `json.loads()`.

**Fix:** `parse_llm_response()` first strips markdown fences with regex, then extracts the first JSON object.

### Validation: Manual → Pydantic
**First implementation:** `validate_extraction()` manually checked dict keys.

**Problem:** This missed type errors (invalid `claim_type` strings, invalid `entity_type` values) that only surfaced later in `run_pipeline.py`.

**Improvement:** Replaced with `ExtractionResult.model_validate(data)` using new Pydantic models `RawEntityExtraction`, `RawClaimExtraction`. Now `claim_type` and `entity_type` are validated against enums; `supporting_excerpt` is enforced non-empty at the schema level.

**Result:** 14/14 extraction tests passing.

---

## Phase 3 — Deduplication (TDD)

### Three-Level Architecture
1. **Artifact dedup:** MD5 hash of normalized body → skip seen emails. Quoted-content substring detection catches forwarded email duplicates.
2. **Entity dedup:** Embedding cosine similarity > 0.85 within same entity type + exact normalized name match + person name heuristics.
3. **Claim dedup:** Group by (type, subject, object) → merge duplicates; conflicting (different object) → mark older as SUPERSEDED.

### Person Name Heuristic Bug
**Test failing:** `test_same_person_different_names_merged` — "Jeffrey Skilling" was not merging with "Jeff Skilling".

**Root cause:** Embedding cosine similarity for "Jeffrey Skilling" vs "Jeff Skilling" was ~0.83, just below the 0.85 threshold. Exact normalized match also failed.

**Fix:** Added `_names_likely_same_person(name_a, name_b)`:
- If last names match, check if first names are prefix-compatible (`Jeff`.startswith(`Jeff` of `Jeffrey`)? No, but `Jeffrey`.startswith(`Jeff`)? Yes)
- This catches Jeff/Jeffrey, Ken/Kenneth, Andy/Andrew, Bill/William without needing high embedding similarity.

### Conservative Threshold
**Decision:** Keep cosine threshold at 0.85 (strict). Better to under-merge (two entries for the same person) than over-merge (conflating "Ken Lay" and "Ken Rice"). The person name heuristics handle the common name variant cases more precisely.

**Result:** 13/13 dedup tests passing.

---

## Phase 4 — Graph Builder (TDD)

### Architecture Choice: NetworkX + SQLite
**Decision rationale:**
- NetworkX for in-memory graph traversal and visualization — no operational overhead, great Python API.
- SQLite as source of truth — ACID persistence, SQL queries for backfill, single-file portability.
- Graph is rebuilt from SQLite on each app load; SQLite is not rebuilt from the graph.

### Attribute Claims
**Design:** Claims with `object_entity_id` (entity→entity) become graph edges. Claims with only `object_value` (e.g., "discussed California situation") become node-level `attribute_claims` list — no dangling edge to a non-entity node.

### Idempotent Save
**Design:** All inserts use `INSERT OR REPLACE` — re-running the pipeline on the same data is a no-op. Artifact dedup by body hash ensures the same email is not re-extracted.

**Result:** 9/9 graph builder tests passing.

---

## Phase 5 — Retrieval (TDD)

### Retrieval Design
**Primary:** Exact/substring match on entity canonical name and aliases (handles "Jeff Skilling" → entity match).

**Fallback:** Cosine similarity on concatenated claim text (handles concept queries like "California energy" that don't map to a named entity).

**Scoring (initial):** `(entity_sim × 0.4) + (claim_confidence × 0.3) + (recency × 0.3)`

**Recency normalization:** Normalize to [0,1] over the corpus date range to prevent recency from dominating when all emails are from the same month.

### Context Pack Grounding
**Design:** Every claim in the context pack includes its `Evidence` objects with `excerpt`, `source_id`, `sender`, `date`. No claim enters the context pack without provenance.

### Entity Name Resolution Bug
**Problem:** Context pack was showing UUID strings (e.g., `"e1 works_at e3"`) instead of human-readable entity names.

**Fix:** Added `entity_by_id = {e.entity_id: e for e in entities}` lookup dict inside `build_context_pack()`. Claim strings are now built as `"{subject_name} {claim_type} {object_name}"`.

**Result:** 10/10 retrieval tests passing, 3/3 integration tests passing. **Total: 60/60 tests.**

---

## Phase 6 — Integration Tests & Full Pipeline Run

All 60 tests passing. Ran `run_pipeline.py` on 200 emails from 2001.

**Pipeline results:**
```
Entities:              1096
Claims:                1074
Evidence:              1174
Merges:                528
Avg confidence:        0.985
Conflicts (SUPERSEDED): 234
Validation errors:     0
Duplicate emails:      0
```

---

## Phase 7 — Streamlit App

### streamlit-agraph Failure
**Error:** `FileNotFoundError: .../streamlit_agraph/frontend/build/ae9229e7-....js`

**Root cause:** Broken frontend chunk in the installed package — the compiled JS bundle was incomplete. This is a package installation artifact that cannot be caught in Python.

**Fix:** Removed streamlit-agraph entirely. Switched to `pyvis` + `st.components.v1.html()` for graph rendering. pyvis generates a self-contained HTML file that Streamlit embeds in an iframe.

### Model Loading Multiple Times
**Problem:** Streamlit re-executes the entire script on every user interaction. The `SentenceTransformer` model was being loaded on every render, causing ~3-5s delay and flooding logs with CUDA/HuggingFace warnings.

**Fix (two layers):**
1. Module-level `_MODEL = None` singleton in `retrieval.py` and `dedup.py` — model loaded once per process.
2. `@st.cache_resource` on `get_embedding_model()` in `app.py` — Streamlit's cache persists across rerenders. The cached model is injected back: `retrieval._MODEL = get_embedding_model()`.

---

## Improvements Batch 1 (Performance)

### litellm Provider Abstraction
**Motivation:** Replace direct `openai.OpenAI` client with `litellm.completion` for provider-agnostic calls.

**Benefit:** Switching from TrueFoundry → Ollama (local, zero cost) requires only config changes, no code changes:
```python
MODEL = "ollama/llama3"
BASE_URL = "http://localhost:11434"
```

### Pydantic as Extraction Contract
**Motivation:** Manual `validate_extraction()` didn't catch enum mismatches or type errors.

**Implementation:** Added `RawEntityExtraction`, `RawClaimExtraction`, `ExtractionResult` to `schema.py`. `validate_extraction()` now wraps `ExtractionResult.model_validate()`.

**Key guarantee:** `RawClaimExtraction.supporting_excerpt` has a `@field_validator` — it is literally impossible for a claim to pass validation with an empty excerpt.

### Hybrid Chain-of-Thought Prompt
**Problem:** Single-step extraction sometimes produced claims referencing entity names that weren't in the entity list.

**Solution:** Two-step prompt in one response:
- STEP 1: List all entities first
- STEP 2: Extract claims referencing only STEP 1 entities

This reduces orphaned entity references without doubling API calls.

---

## Improvements Batch 2 (Scale & Latency)

### Centralized Embedding Singleton (`embeddings.py`)
**Problem:** `SentenceTransformer('all-MiniLM-L6-v2')` was instantiated separately in `dedup.py`, `retrieval.py`, and `app.py` — three potential model loads per process.

**Fix:** Created `embeddings.py` with a single `_MODEL` global and `get_model()` / `encode()` helpers. All three modules now import from `embeddings.py`.

### FAISS Vector Index (`vector_store.py`)
**Problem:** Retrieval was calling `model.encode([query])` at search time, then computing cosine similarity against all entities/claims sequentially — O(n) inference per query.

**Fix:** Pre-compute entity and claim embeddings during `run_pipeline.py`. Save FAISS `IndexFlatIP` (inner product on normalized vectors = cosine similarity) to disk. Load at app startup. Queries hit the FAISS index in <10ms regardless of corpus size.

### RRF — Reciprocal Rank Fusion (`retrieval.py`)
**Problem:** Weighted score `(entity_sim × 0.4) + (claim_confidence × 0.3) + (recency × 0.3)` required manual weight tuning and could be dominated by any single signal.

**Fix:** Replace with Reciprocal Rank Fusion:
```
rrf_score(d) = Σ  1 / (k + rank_i(d))   [k=60, standard RRF constant]
```
Two ranking signals are fused: FAISS semantic rank + BM25 keyword rank. RRF is parameter-free (k=60 works well universally) and robust to score distribution differences between signals.

### Semantic Chunking (`extraction.py`)
**Problem:** Emails >2000 words were sent as single large prompts, sometimes exceeding context limits or getting truncated extraction outputs.

**Fix:** `chunk_body()` splits long emails into 400-word chunks with 50-word overlap. Each chunk is extracted separately; results are merged (union of entities and claims). Overlap prevents missing entities that span chunk boundaries.

### Chunked CSV Loading (`download_corpus.py`)
**Problem:** `pd.read_csv("dataset/Enron_emails.csv")` loads the full 918MB file into memory at once — OOM risk on low-memory machines.

**Fix:** `pd.read_csv(..., chunksize=10000)` reads in 10K-row batches. The 2001 filter is applied per chunk; surviving rows are concatenated. Memory usage is bounded to ~20MB at any point.

### Bulk SQLite Inserts (`graph_builder.py`)
**Problem:** Individual `conn.execute()` calls in a Python loop create one SQLite transaction per row — slow for 1000+ entities/claims.

**Fix:** `conn.executemany()` batches all rows into a single transaction, achieving 10-50× speedup for batch inserts.

### Async Concurrent Extraction (`run_pipeline.py`)
**Problem:** `extract_batch()` was sequential — each email waited for the previous LLM call to complete. With 200 emails × ~1s per call = ~3.3 minutes sequential.

**Fix:** `async_extract_batch()` uses `asyncio` + `asyncio.Semaphore(20)` to cap concurrency at 20 simultaneous LLM calls. Wall-clock time drops from ~3.3 minutes to ~20-30 seconds for 200 emails.

### DBSCAN/Agglomerative Clustering (`dedup.py`)
**Problem:** O(n²) pairwise cosine similarity loop in `deduplicate_entities()` — for 1000 entities of the same type, this is 500K comparisons.

**Fix:** Compute the full cosine similarity matrix as a single vectorized `numpy` matmul. Use `sklearn.cluster.AgglomerativeClustering` with `metric='cosine'` and `linkage='average'` to identify merge clusters in O(n² log n) — but critically, the pairwise matrix computation is a single BLAS operation vs. 500K Python iterations.

---

## Improvements Batch 3 (Audit & Bug Fixes)

Full post-implementation audit against `TASK.md` and `CLAUDE.md` uncovered 13 bugs across two fix sessions. All were resolved with TDD (tests written first).

### Bugs Fixed (Batch 3 — sessions 1 & 2)

| # | Location | Description |
|---|----------|-------------|
| 1 | `run_pipeline.py` | `valid_from=ts` now passed to Claim — temporal superseding works correctly |
| 2 | `run_pipeline.py` | Empty `subject_name` guard prevents blank/anonymous entity nodes |
| 3 | `run_pipeline.py` | `first_seen=ts, last_seen=ts` set on Entity creation |
| 4 | `app.py` | Embedding model injected into `embeddings._MODEL` not `retrieval._MODEL` |
| 5 | `app.py` | `VectorStore.load()` result injected into `retrieval._VECTOR_STORE` at startup |
| 6 | `dedup.py` | Conflict detection splits `entity_rel` vs `attr_claims` before comparing `obj_ids` |
| 7 | `run_pipeline.py` | Evidence `char_start`/`char_end` computed via `body_text.find(excerpt)` |
| 8 | `dedup.py` / `retrieval.py` | Removed duplicate `_MODEL` singleton; all modules use `embeddings.py` |
| 9 | `retrieval.py` | Removed double cosine similarity computation per claim (was encoding twice) |
| 10 | `run_pipeline.py` | Object entity auto-create guard — only creates entity if name is non-empty |
| 11 | `graph_builder.py` | SQLite evidence offsets `char_start`/`char_end` persisted correctly |
| 12 | `dedup.py` | O(N²) quoted dedup replaced with set-based substring check |
| 13 | `dedup.py` | O(E×M) entity ID remap replaced with single-pass dict lookup |

**Test count after Batch 3: 63/63**

Note: Bug 19 (UUID strings in claim embeddings) and Bug 18 (object-side claims missing from retrieval) are addressed in the next batch (Batch 4).

---

## Improvements Batch 4 (Audit — Bugs 13, 16, 18, 19)

TDD session: tests written first (red), then fixes (green), 63 → 68 tests.

| # | Location | Description |
|---|----------|-------------|
| 13 | `schema.py` | `Claim.confidence` now `Field(ge=0.0, le=1.0)` — out-of-range LLM values raise ValidationError |
| 16 | `schema.py` | `RawClaimExtraction.subject` validator rejects empty strings (mirrors excerpt validator) |
| 18 | `retrieval.py` | `get_claims_for_entity()` now returns claims where entity is **subject OR object** |
| 19 | `retrieval.py`, `vector_store.py` | Claim texts use canonical entity names not UUID strings — semantic claim search is now meaningful |

**Test count after Batch 4: 68/68**

---

## Improvements Batch 5 (Data-Driven Ontology)

- **Problem:** LLM was already producing semantically valid types (`approved`, `rejected`, `informed`, `proposed`, `agreed`, `authorized`) that failed Pydantic validation and were silently dropped, reducing extraction yield.
- **Fix:** `pipeline/discover_claim_types.py` — samples N emails biased toward longer bodies, calls LLM with open-ended prompt (no constrained `claim_type` list), normalizes free-text relationship labels with `re.sub`, applies synonym clustering via `SYNONYMS` dict, prints frequency table + suggested enum additions.
- **ClaimType expanded:** 11 → 17 values. The 6 new types were the most frequent LLM-generated labels not previously in the enum.
- **PROMPT_TEMPLATE updated:** All 17 types listed verbatim in the `claim_type` field of the extraction prompt.
- **Drift-prevention tests added:** `TestClaimTypeDiscovery` (1 test) + `TestClaimTypeConsistency` (2 tests) — fail if enum and prompt diverge in either direction.
- **Test count:** 72 → 75

---

## Improvements Batch 6 (Multi-Hop Graph Reasoning via Kùzu)

- **Problem:** 1-hop retrieval misses complex relational queries requiring N-hop traversal, e.g. *"Did anyone who reports to Kenneth Lay discuss the Raptor project?"*
- **Fix:** `memory/kuzu_store.py` — `KuzuGraphStore` class wrapping Kùzu embedded graph DB.
  - Schema: one `Entity` node table (PRIMARY KEY `entity_id`) + one generic `Claim` rel table (extensible without migration when new `ClaimType` values are added).
  - `load(entities, claims)` — drops+recreates tables for idempotency; skips attribute claims (no `object_entity_id`).
  - `neighborhood(entity_id, depth=2)` — iterative 1-hop BFS in both directions; avoids variable-length Cypher path API instability across kuzu versions.
  - `execute_cypher(query, params)` — raw Cypher passthrough using `has_next()` / `get_next()` iterator (no pandas dependency).
- **Pipeline integration:** Step 9.5 in `pipeline/run_pipeline.py` after FAISS build — constructs `KuzuGraphStore`, loads entities+claims, injects into `memory.retrieval._KUZU_STORE`.
- **Retrieval:** `_KUZU_STORE` module-level global added to `memory/retrieval.py`. Signal 4 added to `build_context_pack()` — 2-hop neighborhood expansion for top-3 matched entities, fed as a 4th list into existing RRF fusion.
- **UI:** Advanced Cypher panel added to `app/app.py` inside `st.expander()` (hidden by default). Includes 3 pre-built templates (Custom, All claims of a type, 2-hop neighborhood) and live query execution with `st.dataframe()` output.
- **Config:** `KUZU_DB_PATH = "outputs/kuzu_db"` added to `config.py`.
- **Tests:** 12 new TDD tests in `tests/test_kuzu_store.py` — schema creation, idempotent load, 1-hop, 2-hop, 3-hop, unknown entity, raw Cypher, graceful degradation when `_KUZU_STORE = None`.
- **Test count:** 75 → 87

---

## Final State

- **87/87 tests passing** (schema 17, extraction 14, dedup 13, graph 11, retrieval 12, integration 5, kuzu 12)
- **ClaimType enum:** 17 values (11 original + 6 corpus-discovered)
- **Pipeline output:** 1096 entities, 1074 claims, 1174 evidence, 528 merges
- **Streamlit app:** pyvis graph rendering, entity browser, retrieval panel with grounded evidence cards, Advanced Cypher panel
- **write_up.md:** 12-section design document + §6.8 Kùzu, updated §7 (4 RRF signals), updated §10 tradeoffs
- **README.md:** Updated with correct project structure, 87/87 test count, `kuzu_db/` in outputs, Kùzu in architecture diagram

---

## Key Lessons

1. **FAISS + pre-computation beats on-the-fly embeddings** for any retrieval path that runs more than once.
2. **RRF is universally better than weighted sums** for fusing heterogeneous ranking signals — no tuning required.
3. **Pydantic validators as the extraction contract** catches schema drift immediately when prompt/model changes.
4. **Conservative dedup thresholds** (0.85 cosine) prevent false merges. Person name heuristics handle the residual hard cases.
5. **streamlit-agraph is fragile** — always test graph rendering with a 5-node graph on Day 1 so you know whether to fall back to pyvis.
6. **Module-level singletons + `@st.cache_resource`** are the correct pattern for expensive model objects in Streamlit.
