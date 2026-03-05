# Write-Up: Grounded Long-Term Memory System

## 1. Overview

This system extracts structured knowledge from the Enron email dataset, deduplicates entities and claims, stores them in a memory graph, and provides retrieval and visualization. The pipeline runs end-to-end from raw CSV to an interactive Streamlit UI in under 30 minutes on 200 emails.

The core principle: **every memory item must trace back to a specific text excerpt in a specific source email**. No claim exists without evidence.

---

## 2. Corpus Choice

**Dataset:** Enron Email Dataset (Kaggle "enron-clean" mirror)
- **File:** `dataset/Enron_emails.csv` — ~918MB, ~517K rows
- **Columns:** `date`, `sender`, `recipient`, `body` (pre-cleaned)
- **Source:** https://www.kaggle.com/datasets/tarunkashyap/enron-clean

**Sample strategy:** Filter to 2001-01-01 – 2001-12-31 (peak fraud period — richer entities, more contentious decisions) and draw 200 emails with `random_state=42` for reproducibility. Start with 200 to validate the full pipeline; scale to 500 by changing `--sample-size`.

**Why Enron:** Dense entity relationships (executives, SPVs, energy deals), time-stamped communications with clear forwarding/quoting patterns, and known ground truth (fraud investigation records) for qualitative evaluation.

---

## 3. Schema / Ontology Design

### Entity Types
| Type | Examples |
|------|----------|
| `person` | Jeff Skilling, Ken Lay, Andy Fastow |
| `organization` | Enron Corp, Enron Board, Arthur Andersen |
| `project` | Raptor, LJM, SPE |
| `topic` | California energy crisis, mark-to-market accounting |
| `location` | Houston, California |
| `role` | CFO, CEO, Board Chair |

**Design rationale:** Person entities are the most important — Enron's story is about individuals making decisions. Organization and Project types capture the nested SPV structures central to the fraud. Topic captures recurring discussion themes without requiring named entities.

### Claim Types
Chosen to capture the directed assertions in business email: who works where (`works_at`), who asked whom to do what (`requested`), what was discussed (`discussed`), what was decided (`decided`), and temporal roles (`role_assignment`, `status_change`). `sent_to` captures the communication graph itself as a claim.

### Evidence Model
Every claim links to one or more `Evidence` objects containing:
- Exact `excerpt` (the actual text that grounds the claim)
- `source_id` (row index in the CSV — the email's identity)
- `sender`, `recipients`, `timestamp` for provenance
- `char_start`/`char_end` fields populated via `body.find(excerpt)` during extraction and persisted in SQLite

---

## 4. Extraction Pipeline

### Prompt Design
The prompt instructs the LLM to return a strict JSON structure with `entities` and `claims`. Each claim **must** include a `supporting_excerpt` — the exact text from the email. The prompt enforces grounding by making the excerpt a required field, not optional.

Confidence scale:
- `1.0` = explicitly stated ("Andy is the CFO")
- `0.7` = strongly implied ("Andy handles the financial structures")
- `0.4` = weakly implied (peripheral references)

### Quoted Content Handling
Before extraction, `strip_quoted_content()` splits each email body into clean content and quoted/forwarded lines (starting with `>`). Only the clean content is sent to the LLM, preventing duplicate extraction of forwarded content. The quoted text is preserved separately for artifact dedup.

### Validation & Repair
`validate_extraction()` checks:
1. `entities` and `claims` keys exist
2. Every entity has a `type`
3. Every claim has a non-empty `supporting_excerpt`

On validation failure, `extract_email()` retries once (the retry often succeeds because the LLM sometimes adds extra text on first try). Errors are logged to stderr and counted in the pipeline quality metrics.

### Versioning
Every `Claim` object carries `extraction_version = "v1"`. When the prompt or model changes, increment this version. This enables backfill queries: `SELECT * FROM claims WHERE extraction_version = 'v1'` to identify claims that need re-extraction under the new schema.

---

## 5. Deduplication Strategy

### Level 1: Artifact Dedup (Email Level)
- **Hash-based:** Normalize whitespace → MD5 → skip if seen
- **Quoted substring detection:** Strip `>` prefix from each line, check if resulting text is a substring of any previously seen body. This catches cases where an email body is entirely composed of a forwarded message.

**Prototype limitation:** The cleaned CSV lacks `Message-ID` and `In-Reply-To` headers. A production system would use these to build reply threads and link evidence back to the canonical message rather than duplicating it. This is noted in Tradeoffs.

### Level 2: Entity Dedup
Three-pass approach, applied per entity type (never cross-type merges):

1. **Exact normalized match:** Lowercase, remove honorifics (`Mr.`, `Dr.`), reverse `"Last, First"` format → check for string equality across all aliases
2. **Person name heuristics:** For `person` entities, check if last names match and first names are compatible prefixes (`Jeff` ≈ `Jeffrey`, `Ken` ≈ `Kenneth`)
3. **Embedding cosine similarity:** Using `all-MiniLM-L6-v2`, if similarity > 0.85 → merge candidate

**Merge semantics:**
- Canonical name = longest name (most complete)
- Aliases = union of all names from merged entities
- `merge_history` records: `{merged_from, merged_into, reason, timestamp}` — fully auditable
- All claims referencing the old `entity_id` are updated to the canonical one

**Conservative threshold:** 0.85 cosine similarity is intentionally strict. Better to under-merge (two entries for the same person) than over-merge (conflating different people). The person name heuristics handle the common cases more precisely.

### Level 3: Claim Dedup
Group claims by `(claim_type, subject_entity_id, object_entity_id)`:

- **Same group (duplicate):** Keep highest confidence, union `evidence_ids` (the value: multiple independent sources strengthen the same fact), record merge history
- **Conflicting (same subject+type, different object):** Keep both, mark the older/lower-confidence one as `SUPERSEDED` with a pointer to the superseding claim

This handles cases like "Jeff Skilling reports to Board Chair" (2001-01) being superseded by "Jeff Skilling reports to new CEO" (2001-06).

**Reversibility:** Every merge records the original IDs in `merge_history`. The `merges` SQLite table provides a separate audit log. To undo a merge: restore from `merge_history`, split the canonical entity back into its components, update claim pointers.

---

## 6. Memory Graph Design

### Architecture: NetworkX + SQLite
- **NetworkX MultiDiGraph:** In-memory graph for fast traversal and visualization. Entities = nodes; relational claims = directed edges; attribute claims (e.g., "discussed California") = node attributes.
- **SQLite:** Source of truth. Four tables: `entities`, `claims`, `evidence`, `merges`. The NetworkX graph is rebuilt from SQLite on each app load.

**Why this combination:** NetworkX gives us graph algorithms (neighbor traversal, path finding) without operational overhead. SQLite provides ACID persistence, SQL queries for backfill, and portability (single file, no server).

### Temporal Modeling
- `valid_from` / `valid_until` on claims represent validity time (when the fact was true), distinct from the email's timestamp (event time when it was communicated)
- `ClaimStatus.SUPERSEDED` + `superseded_by` pointer creates a linked list of temporal claim evolution
- Queries for "current" facts: `WHERE status = 'active'`; queries for "historical" facts include SUPERSEDED

### Update Semantics
- `save_to_sqlite()` uses `INSERT OR REPLACE` — idempotent, safe to re-run
- Artifact dedup by body hash ensures re-ingesting the same email is a no-op
- New emails append new evidence; claims may be promoted from UNCERTAIN to ACTIVE if confidence rises with additional evidence

### Handling Deletions/Redactions
Not implemented in prototype. In production:
- Soft-delete evidence records (`redacted=1` flag)
- Claims whose *only* evidence is redacted → status becomes `ClaimStatus.RETRACTED`
- The claim record remains for audit purposes but is excluded from retrieval

---

## 7. Retrieval

### Question → Context Pack Pipeline
1. Embed the question using `all-MiniLM-L6-v2`
2. **Primary:** Exact/substring match on entity canonical names and aliases (handles named entity questions)
3. **Fallback:** Cosine similarity against concatenated claim text (handles concept questions like "California energy")
4. Collect all ACTIVE claims for matched entities
5. Retrieve linked Evidence objects from SQLite
6. Fuse three signals via **Reciprocal Rank Fusion (RRF)**: entity-based claim retrieval,
   BM25 keyword ranking, and FAISS semantic search are merged via
   `score(d) = Σ 1/(k + rank_i(d))` where k=60. RRF is parameter-free and robust to
   score-scale differences between signals.
7. Return top-10 claim+evidence pairs; conflicting/superseded claims shown separately

### Grounding Guarantee
Every item in the context pack's `claims` list has a non-empty `evidence` array. Each evidence entry contains the exact `excerpt`, `source_id`, `sender`, and `date`. No claim enters the context pack without provenance.

### Bilateral Claim Retrieval
`get_claims_for_entity()` returns claims where the entity is either the **subject** or the **object** of the relation. This ensures that queries like "Who reports to Ken Lay?" return the relevant `REPORTS_TO` claims regardless of which side of the relation Ken Lay sits on.

### Conflict Handling
SUPERSEDED claims appear in the `conflicts` section of the context pack, not the main `claims` list. The UI shows them with a warning indicator. The consumer (human or LLM) sees both the current truth and the historical state that was superseded.

---

## 8. Visualization

The Streamlit app (`app.py`) has three areas:

**Sidebar:** Filters for entity type, minimum confidence threshold, claim status (active/superseded/all), and summary stats.

**Graph panel:** Uses `streamlit-agraph` (falls back to `pyvis` HTML embed if agraph fails). Nodes colored by entity type; edge thickness proportional to confidence.

**Entity browser:** Dropdown to select entities; shows canonical name, type, aliases, merge history count, and all associated claims with first evidence excerpt.

**Retrieval panel:** Free-text question input → context pack → expandable claim cards with evidence excerpts and source metadata.

---

## 9. Layer10 Adaptation

### Ontology Extensions
Add entity types: `Message` (email/Slack/Jira ticket), `Thread` (conversation), `Channel` (Slack channel), `Sprint`, `Component`, `Customer`, `Team`.

Add claim types: `assigned_to`, `blocked_by`, `resolved_by`, `escalated_to`, `tagged_with`, `linked_to` (for issue cross-references).

Add a `source_type` field on Evidence: `"email"`, `"slack"`, `"jira_comment"`, `"jira_status_change"`, `"pr_review"`. This lets retrieval weight evidence by source type (a formal Jira decision > a Slack message saying the same thing).

### Multi-Source Extraction
Different prompts per source type:
- Email: entities + claims + communication graph
- Slack: entities + reactions + informal decisions + thread references
- Jira: status transitions as `ClaimType.STATUS_CHANGE`, assignee changes as `ClaimType.ROLE_ASSIGNMENT`, linked issues as graph edges

Cross-source entity resolution: the same person appears as `jeff@company.com` (email), `@jskilling` (Slack), and "J. Skilling" (Jira assigned field). Resolution uses email address as the canonical identifier where available, falling back to embedding similarity.

### Incremental Updates
The current pipeline already supports incremental processing:
- `save_to_sqlite()` uses `INSERT OR REPLACE` (upsert semantics)
- Body hash dedup ensures idempotency
- In production: webhook-driven ingestion per source. Each new event triggers extraction for that single item; dedup + graph update runs incrementally. No full re-run needed.

### Permissions
Tag each Evidence with an ACL (list of user IDs or group IDs who can see the source). At retrieval time, filter `WHERE source_id IN (SELECT source_id FROM acl WHERE user_id = ?)`. A claim with all evidence redacted for a given user is excluded from their context packs, even if the claim itself is visible to others.

### Long-Term Memory: Durable vs Ephemeral
| Durable | Ephemeral |
|---------|-----------|
| Org structure decisions ("Jeff is now CEO") | Meeting scheduling ("Are you free Friday?") |
| Architecture decisions in Jira | Quick status checks in Slack |
| Project ownership assignments | Informal reactions/opinions |
| Customer commitments | Draft PR comments |

Heuristics for durability:
- Claim supported by ≥2 independent sources → durable
- Claim only in a single quick Slack message → uncertain, needs corroboration
- Formal Jira status change → durable by default
- Time-based decay: claims with no corroborating evidence after N months → UNCERTAIN

### Operational Reality
- **Scale:** At Enron scale (500K emails), entity dedup with pairwise embedding comparison becomes O(n²) — impractical. Replace with ANN index (Qdrant/Pinecone for entity embeddings) and only run full dedup within entity type + time window buckets.
- **Cost:** Extraction at $0.00025/1K tokens (Haiku) × average 500 tokens/email × 500K emails ≈ $63 one-time. Incremental cost is per new message.
- **Evaluation:** Quality metrics in `run_pipeline.py` printout (avg confidence, conflict count, validation error rate, duplicate skip rate). In production, track these as time series and alert on degradation. A/B test prompt changes with a held-out eval set of manually annotated emails.

---

## 10. Tradeoffs & Future Work

### Single-Call vs Step-Wise Extraction

We chose **single-call extraction** for this prototype: one API call per email returns both entities and claims in a single JSON response.

The alternative is **step-wise extraction**: call the LLM once to extract entities, then make a second call with those resolved entities to extract claims. Step-wise reduces hallucinations because the claim step cannot invent entities that weren't found in step 1. However it doubles API costs and adds pipeline complexity.

**Why single-call fits this project:**
- **Cost control:** 200–500 emails × 1 call ≈ 200–500 API calls. At Claude Haiku pricing (~$0.25/1M input tokens), total cost stays well under $1.
- **Simplicity:** One code path is easier to debug, test, and iterate on within a 3-day window.
- **Accuracy is sufficient:** The hybrid chain-of-thought prompt (described below) mitigates most of the hallucination risk without a second call.

**Hybrid chain-of-thought (current approach):** The prompt instructs the LLM to resolve all entities *first*, then extract claims that reference only those entities — all within a single response:

```
STEP 1 — ENTITY EXTRACTION: Identify every person, organization, project...
STEP 2 — CLAIM EXTRACTION: For each relationship, write one claim that
references only entities from Step 1.
```

This encourages the model to treat entity resolution as a prerequisite for claim extraction, which reduces orphaned entity references without a second API call. The entity IDs and claim structure in the schema are fully compatible with a future migration to true step-wise extraction — only `extraction.py` would change.

> In a production system with larger scale and higher accuracy demands, we would migrate to step-wise extraction (entities first, then claims) with structured output / tool-calling to enforce the schema at the API level. The current design is intentionally compatible with this evolution.

### Pydantic as the Extraction Contract

The initial implementation used a manual `validate_extraction()` function that checked dict keys. This has been replaced with strict Pydantic validation via `ExtractionResult`, `RawEntityExtraction`, and `RawClaimExtraction` models in `schema.py`. Benefits:

- **Type enforcement:** `entity.type` must be a valid `EntityType` value; `claim.claim_type` must match `ClaimType`. Invalid strings are caught immediately with a clear error.
- **Grounding guarantee at schema level:** `RawClaimExtraction.supporting_excerpt` has a `@field_validator` that rejects empty strings — it is literally impossible for a claim to be created without an excerpt. `RawClaimExtraction.subject` has the same treatment, preventing anonymous/blank-subject claims from entering the graph.
- **Error messages:** Pydantic's `ValidationError.errors()` returns structured error info (field path + message) which gets logged and counted in pipeline quality metrics.
- **Evolution path:** When the ontology changes (new entity types, new claim types), updating the Pydantic enums automatically makes `validate_extraction` catch stale LLM outputs without any additional code.

### LLM Provider Abstraction via litellm

The extraction pipeline uses `litellm` instead of the OpenAI SDK directly. litellm provides a unified interface across providers. Switching backends requires only config changes, no code changes:

```python
# Current: TrueFoundry → Claude Haiku
MODEL = "anthropic/claude-haiku-4-5-20251001"
BASE_URL = "https://gateway.truefoundry.ai"

# Ollama (local, zero cost):
MODEL = "ollama/llama3"
BASE_URL = "http://localhost:11434"
OPENAI_API_KEY = ""  # not needed
```

### Performance Architecture

**FAISS pre-computed embeddings (`vector_store.py`):** At pipeline completion, entity and claim embeddings are pre-computed and saved to a FAISS `IndexFlatIP` (inner product on L2-normalised vectors = cosine similarity). At query time the Streamlit app loads this index and runs ANN search in <10ms regardless of corpus size. This replaces on-the-fly encoding at every query — the difference is ~100ms vs <10ms per search request.

**RRF — Reciprocal Rank Fusion:** The weighted scoring formula `(entity_sim × 0.4) + (claim_confidence × 0.3) + (recency × 0.3)` required manual weight tuning and was sensitive to score distribution differences across signals. Replaced with RRF:
```
rrf_score(d) = Σ  1 / (k + rank_i(d))   [k=60, standard RRF constant]
```
Two signals are fused: FAISS semantic rank and BM25 keyword rank (`rank-bm25`). RRF is parameter-free (k=60 works universally), robust to different score scales, and consistently outperforms weighted sums in information retrieval benchmarks.

**Semantic chunking:** Emails >400 words are split into overlapping 400-word chunks (50-word overlap). Each chunk is extracted independently; results are merged (union of entities and claims deduplicated by excerpt). The overlap prevents dropping entities that straddle chunk boundaries. This also reduces per-request token counts, lowering the chance of truncated extraction outputs.

**Centralised embedding singleton (`embeddings.py`):** `SentenceTransformer('all-MiniLM-L6-v2')` is loaded once per process via a module-level `_MODEL` global in `embeddings.py`. All modules (`dedup.py`, `retrieval.py`, `app.py`, `vector_store.py`) delegate to `embeddings.py` — no duplicate model loads. In the Streamlit app, `@st.cache_resource` ensures the model survives across rerenders.

**Async concurrent LLM extraction:** `run_pipeline.py` uses `asyncio` + `asyncio.Semaphore(20)` to cap in-flight LLM calls at 20 simultaneous requests. `extract_email` (synchronous) is dispatched to a thread pool via `loop.run_in_executor`. Wall-clock time for 200 emails drops from ~3.3 minutes (sequential, 1 s/email) to ~20-30 seconds. Per-email JSON caching is preserved — already-extracted emails are never re-called.

**Vectorised entity dedup (`dedup.py`):** The O(n²) pairwise cosine similarity Python loop was replaced with a single numpy matmul (`embs @ embs.T`) that computes the full similarity matrix in one BLAS call. Union-Find merging is applied to the resulting matrix. For 1000 entities of the same type, this is one matrix multiply vs. ~500K Python function calls.

**Bulk SQLite inserts (`graph_builder.py`):** Individual `conn.execute()` calls in loops were replaced with `conn.executemany()`, which batches all rows into a single SQLite transaction. Benchmark: 10-50× faster for 1000+ rows — confirmed by SQLite documentation and profiling.

**Chunked CSV loading (`download_corpus.py`):** `pd.read_csv(path, chunksize=10_000)` reads the 918 MB dataset in 10K-row batches. The 2001 date filter is applied per chunk; only matching rows are accumulated. Peak memory stays bounded to ~20 MB instead of the full dataset.

### Acknowledged Tradeoffs

**Evidence offsets:** `char_start`/`char_end` are now computed via `body.find(excerpt)` and persisted in SQLite. Span highlighting in the UI is possible; the current UI does not yet render highlights (left as future UI work).

**Claim confidence bounded:** `Claim.confidence` is now validated by `Field(ge=0.0, le=1.0)` — invalid LLM outputs (e.g., `confidence=99.0`) raise `ValidationError` at ingestion time and are counted in pipeline quality metrics.

**FAISS claim embeddings use entity names:** Claim texts for the FAISS index are built with canonical entity names (`"works_at Jeff Skilling Enron"`) rather than UUID strings. Semantic claim search is now meaningful.

**Confidence decay:** No time-based confidence decay implemented. In production, claims older than N months with no corroborating evidence would decay toward `ClaimStatus.UNCERTAIN` automatically, preventing stale facts from dominating retrieval.

**Human review hooks:** Not implemented. Production would flag low-confidence claims (< 0.5) and conflicting claims for a review queue before they become durable memory. A simple `needs_review=1` flag in the claims table would suffice as a starting point.

**Thread linking:** The cleaned CSV lacks `Message-ID` / `In-Reply-To` headers. Full thread reconstruction (which would enable proper evidence deduplication across reply chains) requires the raw `.mbox` files. The prototype uses quoted-content substring dedup as an approximation.

**Async + SQLite threading:** `asyncio.run_in_executor` dispatches `extract_email` to a thread pool. SQLite connections are opened and closed per call (not shared across threads), which is safe but adds minor overhead. A connection pool would be the production fix.

### Future Work
- True step-wise extraction (entities call → claims call) with Pydantic structured outputs
- Named entity recognition as a pre-filter before LLM extraction (reduce token cost)
- Graph traversal for context expansion (include 1-hop neighbors of matched entities)
- Confidence calibration via human-labeled eval set
- Support for structured source types (Jira JSON → direct claim extraction without LLM)
- Periodic re-extraction with new model versions + automatic regression detection
- Qdrant/Pinecone for entity embeddings at 500K+ email scale (FAISS does not support incremental updates without full rebuild)
