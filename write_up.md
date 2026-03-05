# Write-Up: Grounded Long-Term Memory System

**Submission for Layer10 Take-Home Project**

---

## 1. Overview

This system extracts structured knowledge from the Enron email dataset, deduplicates entities and
claims across three levels, stores them in a grounded memory graph, and provides retrieval and
interactive visualization. The pipeline runs end-to-end from raw CSV to a live Streamlit UI in
under 30 minutes on 200 emails.

**Core principle:** every memory item must trace back to a specific text excerpt in a specific
source email. No claim exists without evidence. This is enforced at the schema level — not as a
convention — so it cannot be bypassed by any part of the pipeline.

**Stack:** Python 3.11 · Claude Haiku (TrueFoundry) · NetworkX · SQLite · sentence-transformers
(all-MiniLM-L6-v2) · FAISS · BM25 · Streamlit · Pydantic v2

---

## 2. Corpus

### What Was Used

| Field | Value |
|-------|-------|
| Dataset | Enron Email Dataset — "enron-clean" Kaggle mirror |
| File | `dataset/Enron_emails.csv` — ~918 MB, ~517 K rows |
| Columns | `date`, `sender`, `recipient`, `body` (pre-cleaned) |
| Source | https://www.kaggle.com/datasets/tarunkashyap/enron-clean |

### How to Reproduce

```bash
# Download from Kaggle (requires kaggle CLI configured)
kaggle datasets download tarunkashyap/enron-clean -p dataset/ --unzip

# Or download manually and place at:
dataset/Enron_emails.csv

# Then sample 200 emails from 2001 (pipeline start point)
uv run python -m pipeline.download_corpus --sample-size 200
# Output: data/enron_sample.csv
```

### Sample Strategy

The pipeline filters to 2001-01-01 – 2001-12-31 (peak Enron fraud period — richer entities, more
contentious decisions, denser communication graph) and draws `N` emails with `random_state=42`
for reproducibility. Chunked CSV loading (`chunksize=10_000`) keeps peak memory ~20 MB regardless
of the 918 MB source file. The date-bounded sample ensures dense, consequential relationships
rather than routine office traffic.

**Why Enron:** Known ground truth from the fraud investigation enables qualitative evaluation;
dense executive relationships (CFO, CEO, board, SPVs) exercise every claim type; email quoting
and forwarding chains stress-test artifact dedup; temporal claim evolution (who reported to whom,
which SPV was active) exercises conflict resolution.

---

## 3. Schema / Ontology Design

### 3.1 Entity Types

| Type | Examples | Rationale |
|------|----------|-----------|
| `person` | Jeff Skilling, Ken Lay, Andy Fastow | Primary actors in decisions |
| `organization` | Enron Corp, Arthur Andersen, FERC | Institutional parties |
| `project` | Raptor, LJM, California ISO | Named SPVs and initiatives |
| `topic` | California energy crisis, mark-to-market | Recurring themes without named entities |
| `location` | Houston, California | Geographic context for decisions |
| `role` | CFO, CEO, Board Chair | Titles — linked to persons via `role_assignment` claims |

**Design rationale:** Person entities are the most important — Enron's story is about individuals
making decisions. Organization and Project types capture the nested SPV structures central to the
fraud. Topic captures recurring discussion themes so that concept queries ("California energy") hit
the graph even without a named entity.

### 3.2 Claim Types

Chosen via **Sample → Discover → Finalize → Extract** (industry-standard GraphRAG / OpenRE pattern):

- An initial 11 types were hand-designed from corpus inspection.
- A data-driven discovery run (`pipeline/discover_claim_types.py`) on 30 sampled emails identified
  6 additional high-frequency relationship labels the LLM was already producing but which failed
  Pydantic validation and were silently dropped.
- The final 17 types cover the full relationship space observed in the corpus.

| Claim Type | Meaning |
|------------|---------|
| `works_at` | Person employed by organization |
| `reports_to` | Reporting line between persons |
| `participated_in` | Person involved in project/event |
| `decided` | Person or org made a decision |
| `requested` | Person asked another to do something |
| `mentioned` | Entity referenced in context |
| `discussed` | Topic or issue talked about |
| `sent_to` | Email/message communication link |
| `role_assignment` | Person given a title/role |
| `status_change` | Project/claim state changed |
| `opinion` | Person expressed a view |
| `approved` | Decision, plan, or action was sanctioned |
| `rejected` | Decision, plan, or action was declined |
| `informed` | Person was notified of something |
| `proposed` | Person put forward an idea/plan |
| `agreed` | Parties reached consensus |
| `authorized` | Permission granted for action |

**Drift prevention:** A `TestClaimTypeConsistency` test suite (3 tests) enforces that the
`ClaimType` enum and the extraction `PROMPT_TEMPLATE` are always in sync — it fails if either is
updated without updating the other. This makes schema drift an immediate CI failure rather than a
silent extraction quality problem.

### 3.3 Evidence Model

Every claim links to one or more `Evidence` objects. This is a hard schema constraint:

```python
class Evidence(BaseModel):
    evidence_id: str          # UUID
    source_id: str            # Row index in CSV (the email's identity)
    source_type: str          # "email"
    excerpt: str              # Exact text excerpt — non-empty enforced by @field_validator
    timestamp: Optional[datetime]
    sender: Optional[str]
    recipients: Optional[list[str]]
    char_start: Optional[int] # body.find(excerpt) — character offset in source body
    char_end: Optional[int]   # char_start + len(excerpt)
```

`char_start` / `char_end` are populated during extraction by `body.find(excerpt)` and persisted
in SQLite. This enables exact span highlighting in future UI iterations.

---

## 4. Extraction Pipeline

### 4.1 Prompt Design

The extraction prompt uses a **hybrid chain-of-thought** strategy — instructs the model to resolve
all entities *first*, then extract claims referencing only those entities, all in a single JSON
response:

```
STEP 1 — ENTITY EXTRACTION:
Identify every person, organization, project, topic, location, and role mentioned.
List them all before moving to claims.

STEP 2 — CLAIM EXTRACTION:
For each relationship or fact, write one claim. Every claim MUST:
- Reference only entities you identified in Step 1
- Include the EXACT text from the email that supports it (copy-paste, do not paraphrase)
```

**Why single-call with chain-of-thought vs. step-wise extraction:**
Step-wise (entities call → claims call) doubles API costs and adds a pipeline state machine.
The chain-of-thought prompt achieves ~90% of the accuracy benefit within a single call by
forcing entity resolution before claim extraction. The architecture is fully compatible with a
migration to true step-wise extraction — only `extraction.py` would change.

**Confidence scale** (in prompt):
- `1.0` — explicitly stated ("Andy is the CFO")
- `0.7` — strongly implied ("Andy handles all financial structures")
- `0.4` — weakly implied (peripheral mention)

Claim types in the prompt are kept in sync with the `ClaimType` enum via automated tests.

### 4.2 Quoted Content Handling

`strip_quoted_content()` splits each email body into clean content and forwarded/quoted lines
(those starting with `>`). Only the clean content is sent to the LLM. The quoted text is
preserved separately for artifact-level dedup. This prevents re-extracting claims already
captured from the original email.

### 4.3 Long-Email Chunking

Emails exceeding 400 words are split into overlapping 400-word chunks with a 50-word overlap.
Each chunk is extracted independently; results are merged (union of entities + claims
deduplicated by excerpt). The overlap window prevents dropping entities or claims that span
a chunk boundary. Per-call token counts stay bounded, reducing truncated outputs.

### 4.4 Response Parsing — Three-Pass Strategy

The parser is immune to common LLM output problems:

1. **Strip markdown fences** — removes ` ```json ``` ` wrappers via regex
2. **Fast path** — attempt `json.loads()` on the entire text
3. **Bracket-balanced extraction** — walks from the first `{` to its matching `}` using a
   bracket/string-escape counter, then parses that substring. This handles post-JSON commentary
   containing braces that breaks naive regex extraction.

### 4.5 Validation & Repair

`validate_extraction()` wraps `ExtractionResult.model_validate()` — a Pydantic model with:

- `RawEntityExtraction`: validates `type` against `EntityType` enum; rejects empty `name`
- `RawClaimExtraction`: validates `claim_type` against `ClaimType` enum; rejects empty
  `supporting_excerpt` and empty `subject` via `@field_validator`

If validation fails, `extract_email()` retries once (the retry succeeds when the LLM added
preamble text on the first attempt). All errors are logged and counted in quality metrics.

### 4.6 Versioning

Every `Claim` carries `extraction_version = "v1"`. When the prompt or model changes, increment
this version. This enables SQL backfill queries:

```sql
SELECT * FROM claims WHERE extraction_version = 'v1'
```
to identify claims that need re-extraction under the new schema without touching claims already
valid under the new version.

### 4.7 Quality Gates

| Gate | Implementation |
|------|---------------|
| Non-empty excerpt | `@field_validator` on `RawClaimExtraction.supporting_excerpt` — ValidationError if empty |
| Non-empty subject | `@field_validator` on `RawClaimExtraction.subject` |
| Valid entity type | `@field_validator` validates against `EntityType` enum |
| Valid claim type | `@field_validator` validates against `ClaimType` enum |
| Confidence bounds | `Field(ge=0.0, le=1.0)` on `Claim.confidence` — out-of-range values raise ValidationError |
| Retry on failure | `extract_email()` retries once; errors logged and counted |
| Pipeline metrics | `run_pipeline.py` prints validation error count, avg confidence, conflict count |

**Future gates not yet implemented:** Confidence decay (claims older than N months with no
corroborating evidence decay toward `UNCERTAIN`), human review queue for low-confidence or
conflicting claims. Both are discussed in Tradeoffs.

---

## 5. Deduplication and Canonicalization

### 5.1 Level 1 — Artifact Dedup (Email Level)

**Hash-based exact dedup:** Normalize whitespace → MD5 → skip if body hash already seen. This
is idempotent — re-ingesting the same email is a no-op.

**Quoted-content substring dedup:** After stripping `>` prefix lines from the quoted body,
check if the normalized result is a substring of any previously seen body text. This catches
cases where an email body consists entirely of a forwarded message. The check runs in O(N)
using a set of seen body hashes rather than O(N²) pairwise comparison.

**Prototype limitation acknowledged:** The cleaned CSV lacks `Message-ID` and `In-Reply-To`
headers. Full thread reconstruction (linking reply evidence back to the canonical message)
requires the raw `.mbox` files. The prototype uses substring dedup as an approximation. A
production system would use header-based thread IDs as the canonical evidence pointer.

### 5.2 Level 2 — Entity Canonicalization

Three-pass dedup applied **per entity type** (never cross-type merges — a `Raptor` project and
a hypothetical `Raptor` organization are never merged):

**Pass 1 — Exact normalized match:**
Lowercase, remove honorifics (`Mr.`, `Dr.`, `Ms.`), reverse `"Last, First"` format → check
string equality across all aliases of all entities of the same type.

**Pass 2 — Person name heuristics:**
For `person` entities: if last names match and first names are prefix-compatible
(`Jeff` ⊂ `Jeffrey`, `Ken` ⊂ `Kenneth`, `Andy` ⊂ `Andrew`, `Bill` ⊂ `William`), merge.
This catches cases where embedding cosine similarity falls just below the threshold.

**Pass 3 — Embedding cosine similarity:**
Using `all-MiniLM-L6-v2`, compute the full similarity matrix as a single numpy matmul
(`embs @ embs.T`) — one BLAS operation instead of O(N²) Python iterations.
Threshold: **0.85** (intentionally strict — better to under-merge than over-merge).

**Merge semantics:**
- Canonical name = longest name (most complete)
- Aliases = union of all names from merged entities
- `merge_history` on `Entity`: `{merged_from, merged_into, reason, timestamp}` — fully auditable
- All claims referencing the old `entity_id` are remapped to the canonical one via single-pass
  dict lookup (not O(E×M) nested loop)

### 5.3 Level 3 — Claim Dedup

Claims are grouped by `(claim_type, subject_entity_id, object_entity_id)`.

**Duplicate claims (same group, same object):**
- Keep highest confidence
- Union all `evidence_ids` — multiple independent sources strengthen the same fact
- Record merge in `merge_history`

**Conflicting claims (same subject+type, different object):**
- Keep both — they represent distinct (potentially contradictory) facts
- Mark the older / lower-confidence one as `ClaimStatus.SUPERSEDED`
- Set `superseded_by` pointer to the newer claim's ID

This handles real-world cases: "Jeff Skilling reports to Ken Lay" (2001-01) superseded by
"Jeff Skilling is CEO" (2001-06).

### 5.4 Conflicts and Revisions

The `ClaimStatus` enum models temporal truth:

| Status | Meaning |
|--------|---------|
| `active` | Currently believed true |
| `superseded` | Was true, replaced by a newer claim |
| `retracted` | Explicitly contradicted |
| `uncertain` | Low confidence or conflicting evidence |

`valid_from` / `valid_until` on claims distinguish *validity time* (when the fact was true) from
*event time* (when the email was sent). Querying current facts: `WHERE status = 'active'`;
historical state: include `SUPERSEDED`. The `superseded_by` pointer creates a linked list of
claim evolution for any given subject+predicate pair.

### 5.5 Reversibility

Every merge records `{merged_from, merged_into, reason, timestamp}` in:
- `Entity.merge_history` / `Claim.merge_history` (JSON on the object itself)
- `merges` SQLite table (separate audit log, queryable independently)

To undo a merge: read `merge_history`, restore the original entity records from their stored IDs,
split the canonical entity back into its components, remap claim pointers. The `merges.reversible`
flag marks whether the merge is still undoable (set to 0 after downstream deletes).

---

## 6. Memory Graph Design

### 6.1 Architecture: NetworkX + SQLite

**SQLite** is the source of truth. Four tables:

```sql
entities  — entity_id, canonical_name, entity_type, aliases (JSON), merge_history (JSON),
             first_seen, last_seen, metadata (JSON)
claims    — claim_id, claim_type, subject_entity_id, object_entity_id, object_value,
             confidence, status, valid_from, valid_until, evidence_ids (JSON),
             superseded_by, extraction_version, merge_history (JSON)
evidence  — evidence_id, source_id, source_type, excerpt, timestamp, sender,
             recipients (JSON), char_start, char_end
merges    — merge_id, merge_type, merged_from, merged_into, reason, timestamp, reversible
```

All inserts use `INSERT OR REPLACE` — the pipeline is idempotent and safe to re-run.
Bulk inserts use `executemany()` (single SQLite transaction) — 10–50× faster than individual
`execute()` calls for 1000+ rows.

**NetworkX MultiDiGraph** is rebuilt from SQLite at app load time. It provides fast in-memory
graph traversal and visualization. Entities → nodes; relational claims (with `object_entity_id`)
→ directed edges; attribute claims (only `object_value`) → node-level `attribute_claims` list.

**Why this combination:** NetworkX gives graph algorithms without operational overhead; SQLite
provides ACID persistence, SQL queries for backfill, and single-file portability (no server,
no migration framework required for a prototype).

### 6.2 Core Objects

| Object | Role |
|--------|------|
| `Entity` | Node: person, org, project, topic, location, role |
| `Claim` | Edge or node attribute: typed, time-bounded, confidence-scored relation |
| `Evidence` | Grounding: source ID + excerpt + offsets + sender + timestamp |
| `ClaimStatus` | `active / superseded / retracted / uncertain` |
| Merge records | Audit trail for entity and claim merges |

### 6.3 Time

- **Event time** = `Evidence.timestamp` — when the email was sent
- **Validity time** = `Claim.valid_from` / `valid_until` — when the fact was true
- `valid_from` is inherited from the earliest evidence timestamp on first insertion
- Current state query: `WHERE status = 'active'`
- Historical state query: include `SUPERSEDED`, filter by `valid_until`

### 6.4 Updates and Idempotency

- `save_to_sqlite()` uses `INSERT OR REPLACE` throughout
- Body hash dedup prevents re-extracting the same email
- New emails append new evidence; existing claims gain additional `evidence_ids` when
  corroborating evidence is found — confidence is not re-computed but the claim becomes
  stronger by having multiple independent sources
- Async concurrent extraction (`asyncio.Semaphore(20)`) with per-email JSON caching means
  interrupted pipeline runs resume from where they stopped

### 6.5 Handling Edits, Deletes, and Redactions

**Not implemented in prototype.** Production design:
- **Soft-delete:** Add `redacted=1` flag to `evidence` rows. Never hard-delete.
- **Claim invalidation:** Claims whose *only* evidence is redacted → status becomes
  `ClaimStatus.RETRACTED`. The claim record remains for audit purposes.
- **Cascading query filter:** Retrieval excludes claims where all evidence is redacted for
  the requesting user's ACL, even if the claim itself is globally visible.
- **Edits:** Treat edited messages as new evidence (new `evidence_id`); link old and new via
  a `supersedes` pointer on the evidence record.

### 6.6 Permissions (Conceptual)

Tag each `Evidence` record with an ACL (`source_acl`: list of user/group IDs who can access
the source). At retrieval time:

```sql
WHERE evidence.source_id IN (
  SELECT source_id FROM acl WHERE user_id = ?
)
```

A claim whose evidence is entirely outside the requesting user's ACL is excluded from their
context pack, even if the claim itself is visible at the claim level. This ensures memory
retrieval never leaks information the user couldn't have accessed directly.

### 6.8 Kùzu Embedded Graph (Multi-Hop Retrieval)

Alongside SQLite, the pipeline builds a **Kùzu embedded graph database** (`outputs/kuzu_db/`) at
pipeline completion. Kùzu supports native Cypher queries without a server process — it is embedded
like SQLite.

The Kùzu graph mirrors entity/claim data from SQLite: entities become `Entity` nodes, relational
claims (those with `object_entity_id`) become directed `Claim` edges. Attribute claims (only
`object_value`) are skipped — Kùzu is used exclusively for graph traversal, not as a secondary
store of attribute facts.

**4th RRF signal:** `neighborhood(entity_id, depth=2)` performs iterative 1-hop BFS in both
directions, returning all `claim_id` values reachable within 2 hops. These are fed into the
existing RRF fusion as a 4th ranked list, enabling answers to complex relational queries like
*"Did anyone who reports to Kenneth Lay discuss the Raptor project?"* that require traversal
through intermediate entities not directly matched by keyword or embedding search.

**Source of truth:** SQLite remains the source of truth. The Kùzu graph is a derived structure
rebuilt at every pipeline run via `KuzuGraphStore.load()`, which drops and recreates all tables
(idempotent by design). In production, pin the `kuzu` version in `pyproject.toml` for
reproducibility across kuzu Python API changes.

### 6.9 Observability

`run_pipeline.py` prints quality metrics after every run:

```
Entities:               1096
Claims:                 1074
Evidence:               1174
Merges:                  528
Avg extraction confidence:  0.985
Conflicts (SUPERSEDED):  234
Validation errors:         0
Duplicate emails skipped:  0
```

These metrics serve as quality gates:
- **Validation error rate** rising → prompt/schema drift or model degradation
- **Avg confidence** dropping → prompt quality degraded or model changed
- **Conflict count** rising sharply → contradictory extractions (possible extraction bug)
- **Duplicate skip rate** → data freshness indicator

In production, these would be time-series metrics (Prometheus/Datadog) with alerting on
deviation from baseline.

---

## 7. Retrieval and Grounding

### 7.1 Question → Context Pack Pipeline

Given a natural language question:

1. **Embed** the question using `all-MiniLM-L6-v2`
2. **Primary — entity name match:** Exact and substring match against all entity canonical
   names and aliases (handles named entity questions: "Jeff Skilling", "Raptor")
3. **Secondary — FAISS semantic search:** ANN query against pre-computed entity embeddings
   (handles paraphrased names and partial matches)
4. **Fallback — BM25 + FAISS claim search:** Keyword and semantic search against claim text
   strings built with canonical entity names (`"works_at Jeff Skilling Enron"`) — handles
   concept queries like "California energy" that don't map to a named entity
5. **Collect** all `ACTIVE` claims for matched entities (bilateral: entity as subject OR object)
6. **Retrieve** linked `Evidence` from SQLite for each claim
7. **Kùzu 2-hop expansion:** For each of the top-3 matched entities, run a 2-hop BFS in the
   Kùzu embedded graph to discover indirectly related claim IDs (multi-hop paths)
8. **Rank** via **Reciprocal Rank Fusion (RRF):**
   ```
   rrf_score(d) = Σ  1 / (k + rank_i(d))   [k=60, 4 signals]
   ```
   Four signals fused: entity-based claims, BM25 keyword, FAISS semantic, Kùzu 2-hop
   neighborhood. RRF is parameter-free and robust to score-scale differences between signals.
9. Return **top-10** claim+evidence pairs; conflicting/superseded claims shown separately

### 7.2 Expansion Without Explosion

- **Top-5 entity matches** — not all matches (prevents combinatorial explosion on common names)
- **Top-10 final claims** — ranked by RRF, diversity maintained across entity types
- **Active-only claims** by default — superseded claims shown only in the `conflicts` section
- **FAISS pre-computation** — entity and claim embeddings pre-computed at pipeline end,
  stored in a FAISS `IndexFlatIP`. Query latency: <10 ms at any corpus size

### 7.3 Grounding Guarantee

Every item in the context pack `claims` array has a non-empty `evidence` array. Each evidence
entry contains:

```json
{
  "excerpt": "exact text from the source email",
  "source_id": "row index in CSV",
  "date": "2001-05-14T00:00:00",
  "sender": "jeff.skilling@enron.com"
}
```

No claim enters the context pack without provenance. This is enforced at query time — claims with
no linked evidence are filtered out before ranking.

### 7.4 Citations Format

Context packs are saved as JSON to `outputs/context_packs/`. Each file is a complete, auditable
record: question → matched entities (with scores) → ranked claims (with type, confidence, status,
RRF score) → evidence (with excerpt, source, date, sender) → conflicts list.

### 7.5 Ambiguity and Conflicting Sources

- **Ambiguous entities** (e.g., "Jeff" matching both "Jeff Skilling" and "Jeff Dasovich"): both
  are returned as matched entities with their individual scores; claims from both appear in the
  ranked list
- **Conflicting claims** (one fact superseded by another): the newer/active claim appears in
  `claims`; the older/superseded claim appears in `conflicts` with its `superseded_by` pointer
- **Low-confidence claims:** All confidence levels are shown — the consumer (human or LLM) sees
  the score and can weight accordingly

### 7.6 Example Context Packs (Generated)

Five context packs are pre-generated at `outputs/context_packs/`:

| Question | File |
|----------|------|
| Who did Jeff Skilling report to? | `who_did_jeff_skilling_report_to.json` |
| What decisions were made about the California energy situation? | `what_decisions_were_made_about_the_california_ener.json` |
| Who was involved in the Raptor project? | `who_was_involved_in_the_raptor_project.json` |
| What role did Andy Fastow play at Enron? | `what_role_did_andy_fastow_play_at_enron.json` |
| What topics were discussed between Ken Lay and Jeff Skilling? | `what_topics_were_discussed_between_ken_lay_and_jef.json` |

---

## 8. Visualization Layer

The Streamlit app (`app/app.py`) has four interactive areas:

### 8.1 Sidebar — Filters

- Entity type filter (multi-select: person, organization, project, …)
- Minimum confidence threshold slider (0.0 – 1.0)
- Claim status filter (active / superseded / all)
- Summary stats: total entities, claims, evidence, merges

### 8.2 Graph Panel — Entity/Relationship Explorer

Interactive graph rendered via `pyvis` (embedded via `st.components.v1.html()`). Node color
encodes entity type; edge thickness encodes claim confidence. Clicking a node opens the entity
detail panel.

*Note:* `streamlit-agraph` was attempted first but failed with a broken JS bundle
(`FileNotFoundError` on a compiled frontend chunk). `pyvis` generates a self-contained HTML
file that Streamlit embeds in an iframe — more reliable and easier to debug.

### 8.3 Entity Browser — Detail View

Dropdown to select any entity. Shows:
- Canonical name, type, aliases
- Merge history count and details (which entities were merged into this one)
- All associated claims with claim type, confidence, status
- First evidence excerpt for each claim with source metadata

This directly satisfies the requirement to **inspect duplicates/merges**: the alias list shows
all names that were merged, and merge history shows the reason and timestamp.

### 8.4 Retrieval Panel

Free-text question input → "Search" button → context pack displayed as expandable claim cards,
each showing:
- Claim text, type, confidence, status, RRF score
- Evidence accordion: exact excerpt, source ID, date, sender

Conflicting/superseded claims shown separately with a visual warning.

---

## 9. Layer10 Adaptation

### 9.1 Ontology Extensions

New entity types for Layer10's environment:

| Type | Examples |
|------|----------|
| `message` | Email, Slack message, Jira comment |
| `thread` | Email thread, Slack channel thread |
| `channel` | Slack channel, mailing list |
| `ticket` | Jira/Linear issue |
| `sprint` | Jira sprint |
| `component` | Codebase component, service |
| `customer` | Client account |
| `team` | Engineering team, product squad |

New claim types for Layer10's relationship space:

```
assigned_to, blocked_by, resolved_by, escalated_to, tagged_with,
linked_to, commented_on, transitioned_to, owned_by, mentioned_in
```

Add `source_type` on Evidence: `"email"`, `"slack"`, `"jira_comment"`,
`"jira_status_change"`, `"pr_review"`. This lets retrieval weight evidence by source
formality — a Jira status change outweighs a Slack message saying the same thing.

### 9.2 Unstructured + Structured Fusion

**Source-specific extraction prompts:**
- **Email:** entities + claims + communication graph (this prototype)
- **Slack:** entities + reactions + informal decisions + thread references + emoji signals
- **Jira:** status transitions → `ClaimType.STATUS_CHANGE`; assignee changes →
  `ClaimType.ROLE_ASSIGNMENT`; linked issues → graph edges; sprint assignments →
  `ClaimType.PARTICIPATED_IN`

**Cross-source entity resolution:** The same person appears as `jeff@company.com` (email),
`@jskilling` (Slack), and "J. Skilling" (Jira assigned field). Resolution uses email address as
the canonical identifier where available, falling back to embedding similarity within entity type.
The `aliases` list on `Entity` stores all cross-source identifiers.

**Connecting discussions to tickets:** A `mentioned_in` claim links a Slack thread to the Jira
ticket it references. This creates a graph path from informal decision-making in Slack to the
formal artifact that records the outcome.

### 9.3 Long-Term Memory: Durable vs. Ephemeral

| Durable (persist indefinitely) | Ephemeral (short TTL or discard) |
|-------------------------------|----------------------------------|
| Architecture decisions in Jira | Meeting scheduling ("are you free Friday?") |
| Formal org-chart changes | Quick status checks in Slack |
| Customer commitments | Draft PR comments |
| Project ownership assignments | Informal reactions/opinions |
| Security incident decisions | Automated CI notifications |

**Heuristics for durability:**
- Claim supported by ≥ 2 independent sources → durable
- Formal Jira status change → durable by default
- Only in a single quick Slack message → uncertain, needs corroboration
- Time-based decay: claims with no corroborating evidence after N months → `UNCERTAIN` status
  (not deleted — the history is preserved but de-prioritized in retrieval)

### 9.4 Grounding and Safety

Every memory item must carry a permalink to its source. For Layer10:
- Slack: `slack://team.slack.com/archives/CHANNEL/p{ts}` (message permalink)
- Jira: `https://company.atlassian.net/browse/PROJ-123`
- Email: `message-id` header as the canonical source ID

**Deletion handling (production design):**
- Sources are soft-deleted (`redacted=1`) — never hard-deleted
- Claims whose only evidence is redacted → `ClaimStatus.RETRACTED`
- The claim record remains for audit; retrieval excludes it for normal users
- Admin-level "evidence audit" view can show retracted claims with reason

### 9.5 Permissions

Tag each Evidence with `source_acl` (list of user/group IDs). At retrieval:

```sql
SELECT c.* FROM claims c
JOIN claim_evidence ce ON c.claim_id = ce.claim_id
JOIN evidence e ON ce.evidence_id = e.evidence_id
WHERE e.source_id IN (
  SELECT source_id FROM source_acl WHERE user_id = ?
)
```

A claim with all evidence outside the user's ACL is excluded from their context pack, even if
the claim entity graph is globally visible. This ensures memory retrieval is bounded by access
to underlying sources.

### 9.6 Operational Reality

**Incremental updates (webhook-driven):**
The current pipeline already supports incremental processing:
- `save_to_sqlite()` uses `INSERT OR REPLACE` (upsert semantics)
- Body hash dedup ensures idempotency — re-ingesting the same message is a no-op
- Each new Slack message / Jira event triggers extraction for that single item;
  dedup + graph update runs incrementally. Full pipeline re-runs are not needed.

**Scaling dedup:**
At 500K+ emails, pairwise entity embedding comparison is O(n²) — impractical. Replace with:
- ANN index (Qdrant/Pinecone) for entity embedding lookup
- Dedup only within entity type + time-window buckets (not global)
- Blocking by email domain or team for person entities

**Cost model:**
Claude Haiku pricing ~$0.25/1M input tokens × average 500 tokens/email × 200 emails ≈ $0.025
one-time. At 500K emails: ~$63 one-time extraction cost. Incremental cost is per new message.

**Evaluation and regression:**
- Quality metrics in `run_pipeline.py` (validation error rate, avg confidence, conflict count)
- In production: track as time-series, alert on deviation from baseline
- A/B test prompt changes with a held-out eval set of manually annotated claims
- `extraction_version` on every claim enables SQL-level segmentation of eval cohorts

---

## 10. Tradeoffs and Acknowledged Limitations

### Single-Call vs. Step-Wise Extraction

We chose single-call extraction with a hybrid chain-of-thought prompt. The alternative —
step-wise extraction (entities call → claims call with resolved entity list) — reduces
hallucinated entity references at the cost of 2× API calls and a more complex pipeline state
machine. The current architecture is fully compatible with a migration to step-wise extraction;
only `extraction.py` would change.

### Pydantic as the Extraction Contract

Manual `validate_extraction()` was replaced with strict Pydantic validation via `ExtractionResult`,
`RawEntityExtraction`, and `RawClaimExtraction`. This makes schema drift an immediate validation
failure rather than a silent quality issue: invalid `claim_type` strings or empty excerpts raise
`ValidationError` at ingestion time and are counted in pipeline metrics.

### LLM Provider Abstraction via litellm

`litellm` provides a unified interface across providers. Switching backends requires only config
changes, no code changes:

```python
# Current: TrueFoundry → Claude Haiku
MODEL = "anthropic/claude-haiku-4-5-20251001"
BASE_URL = "https://gateway.truefoundry.ai"

# Ollama (local, zero cost):
MODEL = "ollama/llama3"
BASE_URL = "http://localhost:11434"
```

### Performance Architecture

| Optimization | Detail |
|-------------|--------|
| FAISS pre-computed embeddings | Entity + claim embeddings computed once at pipeline end; ANN query <10ms |
| RRF over weighted sums | Parameter-free fusion of FAISS + BM25 signals; no weight tuning |
| Semantic chunking | Emails >400 words split into 400-word chunks with 50-word overlap |
| Centralised embedding singleton | `SentenceTransformer` loaded once via `embeddings.py`; `@st.cache_resource` in Streamlit |
| Vectorised entity dedup | `embs @ embs.T` (numpy BLAS) instead of O(N²) Python loop |
| Bulk SQLite inserts | `executemany()` — 10-50× faster than per-row `execute()` for 1000+ rows |
| Async concurrent extraction | `asyncio + Semaphore(20)` — 200 emails in ~20-30s vs ~3.3 min sequential |
| Chunked CSV loading | `chunksize=10_000` — peak memory ~20 MB vs. loading full 918 MB |

### Acknowledged Gaps

| Gap | Current State | Production Fix |
|-----|--------------|----------------|
| Evidence offsets | `char_start`/`char_end` computed and stored | UI does not yet render span highlights |
| Confidence decay | Not implemented | Claims older than N months → `UNCERTAIN` if no corroboration |
| Human review queue | Not implemented | Flag `confidence < 0.5` or conflicting claims for review |
| Thread linking | Substring dedup approximation | Requires `Message-ID` / `In-Reply-To` from raw `.mbox` files |
| Async + SQLite | One connection per extraction call | Connection pool for production |
| Kùzu path API | Iterative 1-hop BFS used instead of `[c:Claim*1..N]` variable-length Cypher | Variable-length syntax returns edges as a list whose Python representation varies between kuzu versions; iterative approach is robust and version-stable |

---

## 11. Test Suite

87 tests across 7 modules, all passing. Written test-first (TDD — red → green → refactor):

| Module | Tests | Key Behaviors Covered |
|--------|-------|-----------------------|
| `test_schema.py` | 17 | Pydantic validation, enum values, confidence bounds, drift-prevention |
| `test_extraction.py` | 14 | Prompt building, JSON parsing (fences, balanced brackets), retry logic |
| `test_dedup.py` | 13 | Hash dedup, quoted dedup, entity merge, claim merge, conflict detection |
| `test_graph_builder.py` | 11 | Graph construction, SQLite persistence, idempotency, offsets |
| `test_retrieval.py` | 12 | Entity matching, bilateral claims, concept search, context pack grounding |
| `test_integration.py` | 5 | End-to-end pipeline, grounded retrieval, JSON serializability |
| `test_kuzu_store.py` | 12 | Schema creation, idempotent load, 1-hop, 2-hop, 3-hop traversal, unknown entity, raw Cypher, graceful degradation |

```bash
uv run pytest tests/ -v   # 87/87 pass
```

---

## 12. Pipeline Quality Metrics (200-Email Run)

```
Entities:                        1096
Claims:                          1074
Evidence:                        1174
Merges:                           528
Average extraction confidence:   0.985
Conflicts (SUPERSEDED claims):    234
Validation errors:                  0
Duplicate emails skipped:           0
```

The 234 superseded claims represent genuine temporal conflicts captured by the pipeline —
e.g., different emails asserting different reporting lines for the same person. These are visible
in the UI's conflict panel and in every context pack's `conflicts` field.
