Layer10 Take-Home Project
Grounded Long-Term Memory via Structured Extraction,
Deduplication, and a Context Graph
2026Layer10 Take-Home
2026
Contents
Project Brief2
Choose a Public Corpus3
Structured Extraction3
Deduplication and Canonicalization3
Memory Graph Design4
Retrieval and Grounding4
Visualization Layer5
Layer10 Considerations5
What to Submit5
How We Evaluate6
1
1Layer10 Take-Home
2026
Project Brief
Layer10 is building a system that turns scattered organizational knowledge (email, chat, docs,
tickets) into grounded long-term memory. We care about correctness over time: messages get
edited, issues change state, decisions get reversed, and sources can be deleted or redacted.
• Choose a publicly available corpus (examples below) that contains unstructured commu-
nication (e.g., email or chat logs) and/or structured work artifacts (e.g., issues).
• Use a free model (open-weight run locally or a free-tier hosted model) to perform struc-
tured extraction into a schema/ontology you define.
• Design and implement a pipeline that produces a memory graph (entities + claims + evi-
dence) with strong grounding and robust deduplication.
• Add a visualization layer so someone can explore the graph and click through to support-
ing evidence.
• Explain how you would adapt the ontology and pipeline to Layer10’s target environment
(email, Slack, Jira/Linear) and long-term memory requirements.
2
2Layer10 Take-Home
2026
Choose a Public Corpus
Use any public dataset you like. If helpful, here are some common choices:
• Enron Email Dataset — email threads, forwarding/quoting, identity resolution chal-
lenges. Example sources: CMU Enron dataset or Kaggle mirrors.
• Open-source project issues — structured + unstructured: GitHub Issues/PRs for a pop-
ular repo; many include discussions, decisions, and changing state.
• Jira-like issue trackers for public projects — tickets, comments, status transitions, as-
signees.
• Mailing list archives (e.g., Apache dev lists) — long-running technical conversations and
decisions.
• Wikipedia Talk pages — discussion + decision-making, citations, reversals.
In your write-up, state exactly which corpus you used, where you obtained it, and how to
reproduce the download.
Structured Extraction
Design an extraction approach that produces typed, grounded objects from your corpus. We
care less about “calling a model” and more about the full extraction system: schema design, ev-
idence requirements, validation/repair, and how extraction outputs stay correct as the corpus
evolves.
• Ontology/schema: define entity types and relationship/claim types that work for your
chosen corpus. Keep it coherent and extensible.
• Grounding: every extracted claim must point to evidence (source id + excerpt + loca-
tion/offsets + timestamp).
• Validation & repair: how you handle invalid/partial outputs; retries; schema drift; de-
terministic normalization.
• Versioning: how you track extraction versions (prompt/model/schema) and how you
would backfill when the ontology changes.
• Quality gates: how you prevent noisy extractions from becoming durable memory (con-
fidence, cross-evidence support, decay, human review hooks).
Deduplication and Canonicalization
A memory system fails if it stores the same thing 100 different ways. Show how you deduplicate
at multiple levels and keep merges safe and reversible.
• Artifact dedup: identical or near-identical messages (email quoting/forwarding, dupli-
cated tickets, cross-posts).
3
3Layer10 Take-Home
2026
• Entity canonicalization: people, teams, projects, components, customers, etc. (aliases,
renames, collisions).
• Claim dedup: merge repeated statements of the same fact while keeping a set of support-
ing evidence.
• Conflicts & revisions: how you represent “it used to be true” vs “it is true now”; decision
reversals; ownership changes.
• Reversibility: how you would support undoing merges and auditing why a merge hap-
pened.
Memory Graph Design
Build a graph (or graph-like store) that represents long-term memory as claims with evidence.
We are open to Neo4j/Neptune style graphs, Postgres adjacency, document stores, or hybrids.
What matters is that it is queryable, grounded, and maintainable over time.
• Core objects: entities, events/artifacts, claims/relations, evidence pointers, and optional
summaries.
• Time: event time vs validity time; how you decide what is “current”.
• Updates: incremental ingestion; idempotency; reprocessing; handling edits/deletes/redactions
safely.
• Permissions (conceptual): how you would ensure a user only retrieves memory
grounded in sources they can access.
• Observability: what you log/measure to know if extraction and memory quality is de-
grading.
Retrieval and Grounding
Demonstrate how your memory graph supports answering questions with grounded context.
Implement a simple retrieval API or script that, given a question, returns a context pack con-
sisting of ranked evidence snippets and linked entities/claims.
• How you map a question to candidate entities/claims (keyword, embedding, hybrid,
etc.).
• How you expand/aggregate without exploding (pruning, diversity, recency, confidence).
• How you ensure every returned item is grounded in evidence and how you format cita-
tions.
• How you handle ambiguity and conflicting sources (show both, choose a winner, or ask
a follow-up).
4
4Layer10 Take-Home
2026
Visualization Layer
Add a visualization layer that makes the extracted memory explorable. It can be a lightweight
web UI, a notebook visualization, or an off-the-shelf graph viewer with a small adapter. The key
is that someone can navigate entities/claims and click through to the supporting evidence.
• Graph view of entities and relationships/claims (filters by time/type/confidence).
• Evidence panel that shows the exact excerpt(s) supporting a claim with source metadata.
• Ability to inspect duplicates/merges (aliases, merged entities, merged claims).
Layer10 Considerations
In a short section, explain how you would adapt what you built for your chosen corpus
to Layer10’s target environment: email, Slack/Teams, docs, and structured systems like
Jira/Linear.
Focus on what you would change in the ontology, extraction contract, dedup strategy, ground-
ing requirements, and long-term memory behavior.
• Unstructured + structured fusion: how you connect chat/email discussions to tick-
ets/projects/components.
• Long-term memory: what becomes durable memory vs ephemeral context; how you
prevent drift.
• Grounding & safety: provenance, citations, and handling deletions/redactions.
• Permissions: memory retrieval constrained by access to underlying sources.
• Operational reality: scaling, cost, incremental updates, and evaluation/regression test-
ing.
What to Submit
Please provide a repo (or zip) that includes:
• Code for extraction, deduplication/canonicalization, and building the memory
graph/store.
• Outputs: a serialized form of your graph/memory store and a set of example retrieved
context packs for a few questions you choose.
• Visualization: runnable UI/notebook or clear instructions plus screenshots/video show-
ing it working.
• Write-up that explains your ontology, extraction contract, dedup strategy, update seman-
tics, and how you’d adapt to Layer10.
• Reproducibility: clear instructions to run end-to-end from the corpus download to the
visualization.
5
5Layer10 Take-Home
2026
How We Evaluate
• Extraction quality as a system: schema design, evidence requirements, validation/repair,
and versioning.
• Grounding: can every memory item be traced to evidence reliably?
• Deduplication: artifact/entity/claim dedup; merge safety; handling renames and colli-
sions.
• Long-term correctness: revisions, conflicts, “current” vs “historical”, and dele-
tion/redaction handling.
• Usability: visualization and retrieval outputs are understandable and auditable.
• Clarity: the write-up makes tradeoffs explicit and the system is reproducible.