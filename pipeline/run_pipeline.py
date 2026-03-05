"""End-to-end pipeline orchestrator.

Improvements over v1:
- Async concurrent LLM extraction via asyncio + asyncio.Semaphore(20).
  Wall-clock time for 200 emails drops from ~3.3 min (sequential) to ~20-30 s.
- FAISS index built and saved after dedup so the Streamlit app loads it at
  startup for <10ms retrieval instead of on-the-fly embedding.
- Chunked CSV loading via download_corpus (already updated).
- executemany bulk inserts in graph_builder (already updated).
"""
import argparse
import asyncio
import json
import os
import uuid
from datetime import datetime

import pandas as pd

from config import DB_PATH, EXTRACTIONS_DIR, CONTEXT_PACKS_DIR, GRAPH_JSON_PATH, OPENAI_API_KEY
from memory.schema import Entity, Claim, Evidence, EntityType, ClaimType, ClaimStatus
from pipeline.extraction import extract_email, validate_extraction, strip_quoted_content
from memory.dedup import hash_email_body, is_quoted_duplicate, deduplicate_entities, deduplicate_claims
from memory.graph_builder import build_graph, save_to_sqlite, save_graph_json
from memory.retrieval import build_context_pack
from memory.vector_store import build_and_save_index

FAISS_INDEX_PATH = "outputs/faiss_index.npz"


def _find_excerpt_offset(body: str, excerpt: str) -> tuple[int | None, int | None]:
    """Find excerpt offset with normalization fallbacks."""
    import re as _re
    # Fast path: exact match
    idx = body.find(excerpt)
    if idx >= 0:
        return idx, idx + len(excerpt)
    # Case-insensitive
    idx = body.lower().find(excerpt.lower())
    if idx >= 0:
        return idx, idx + len(excerpt)
    # Normalized whitespace
    body_norm = _re.sub(r'\s+', ' ', body.lower())
    excerpt_norm = _re.sub(r'\s+', ' ', excerpt.lower())
    idx = body_norm.find(excerpt_norm)
    if idx >= 0:
        return idx, idx + len(excerpt_norm)
    return None, None

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: str, sample_size: int = 200) -> list[dict]:
    """Load sampled email corpus from CSV."""
    df = pd.read_csv(path)
    df = df.head(sample_size)
    emails = []
    for _, row in df.iterrows():
        emails.append({
            "date": str(row.get("date", "")),
            "sender": str(row.get("sender", "")),
            "recipient": str(row.get("recipient", "")),
            "body": str(row.get("body", "")),
            "source_id": str(row.get("source_id", _)),
        })
    return emails


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

def emails_to_schema_objects(extractions: list[dict], emails: list[dict]):
    """Convert raw extraction dicts to schema objects."""
    all_entities: list[Entity] = []
    all_claims: list[Claim] = []
    all_evidence: list[Evidence] = []

    email_by_source = {e["source_id"]: e for e in emails}

    # Bug 15: persist entity name→id mapping across extractions so the same
    # entity is reused (not duplicated) and last_seen advances over time.
    name_to_entity_id: dict[str, str] = {}
    name_to_entity_idx: dict[str, int] = {}  # name.lower() → index in all_entities

    for extraction in extractions:
        source_id = extraction.get("_source_id", str(uuid.uuid4()))
        email_meta = extraction.get("_email_meta", {})
        email = email_by_source.get(source_id, email_meta)

        # Compute email timestamp once — shared by all Evidence/Claim/Entity from this email
        try:
            ts = datetime.fromisoformat(email_meta.get("date") or email.get("date") or "")
        except (ValueError, TypeError):
            ts = None

        # Body text for evidence character offset computation (Bug 7)
        body_text = email.get("body", "")

        # Build evidence per unique excerpt
        excerpt_to_ev_id: dict[str, str] = {}
        for claim_raw in extraction.get("claims", []):
            excerpt = claim_raw.get("supporting_excerpt", "").strip()
            if not excerpt or excerpt in excerpt_to_ev_id:
                continue
            ev_id = str(uuid.uuid4())
            excerpt_to_ev_id[excerpt] = ev_id

            # Compute character offsets with normalization fallbacks (Bug 7 + Bug 6)
            char_start, char_end = _find_excerpt_offset(body_text, excerpt)

            all_evidence.append(Evidence(
                evidence_id=ev_id,
                source_id=source_id,
                source_type="email",
                excerpt=excerpt,
                timestamp=ts,
                sender=email_meta.get("sender") or email.get("sender"),
                recipients=[email_meta.get("recipient") or email.get("recipient", "")]
                    if (email_meta.get("recipient") or email.get("recipient")) else None,
                char_start=char_start,
                char_end=char_end,
            ))

        # Build / update entity name → entity_id mapping (shared across extractions)
        for ent_raw in extraction.get("entities", []):
            name = ent_raw.get("name", "").strip()
            if not name:  # Bug 2 — guard blank entity names
                continue
            etype_str = ent_raw.get("type", "topic").lower()
            try:
                etype = EntityType(etype_str)
            except ValueError:
                etype = EntityType.TOPIC

            key = name.lower()
            if key in name_to_entity_id:
                # Bug 15: entity already exists — update temporal bounds
                idx = name_to_entity_idx[key]
                existing = all_entities[idx]
                if ts:
                    new_first = min(existing.first_seen, ts) if existing.first_seen else ts
                    new_last = max(existing.last_seen, ts) if existing.last_seen else ts
                    # Merge aliases from this extraction into existing entity
                    merged_aliases = list(set(existing.aliases + ent_raw.get("aliases", [])))
                    all_entities[idx] = existing.model_copy(update={
                        "first_seen": new_first,
                        "last_seen": new_last,
                        "aliases": merged_aliases,
                    })
            else:
                entity_id = str(uuid.uuid4())
                all_entities.append(Entity(
                    entity_id=entity_id,
                    canonical_name=name,
                    entity_type=etype,
                    aliases=ent_raw.get("aliases", []),
                    first_seen=ts,  # Bug 3
                    last_seen=ts,   # Bug 3
                ))
                name_to_entity_id[key] = entity_id
                name_to_entity_idx[key] = len(all_entities) - 1

        # Build claims
        for claim_raw in extraction.get("claims", []):
            excerpt = claim_raw.get("supporting_excerpt", "").strip()
            if not excerpt:
                continue

            ev_id = excerpt_to_ev_id.get(excerpt)
            subject_name = claim_raw.get("subject", "").strip()
            object_name = claim_raw.get("object", "").strip()

            if not subject_name:  # Bug 2 — skip claims with blank subject
                continue

            subject_id = name_to_entity_id.get(subject_name.lower())
            if not subject_id:
                subject_id = str(uuid.uuid4())
                all_entities.append(Entity(
                    entity_id=subject_id,
                    canonical_name=subject_name,
                    entity_type=EntityType.PERSON,
                    first_seen=ts,  # Bug 3
                    last_seen=ts,   # Bug 3
                ))
                name_to_entity_id[subject_name.lower()] = subject_id
                name_to_entity_idx[subject_name.lower()] = len(all_entities) - 1

            object_id = name_to_entity_id.get(object_name.lower())
            # Bug 10: auto-create object entity if the LLM mentioned it in claims
            # but omitted it from the entities list — prevents legitimate edges
            # from silently degrading into attribute claims.
            if not object_id and object_name:
                object_id = str(uuid.uuid4())
                all_entities.append(Entity(
                    entity_id=object_id,
                    canonical_name=object_name,
                    entity_type=EntityType.TOPIC,  # conservative default; dedup will refine
                    first_seen=ts,
                    last_seen=ts,
                ))
                name_to_entity_id[object_name.lower()] = object_id
                name_to_entity_idx[object_name.lower()] = len(all_entities) - 1

            claim_type_str = claim_raw.get("claim_type", "mentioned").lower()
            try:
                claim_type = ClaimType(claim_type_str)
            except ValueError:
                claim_type = ClaimType.MENTIONED

            all_claims.append(Claim(
                claim_id=str(uuid.uuid4()),
                claim_type=claim_type,
                subject_entity_id=subject_id,
                object_entity_id=object_id,
                object_value=None if object_id else object_name or None,
                confidence=float(claim_raw.get("confidence", 0.5)),
                evidence_ids=[ev_id] if ev_id else [],
                extraction_version="v1",
                valid_from=ts,  # Bug 1 — inherit email timestamp for temporal superseding
            ))

    return all_entities, all_claims, all_evidence


# ---------------------------------------------------------------------------
# Async extraction
# ---------------------------------------------------------------------------

async def _extract_one_async(email: dict, semaphore: asyncio.Semaphore,
                              output_dir: str, idx: int, total: int) -> dict | None:
    """Extract a single email with semaphore-controlled concurrency."""
    source_id = email["source_id"]
    output_path = os.path.join(output_dir, f"{source_id}.json")

    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)

    async with semaphore:
        loop = asyncio.get_event_loop()
        print(f"  [{idx+1}/{total}] Extracting source_id={source_id}")
        # extract_email is synchronous; run in thread pool to avoid blocking
        result = await loop.run_in_executor(None, extract_email, email)

    if result is not None:
        result["_source_id"] = source_id
        result["_email_meta"] = {
            "sender": email.get("sender"),
            "recipient": email.get("recipient"),
            "date": email.get("date"),
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


async def async_extract_batch(emails: list[dict], output_dir: str,
                               max_concurrent: int = 20) -> tuple[list[dict], int]:
    """Extract all emails concurrently, capped at max_concurrent in-flight requests.

    Returns (results, validation_error_count).
    """
    os.makedirs(output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(emails)

    tasks = [
        _extract_one_async(email, semaphore, output_dir, i, total)
        for i, email in enumerate(emails)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    validation_errors = 0
    for r in raw_results:
        if isinstance(r, Exception):
            print(f"  Extraction error: {r}")
            validation_errors += 1
            continue
        if r is None:
            validation_errors += 1
            continue
        errs = validate_extraction(r)
        if errs:
            validation_errors += len(errs)
        results.append(r)

    return results, validation_errors


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(sample_size: int = 200, api_key: str = None, skip_download: bool = False,
                 max_concurrent: int = 20):
    """Run the full end-to-end pipeline."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/extractions", exist_ok=True)
    os.makedirs("outputs/context_packs", exist_ok=True)

    # Step 1: Load corpus
    corpus_path = "data/enron_sample.csv"
    if not os.path.exists(corpus_path) and not skip_download:
        print("Downloading corpus...")
        from download_corpus import download_corpus
        download_corpus(sample_size, corpus_path)
    elif not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    print(f"Loading corpus (sample_size={sample_size})...")
    emails = load_corpus(corpus_path, sample_size)
    print(f"Loaded {len(emails)} emails")

    # Step 2: Artifact dedup
    print("Running artifact dedup...")
    seen_hashes: set[str] = set()
    seen_bodies: list[str] = []
    unique_emails: list[dict] = []
    duplicate_count = 0

    for email in emails:
        body = email.get("body", "")
        h = hash_email_body(body)
        if h in seen_hashes:
            duplicate_count += 1
            continue
        if is_quoted_duplicate(body, seen_bodies):
            duplicate_count += 1
            continue
        seen_hashes.add(h)
        clean_body, _ = strip_quoted_content(body)
        seen_bodies.append(clean_body)
        email["body"] = clean_body
        unique_emails.append(email)

    print(f"Unique emails: {len(unique_emails)} (skipped {duplicate_count} duplicates)")

    # Step 3: Async concurrent extraction
    print(f"Extracting from {len(unique_emails)} emails (max_concurrent={max_concurrent})...")
    extractions, validation_errors = asyncio.run(
        async_extract_batch(unique_emails, EXTRACTIONS_DIR, max_concurrent=max_concurrent)
    )
    print(f"Extracted {len(extractions)} emails, {validation_errors} validation errors")

    # Step 4: Convert to schema objects
    print("Converting to schema objects...")
    raw_entities, raw_claims, evidence = emails_to_schema_objects(extractions, unique_emails)
    print(f"Raw: {len(raw_entities)} entities, {len(raw_claims)} claims, {len(evidence)} evidence")

    # Step 5: Entity dedup
    print("Deduplicating entities...")
    entities = deduplicate_entities(raw_entities)
    # Bug 13: build reverse lookup in O(E+M) instead of O(E×M×H) nested loops
    entity_id_map: dict[str, str] = {}
    for merged in entities:
        entity_id_map[merged.entity_id] = merged.entity_id
        for mh in merged.merge_history:
            src = mh.get("merged_from", "")
            if src:
                entity_id_map[src] = merged.entity_id

    print(f"After entity dedup: {len(entities)} entities (merged {len(raw_entities) - len(entities)})")

    # Remap claim entity IDs
    updated_claims = [
        c.model_copy(update={
            "subject_entity_id": entity_id_map.get(c.subject_entity_id, c.subject_entity_id),
            "object_entity_id": entity_id_map.get(c.object_entity_id, c.object_entity_id)
                if c.object_entity_id else None,
        })
        for c in raw_claims
    ]

    # Step 6: Claim dedup
    print("Deduplicating claims...")
    claims = deduplicate_claims(updated_claims)
    conflicts = sum(1 for c in claims if c.status == ClaimStatus.SUPERSEDED)
    merge_count = sum(len(e.merge_history) for e in entities)
    print(f"After claim dedup: {len(claims)} claims, {conflicts} conflicts")

    # Step 7: Build graph
    print("Building memory graph...")
    G = build_graph(entities, claims)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 8: Persist to SQLite (uses executemany)
    print(f"Saving to SQLite: {DB_PATH}...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    save_to_sqlite(entities, claims, evidence, DB_PATH)
    save_graph_json(G, GRAPH_JSON_PATH)

    # Step 9: Build FAISS index
    print("Building FAISS vector index...")
    build_and_save_index(entities, claims, FAISS_INDEX_PATH)

    # Step 9.5: Build Kùzu multi-hop graph
    print("Building Kùzu multi-hop graph...")
    from config import KUZU_DB_PATH
    from memory.kuzu_store import KuzuGraphStore
    import memory.retrieval as _ret
    kuzu_store = KuzuGraphStore(KUZU_DB_PATH)
    kuzu_store.load(entities, claims)
    _ret._KUZU_STORE = kuzu_store
    print(f"  Kùzu graph ready at {KUZU_DB_PATH}")

    # Step 10: Generate context packs
    print("Generating context packs...")
    questions = [
        "What role did Sally Beck play at Enron?",
        "What decisions were made about the California energy situation?",
        "What topics did Kenneth Lay discuss?",
        "Who worked at Enron and what were their roles?",
        "What did Vince Kaminski work on?",
    ]

    for q in questions:
        pack = build_context_pack(q, entities, claims, evidence)
        slug = q.lower().replace(" ", "_").replace("?", "")[:50]
        pack_path = f"outputs/context_packs/{slug}.json"
        with open(pack_path, "w") as f:
            json.dump(pack, f, indent=2, default=str)
        print(f"  Saved: {pack_path} ({len(pack['claims'])} claims)")

    # Step 11: Quality metrics
    confidences = [c.confidence for c in claims if c.status == ClaimStatus.ACTIVE]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Quality Metrics")
    print("=" * 60)
    print(f"  Entities:               {len(entities)}")
    print(f"  Claims:                 {len(claims)}")
    print(f"  Evidence:               {len(evidence)}")
    print(f"  Merges:                 {merge_count}")
    print(f"  Avg confidence:         {avg_conf:.3f}")
    print(f"  Conflicts (SUPERSEDED): {conflicts}")
    print(f"  Validation errors:      {validation_errors}")
    print(f"  Duplicate emails:       {duplicate_count}")
    print("=" * 60)

    return {"entities": entities, "claims": claims, "evidence": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Enron memory pipeline")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20,
                        help="Max concurrent LLM calls during async extraction")
    args = parser.parse_args()
    run_pipeline(args.sample_size, args.api_key, args.skip_download, args.max_concurrent)
