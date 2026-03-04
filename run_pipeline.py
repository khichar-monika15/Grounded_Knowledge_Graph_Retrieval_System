"""End-to-end pipeline orchestrator."""
import argparse
import json
import os
import uuid
from datetime import datetime

import pandas as pd

from config import DB_PATH, EXTRACTIONS_DIR, CONTEXT_PACKS_DIR, GRAPH_JSON_PATH, OPENAI_API_KEY
from schema import Entity, Claim, Evidence, EntityType, ClaimType, ClaimStatus
from extraction import extract_batch, strip_quoted_content
from dedup import hash_email_body, is_quoted_duplicate, deduplicate_entities, deduplicate_claims
from graph_builder import build_graph, save_to_sqlite, save_graph_json
from retrieval import build_context_pack


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


def emails_to_schema_objects(extractions: list[dict], emails: list[dict]):
    """Convert raw extraction dicts to schema objects."""
    all_entities = []
    all_claims = []
    all_evidence = []

    email_by_source = {e["source_id"]: e for e in emails}

    for extraction in extractions:
        source_id = extraction.get("_source_id", str(uuid.uuid4()))
        email_meta = extraction.get("_email_meta", {})
        email = email_by_source.get(source_id, email_meta)

        # Build evidence for each claim excerpt
        excerpt_to_ev_id = {}
        for claim_raw in extraction.get("claims", []):
            excerpt = claim_raw.get("supporting_excerpt", "").strip()
            if not excerpt:
                continue
            if excerpt in excerpt_to_ev_id:
                continue
            ev_id = str(uuid.uuid4())
            excerpt_to_ev_id[excerpt] = ev_id

            try:
                ts = datetime.fromisoformat(email_meta.get("date") or email.get("date") or "")
            except (ValueError, TypeError):
                ts = None

            evidence = Evidence(
                evidence_id=ev_id,
                source_id=source_id,
                source_type="email",
                excerpt=excerpt,
                timestamp=ts,
                sender=email_meta.get("sender") or email.get("sender"),
                recipients=[email_meta.get("recipient") or email.get("recipient", "")]
                    if (email_meta.get("recipient") or email.get("recipient")) else None,
            )
            all_evidence.append(evidence)

        # Build entity name → entity_id mapping
        name_to_entity_id = {}
        for ent_raw in extraction.get("entities", []):
            name = ent_raw.get("name", "").strip()
            etype_str = ent_raw.get("type", "topic").lower()
            try:
                etype = EntityType(etype_str)
            except ValueError:
                etype = EntityType.TOPIC

            entity_id = str(uuid.uuid4())
            entity = Entity(
                entity_id=entity_id,
                canonical_name=name,
                entity_type=etype,
                aliases=ent_raw.get("aliases", []),
            )
            name_to_entity_id[name.lower()] = entity_id
            all_entities.append(entity)

        # Build claims
        for claim_raw in extraction.get("claims", []):
            excerpt = claim_raw.get("supporting_excerpt", "").strip()
            if not excerpt:
                continue

            ev_id = excerpt_to_ev_id.get(excerpt)
            subject_name = claim_raw.get("subject", "").strip()
            object_name = claim_raw.get("object", "").strip()

            subject_id = name_to_entity_id.get(subject_name.lower())
            if not subject_id:
                # Create entity on the fly
                subject_id = str(uuid.uuid4())
                entity = Entity(
                    entity_id=subject_id,
                    canonical_name=subject_name,
                    entity_type=EntityType.PERSON,
                )
                name_to_entity_id[subject_name.lower()] = subject_id
                all_entities.append(entity)

            object_id = name_to_entity_id.get(object_name.lower())

            claim_type_str = claim_raw.get("claim_type", "mentioned").lower()
            try:
                claim_type = ClaimType(claim_type_str)
            except ValueError:
                claim_type = ClaimType.MENTIONED

            claim = Claim(
                claim_id=str(uuid.uuid4()),
                claim_type=claim_type,
                subject_entity_id=subject_id,
                object_entity_id=object_id,
                object_value=None if object_id else object_name or None,
                confidence=float(claim_raw.get("confidence", 0.5)),
                evidence_ids=[ev_id] if ev_id else [],
                extraction_version="v1",
            )
            all_claims.append(claim)

    return all_entities, all_claims, all_evidence


def run_pipeline(sample_size: int = 200, api_key: str = None, skip_download: bool = False):
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
    seen_hashes = set()
    seen_bodies = []
    unique_emails = []
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

    # Step 3: Extraction
    print(f"Extracting from {len(unique_emails)} emails...")
    validation_errors = 0
    extractions = []

    for i, email in enumerate(unique_emails):
        source_id = email["source_id"]
        output_path = f"outputs/extractions/{source_id}.json"

        if os.path.exists(output_path):
            with open(output_path) as f:
                extractions.append(json.load(f))
            continue

        from extraction import extract_email, validate_extraction
        import time
        print(f"  [{i+1}/{len(unique_emails)}] Extracting source_id={source_id}")
        result = extract_email(email)

        if result is not None:
            errs = validate_extraction(result)
            if errs:
                validation_errors += len(errs)
            result["_source_id"] = source_id
            result["_email_meta"] = {
                "sender": email.get("sender"),
                "recipient": email.get("recipient"),
                "date": email.get("date"),
            }
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            extractions.append(result)
        else:
            validation_errors += 1

        if i < len(unique_emails) - 1:
            time.sleep(1.0)

    print(f"Extracted {len(extractions)} emails, {validation_errors} validation errors")

    # Step 4: Convert to schema objects
    print("Converting to schema objects...")
    raw_entities, raw_claims, evidence = emails_to_schema_objects(extractions, unique_emails)
    print(f"Raw: {len(raw_entities)} entities, {len(raw_claims)} claims, {len(evidence)} evidence")

    # Step 5: Dedup
    print("Deduplicating entities...")
    entities = deduplicate_entities(raw_entities)
    entity_id_map = {}
    for orig in raw_entities:
        for merged in entities:
            if orig.entity_id == merged.entity_id or orig.entity_id in [mh.get("merged_from") for mh in merged.merge_history]:
                entity_id_map[orig.entity_id] = merged.entity_id

    print(f"After entity dedup: {len(entities)} entities (merged {len(raw_entities) - len(entities)})")

    # Update claim entity IDs
    updated_claims = []
    for c in raw_claims:
        new_subject = entity_id_map.get(c.subject_entity_id, c.subject_entity_id)
        new_object = entity_id_map.get(c.object_entity_id, c.object_entity_id) if c.object_entity_id else None
        updated_claims.append(c.model_copy(update={
            "subject_entity_id": new_subject,
            "object_entity_id": new_object,
        }))

    print("Deduplicating claims...")
    claims = deduplicate_claims(updated_claims)
    conflicts = sum(1 for c in claims if c.status == ClaimStatus.SUPERSEDED)
    merge_count = sum(len(e.merge_history) for e in entities)
    print(f"After claim dedup: {len(claims)} claims, {conflicts} conflicts")

    # Step 6: Build graph
    print("Building memory graph...")
    G = build_graph(entities, claims)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Step 7: Persist
    print(f"Saving to SQLite: {DB_PATH}...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    save_to_sqlite(entities, claims, evidence, DB_PATH)
    save_graph_json(G, GRAPH_JSON_PATH)

    # Step 8: Generate context packs
    print("Generating context packs...")
    questions = [
        "Who did Jeff Skilling report to?",
        "What decisions were made about the California energy situation?",
        "Who was involved in the Raptor project?",
        "What role did Andy Fastow play at Enron?",
        "What topics were discussed between Ken Lay and Jeff Skilling?",
    ]

    for q in questions:
        pack = build_context_pack(q, entities, claims, evidence)
        slug = q.lower().replace(" ", "_").replace("?", "")[:50]
        pack_path = f"outputs/context_packs/{slug}.json"
        with open(pack_path, "w") as f:
            json.dump(pack, f, indent=2, default=str)
        print(f"  Saved: {pack_path} ({len(pack['claims'])} claims)")

    # Step 9: Quality metrics
    confidences = [c.confidence for c in claims if c.status == ClaimStatus.ACTIVE]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print("\n" + "="*60)
    print("PIPELINE COMPLETE — Quality Metrics")
    print("="*60)
    print(f"  Entities:              {len(entities)}")
    print(f"  Claims:                {len(claims)}")
    print(f"  Evidence:              {len(evidence)}")
    print(f"  Merges:                {merge_count}")
    print(f"  Avg confidence:        {avg_conf:.3f}")
    print(f"  Conflicts (SUPERSEDED): {conflicts}")
    print(f"  Validation errors:     {validation_errors}")
    print(f"  Duplicate emails:      {duplicate_count}")
    print("="*60)

    return {
        "entities": entities,
        "claims": claims,
        "evidence": evidence,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Enron memory pipeline")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.sample_size, args.api_key, args.skip_download)
