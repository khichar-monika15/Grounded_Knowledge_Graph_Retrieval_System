"""Data-driven ClaimType discovery script.

Samples N emails biased toward longer bodies (more relationship-dense), calls the LLM
with an open-ended discovery prompt (no constrained claim_type list), normalizes the
free-text relationship labels, applies synonym clustering, and prints a frequency table
plus a suggested Python enum snippet to stdout.

Usage:
    uv run python -m pipeline.discover_claim_types --sample-size 30
    uv run python -m pipeline.discover_claim_types --sample-size 30 --output-json outputs/discovery.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Discovery prompt — open-ended, no constrained list
# ---------------------------------------------------------------------------
DISCOVERY_PROMPT = """You are a relationship extractor. For each relationship in this email, identify
the subject, the relationship type, and the object.

Return ONLY valid JSON:
{{
  "relationships": [
    {{
      "subject": "...",
      "relationship_type": "short_snake_case_verb_phrase",
      "object": "...",
      "supporting_excerpt": "exact quote"
    }}
  ]
}}

Rules:
- relationship_type must be snake_case (e.g., sent_to, approved, responsible_for)
- Use precise verbs, not vague ones like "related_to"
- Only extract what the email directly supports

EMAIL:
From: {sender}
To: {recipient}
Date: {date}

{body}
"""

# ---------------------------------------------------------------------------
# Synonym clusters — maps variant → canonical label
# ---------------------------------------------------------------------------
SYNONYMS: dict[str, str] = {
    "sent": "sent_to",
    "emailed": "sent_to",
    "wrote_to": "sent_to",
    "talked_about": "discussed",
    "addressed": "discussed",
    "covered": "discussed",
    "asked_for": "requested",
    "solicited": "requested",
    "approved_of": "approved",
    "signed_off": "approved",
    "denied": "rejected",
    "turned_down": "rejected",
    "declined": "rejected",
    "notified": "informed",
    "told": "informed",
    "updated": "informed",
    "suggested": "proposed",
    "recommended": "proposed",
    "concurred": "agreed",
    "consented": "agreed",
    "permitted": "authorized",
    "allowed": "authorized",
    "granted": "authorized",
    "in_charge_of": "responsible_for",
    "manages": "responsible_for",
}


def _normalize(label: str) -> str:
    """Lowercase, replace non-alphanumeric runs with underscores, strip edges."""
    label = label.lower().strip()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = label.strip("_")
    return label


def _apply_synonyms(label: str) -> str:
    return SYNONYMS.get(label, label)


def _load_sample(sample_size: int) -> list[dict]:
    """Load emails from data/enron_sample.csv, sorted by body length descending."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required. Install with: uv add pandas", file=sys.stderr)
        sys.exit(1)

    csv_path = Path("data/enron_sample.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run download_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["body"])
    df = df.assign(_body_len=df["body"].str.len())
    df = df.sort_values("_body_len", ascending=False).drop(columns=["_body_len"])
    df = df.head(sample_size)

    emails = []
    for _, row in df.iterrows():
        emails.append({
            "sender": row.get("sender", "unknown"),
            "recipient": row.get("recipient", "unknown"),
            "date": str(row.get("date", "unknown")),
            "body": str(row.get("body", "")),
        })
    return emails


def _call_discovery_llm(email: dict) -> list[str]:
    """Call LLM with open-ended discovery prompt, return list of normalized labels."""
    from pipeline.extraction import call_llm, parse_llm_response

    prompt = DISCOVERY_PROMPT.format(
        sender=email.get("sender", "unknown"),
        recipient=email.get("recipient", "unknown"),
        date=email.get("date", "unknown"),
        body=email.get("body", "")[:2000],  # cap body to stay within token budget
    )

    try:
        raw = call_llm(prompt)
    except Exception as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
        return []

    parsed = parse_llm_response(raw)
    if not parsed or "relationships" not in parsed:
        return []

    labels = []
    for rel in parsed.get("relationships", []):
        raw_label = rel.get("relationship_type", "")
        if raw_label:
            normalized = _apply_synonyms(_normalize(raw_label))
            labels.append(normalized)
    return labels


def discover(sample_size: int = 30, output_json: str | None = None) -> dict:
    """Run discovery on sample_size emails and return results dict."""
    print(f"Loading {sample_size} emails (longest bodies first)…")
    emails = _load_sample(sample_size)
    print(f"Loaded {len(emails)} emails.\n")

    all_labels: list[str] = []
    raw_results: list[dict] = []

    for i, email in enumerate(emails):
        print(f"  [{i + 1}/{len(emails)}] Extracting relationships…", end=" ", flush=True)
        labels = _call_discovery_llm(email)
        all_labels.extend(labels)
        raw_results.append({"email_index": i, "labels": labels})
        print(f"found {len(labels)}: {labels}")

    counter = Counter(all_labels)

    # Known ClaimType values (11 original + 6 batch-5 additions) — anything else is new
    known = {
        "works_at", "reports_to", "participated_in", "decided", "requested",
        "mentioned", "discussed", "sent_to", "role_assignment", "status_change",
        "opinion", "approved", "rejected", "informed", "proposed", "agreed", "authorized",
    }

    print("\n" + "=" * 60)
    print(f"DISCOVERY RESULTS — {len(emails)} emails, {len(all_labels)} relationship labels")
    print("=" * 60)
    print(f"{'Label':<30} {'Count':>5}  {'Status'}")
    print("-" * 60)
    for label, count in counter.most_common():
        status = "known" if label in known else "*** NEW ***"
        print(f"  {label:<28} {count:>5}  {status}")

    new_labels = [lbl for lbl, _ in counter.most_common() if lbl not in known]
    if new_labels:
        print("\n--- Suggested enum additions ---")
        for lbl in new_labels:
            const = lbl.upper()
            print(f'    {const} = "{lbl}"')
    else:
        print("\nNo new labels found beyond known types.")

    result = {
        "sample_size": len(emails),
        "total_labels": len(all_labels),
        "frequency": dict(counter.most_common()),
        "new_labels": new_labels,
        "raw": raw_results,
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nRaw results saved to {output_json}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover ClaimType labels from the corpus.")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of emails to sample (default: 30)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to save raw JSON results")
    args = parser.parse_args()
    discover(sample_size=args.sample_size, output_json=args.output_json)


if __name__ == "__main__":
    main()
