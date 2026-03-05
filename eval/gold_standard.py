"""Gold-standard precision/recall evaluation for entity resolution.

10 known Enron entities with expected aliases, and 5 known-false-merge pairs.
Run standalone: uv run python -m eval.gold_standard
Or called from run_pipeline.py after dedup.
"""

GOLD_ENTITIES = {
    "Kenneth Lay": {
        "type": "person",
        "should_merge_with": ["Ken Lay", "ken.lay@enron.com", "Lay"],
    },
    "Jeffrey Skilling": {
        "type": "person",
        "should_merge_with": ["Jeff Skilling", "Skilling", "jeff.skilling@enron.com"],
    },
    "Andrew Fastow": {
        "type": "person",
        "should_merge_with": ["Andy Fastow", "andrew.fastow@enron.com"],
    },
    "Enron": {
        "type": "organization",
        "should_merge_with": ["Enron Corp", "Enron Corporation"],
    },
    "Sally Beck": {
        "type": "person",
        "should_merge_with": ["sally.beck@enron.com"],
    },
    "Vince Kaminski": {
        "type": "person",
        "should_merge_with": ["Vincent Kaminski", "vince.kaminski@enron.com"],
    },
    "John Lavorato": {
        "type": "person",
        "should_merge_with": ["john.lavorato@enron.com"],
    },
    "Mark Haedicke": {
        "type": "person",
        "should_merge_with": ["mark.haedicke@enron.com"],
    },
    "California Energy Crisis": {
        "type": "topic",
        "should_merge_with": ["California energy situation", "California power crisis"],
    },
    "Raptor": {
        "type": "project",
        "should_merge_with": ["Raptor SPV", "Raptor project"],
    },
}

GOLD_SHOULD_NOT_MERGE = [
    ("John Lavorato", "John Arnold"),
    ("John Lavorato", "John Keffer"),
    ("Mark Haedicke", "Greg Whalley"),
    ("Kenneth Lay", "Andrew Fastow"),
    ("Jeffrey Skilling", "Kenneth Lay"),
]


def _normalize_name(name: str) -> str:
    import re
    name = name.lower().strip()
    for prefix in ['mr.', 'mrs.', 'ms.', 'dr.']:
        name = name.replace(prefix, '').strip()
    if ',' in name:
        parts = [p.strip() for p in name.split(',', 1)]
        name = f"{parts[1]} {parts[0]}"
    return name


def evaluate_entity_dedup(merged_entities) -> dict:
    """Compute precision and recall for entity resolution."""
    tp = fp = fn = 0

    for canon, gold in GOLD_ENTITIES.items():
        matched = None
        for e in merged_entities:
            all_names = {_normalize_name(n) for n in [e.canonical_name] + e.aliases}
            if _normalize_name(canon) in all_names:
                matched = e
                break

        if matched is None:
            fn += 1
            continue

        actual_aliases = {_normalize_name(n) for n in [matched.canonical_name] + matched.aliases}
        expected_aliases = (
            {_normalize_name(n) for n in gold["should_merge_with"]}
            | {_normalize_name(canon)}
        )

        for expected in expected_aliases:
            if expected in actual_aliases:
                tp += 1
            else:
                fn += 1

    for name_a, name_b in GOLD_SHOULD_NOT_MERGE:
        for e in merged_entities:
            all_names = {_normalize_name(n) for n in [e.canonical_name] + e.aliases}
            if _normalize_name(name_a) in all_names and _normalize_name(name_b) in all_names:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from config import DB_PATH
    from memory.graph_builder import load_entities_from_sqlite

    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}. Run the pipeline first.")
        sys.exit(1)

    entities = load_entities_from_sqlite(DB_PATH)
    result = evaluate_entity_dedup(entities)
    print(f"Entity Resolution Evaluation (gold standard):")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall:    {result['recall']}")
    print(f"  F1:        {result['f1']}")
    print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}")
