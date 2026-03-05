"""LLM-based structured extraction from emails.

Uses litellm for provider-agnostic LLM calls (TrueFoundry gateway / Ollama / etc.)
and Pydantic (ExtractionResult) for strict output validation.

Prompt strategy: hybrid chain-of-thought — instruct the model to resolve
all entities FIRST, then extract claims that reference only those entities.
This reduces hallucinated entity references without a second API call.

Chunking: emails longer than CHUNK_WORD_LIMIT words are split into overlapping
chunks (CHUNK_WORDS words each, CHUNK_OVERLAP_WORDS overlap). Each chunk is
extracted independently; results are merged (union of entities + claims).
The overlap prevents missing entities that straddle chunk boundaries.
"""
import json
import re
import time
import logging
import os
from typing import Optional

# Semantic chunking parameters
CHUNK_WORD_LIMIT = 400    # trigger chunking above this many words
CHUNK_WORDS = 400         # target chunk size in words
CHUNK_OVERLAP_WORDS = 50  # overlap between consecutive chunks

from pydantic import ValidationError

from config import OPENAI_API_KEY, BASE_URL, MODEL, EXTRACTIONS_DIR
from memory.schema import ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hybrid chain-of-thought prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an information extraction system. Analyze the email below carefully.

Follow these two steps IN ORDER — this is important for accuracy:

STEP 1 — ENTITY EXTRACTION:
Identify every person, organization, project, topic, location, and role mentioned.
- "topic": ONLY for high-level recurring themes (e.g. "California Energy Crisis", "Raptor SPV",
  "quarterly earnings"). NOT for one-off actions, dates, numbers, or sentence fragments.
  Maximum 3 topic entities per email.
List them all before moving to claims.

STEP 2 — CLAIM EXTRACTION:
For each relationship or fact, write one claim. Every claim MUST:
- Reference only entities you identified in Step 1
- Include the EXACT text from the email that supports it (copy-paste, do not paraphrase)

Return ONLY valid JSON — no explanation, no markdown fences:
{{
  "entities": [
    {{"name": "...", "type": "person|organization|project|topic|location|role", "aliases": []}}
  ],
  "claims": [
    {{
      "claim_type": "works_at|reports_to|participated_in|decided|requested|mentioned|discussed|sent_to|role_assignment|status_change|opinion|approved|rejected|informed|proposed|agreed|authorized",
      "subject": "entity name from Step 1",
      "object": "entity name from Step 1, or free-text value",
      "confidence": 0.0-1.0,
      "supporting_excerpt": "EXACT quote from the email"
    }}
  ]
}}

Confidence scale:
- 1.0 = explicitly stated ("Andy is the CFO")
- 0.7 = strongly implied ("Andy handles all financial structures")
- 0.4 = weakly implied (peripheral mention)

Rules:
- Only extract what is directly supported by the email text
- Extract sender→recipient as a sent_to claim
- Do NOT invent entities or excerpts

EMAIL:
From: {sender}
To: {recipient}
Date: {date}

{body}
"""


def build_prompt(email_dict: dict) -> str:
    """Build hybrid chain-of-thought extraction prompt from email dict."""
    return PROMPT_TEMPLATE.format(
        sender=email_dict.get("sender", "unknown"),
        recipient=email_dict.get("recipient", email_dict.get("recipients", "unknown")),
        date=email_dict.get("date", "unknown"),
        body=email_dict.get("body", ""),
    )


def strip_quoted_content(body: str) -> tuple[str, str]:
    """Split email body into (clean_content, quoted_content)."""
    lines = body.split("\n")
    clean_lines = []
    quoted_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            quoted_lines.append(stripped.lstrip(">").strip())
        else:
            clean_lines.append(line)
    clean = "\n".join(clean_lines).strip()
    quoted = "\n".join(quoted_lines).strip()
    return clean, quoted


def call_llm(prompt: str) -> str:
    """Call LLM via litellm (provider-agnostic).

    Currently configured for TrueFoundry → Claude Haiku.
    To switch to Ollama: set MODEL="ollama/llama3" and BASE_URL="http://localhost:11434"
    and remove api_key / extra_headers.
    """
    from litellm import completion

    response = completion(
        api_key=OPENAI_API_KEY,
        custom_llm_provider="openai",
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an information extraction system. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=8192,
        stream=False,
        api_base=BASE_URL,
        extra_headers={
            "X-TFY-METADATA": json.dumps({"purpose": "enron-extraction"}),
            "X-TFY-LOGGING-CONFIG": json.dumps({"enabled": True}),
        },
    )

    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if finish_reason == "length":
        logger.warning("LLM response truncated (finish_reason='length') — JSON likely incomplete")

    return response.choices[0].message.content


def parse_llm_response(raw: str) -> Optional[dict]:
    """Parse LLM response, handling markdown fences and loose text.

    Three-pass strategy:
    1. Strip ```json ... ``` fences if present.
    2. Try the full remaining text as JSON (handles clean responses).
    3. Walk forward from the first '{' and find the matching '}' using a
       bracket counter — this is immune to the greedy-regex problem where
       post-JSON commentary containing braces causes json.loads to fail.
    """
    if raw is None:
        return None

    text = raw.strip()

    # Pass 1: strip markdown fences
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Pass 2: try the whole text directly (fast path for clean responses)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Pass 3: bracket-balanced extraction — find first '{' and walk to its
    # matching '}', ignoring any trailing commentary
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def validate_extraction(data: dict) -> list[str]:
    """Validate extraction output using Pydantic ExtractionResult schema.

    Returns a list of human-readable error strings (empty = valid).
    Replaces the earlier manual dict-inspection approach with strict
    Pydantic validation that enforces entity types, claim types, and
    the grounding requirement (non-empty supporting_excerpt).
    """
    try:
        ExtractionResult.model_validate(data)
        return []
    except ValidationError as e:
        return [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()]
    except Exception as e:
        return [str(e)]


GARBAGE_PERSON_NAMES = frozenset({
    "counterparty", "recipient", "sender", "employee", "manager",
    "board member", "staff", "team", "person", "individual", "client",
    "customer", "user", "analyst", "trader", "attorney", "counsel",
})

MAX_TOPICS_PER_EMAIL = 5
MIN_TOPIC_WORDS = 2


def filter_garbage_entities(entities: list[dict]) -> list[dict]:
    """Remove low-quality entities that pollute the graph."""
    filtered = []
    topic_count = 0

    for e in entities:
        etype = e.get("type", "").lower()
        name = e.get("name", "").strip()

        if etype == "person":
            if "@" in name and " " not in name:
                continue
            if len(name.split()) < 2 and "@" not in name:
                continue
            if name.lower() in GARBAGE_PERSON_NAMES:
                continue
            if len(name) <= 4 and not all(c.isalpha() for c in name):
                continue

        elif etype == "topic":
            if re.match(r'^[\d/\-\.\s,]+$', name):
                continue
            if len(name.split()) < MIN_TOPIC_WORDS:
                continue
            if topic_count >= MAX_TOPICS_PER_EMAIL:
                continue
            topic_count += 1

        filtered.append(e)

    return filtered


def recalibrate_confidence(claim: dict, email_body: str) -> float:
    """Compute confidence from structural signals instead of trusting LLM."""
    excerpt = claim.get("supporting_excerpt", "")
    score = 0.5  # baseline

    if excerpt and excerpt.lower() in email_body.lower():
        score += 0.3

    explicit_types = {"decided", "approved", "rejected", "authorized", "role_assignment"}
    if claim.get("claim_type") in explicit_types:
        score += 0.1

    if len(excerpt.split()) > 10:
        score += 0.1

    weak_types = {"mentioned", "discussed", "opinion"}
    if claim.get("claim_type") in weak_types:
        score -= 0.2

    return round(min(max(score, 0.1), 1.0), 2)


def chunk_body(body: str, chunk_words: int = CHUNK_WORDS,
               overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split a long email body into overlapping word-level chunks.

    Args:
        body: Email body text.
        chunk_words: Maximum words per chunk.
        overlap_words: Words to repeat at the start of the next chunk so that
                       entities straddling a boundary are not lost.

    Returns:
        List of chunk strings. If body is short enough, returns [body].
    """
    words = body.split()
    if len(words) <= chunk_words:
        return [body]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words  # next chunk starts with overlap
    return chunks


def _merge_extractions(results: list[dict]) -> dict:
    """Merge extraction results from multiple chunks.

    Entities and claims are unioned. Duplicate entity names (same name + type)
    are not filtered here — that's handled downstream by dedup.py.
    """
    merged_entities: list[dict] = []
    merged_claims: list[dict] = []
    seen_entity_keys: set[tuple] = set()
    seen_excerpt_keys: set[str] = set()

    for r in results:
        for e in r.get("entities", []):
            key = (e.get("name", "").lower(), e.get("type", ""))
            if key not in seen_entity_keys:
                merged_entities.append(e)
                seen_entity_keys.add(key)
        for c in r.get("claims", []):
            excerpt = c.get("supporting_excerpt", "").strip()
            if excerpt and excerpt not in seen_excerpt_keys:
                merged_claims.append(c)
                seen_excerpt_keys.add(excerpt)

    return {"entities": merged_entities, "claims": merged_claims}


def _extract_single(email_dict: dict, max_retries: int = 2) -> Optional[dict]:
    """Extract from one prompt (single email body or one chunk)."""
    prompt = build_prompt(email_dict)

    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt)
            parsed = parse_llm_response(raw)
            if parsed is None:
                preview = (raw or "")[:300].replace('\n', ' ')
                logger.warning(
                    f"Failed to parse LLM response on attempt {attempt + 1}. "
                    f"Raw preview: {preview!r}"
                )
                continue

            errors = validate_extraction(parsed)
            if errors:
                logger.warning(f"Pydantic validation errors attempt {attempt + 1}: {errors}")
                if attempt < max_retries - 1:
                    continue

            # Post-extraction quality filters
            parsed["entities"] = filter_garbage_entities(parsed.get("entities", []))
            body = email_dict.get("body", "")
            for claim in parsed.get("claims", []):
                claim["confidence"] = recalibrate_confidence(claim, body)

            return parsed

        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


def extract_email(email_dict: dict, max_retries: int = 2) -> Optional[dict]:
    """Extract entities and claims from a single email.

    For emails longer than CHUNK_WORD_LIMIT words, the body is split into
    overlapping chunks. Each chunk is extracted independently and the results
    are merged. For short emails, a single LLM call is made.

    Returns the raw dict (not Pydantic objects) for JSON-serialisable storage.
    """
    body = email_dict.get("body", "")
    chunks = chunk_body(body)

    if len(chunks) == 1:
        return _extract_single(email_dict, max_retries)

    # Multi-chunk extraction
    logger.info(f"Email body split into {len(chunks)} chunks ({len(body.split())} words total)")
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_dict = dict(email_dict)
        chunk_dict["body"] = chunk
        result = _extract_single(chunk_dict, max_retries)
        if result is not None:
            chunk_results.append(result)
        if i < len(chunks) - 1:
            time.sleep(0.5)  # light rate limiting between chunk calls

    if not chunk_results:
        return None

    return _merge_extractions(chunk_results)


def extract_batch(emails: list[dict], output_dir: str = EXTRACTIONS_DIR,
                  rate_limit_sleep: float = 1.0) -> list[dict]:
    """Extract from a batch of emails with rate limiting and per-email caching."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, email in enumerate(emails):
        source_id = str(email.get("source_id", i))
        output_path = os.path.join(output_dir, f"{source_id}.json")

        if os.path.exists(output_path):
            with open(output_path) as f:
                results.append(json.load(f))
            continue

        print(f"Extracting email {i + 1}/{len(emails)} (source_id={source_id})")
        result = extract_email(email)

        if result is not None:
            result["_source_id"] = source_id
            result["_email_meta"] = {
                "sender": email.get("sender"),
                "recipient": email.get("recipient"),
                "date": str(email.get("date", "")),
            }
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            results.append(result)
        else:
            logger.warning(f"Failed to extract email {source_id}")

        if i < len(emails) - 1:
            time.sleep(rate_limit_sleep)

    return results
