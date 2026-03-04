"""LLM-based structured extraction from emails.

Uses litellm for provider-agnostic LLM calls (TrueFoundry gateway / Ollama / etc.)
and Pydantic (ExtractionResult) for strict output validation.

Prompt strategy: hybrid chain-of-thought — instruct the model to resolve
all entities FIRST, then extract claims that reference only those entities.
This reduces hallucinated entity references without a second API call.
"""
import json
import re
import time
import logging
import os
from typing import Optional

from pydantic import ValidationError

from config import OPENAI_API_KEY, BASE_URL, MODEL, EXTRACTIONS_DIR
from schema import ExtractionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hybrid chain-of-thought prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are an information extraction system. Analyze the email below carefully.

Follow these two steps IN ORDER — this is important for accuracy:

STEP 1 — ENTITY EXTRACTION:
Identify every person, organization, project, topic, location, and role mentioned.
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
      "claim_type": "works_at|reports_to|participated_in|decided|requested|mentioned|discussed|sent_to|role_assignment|status_change|opinion",
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
        stream=False,
        api_base=BASE_URL,
        extra_headers={
            "X-TFY-METADATA": json.dumps({"purpose": "enron-extraction"}),
            "X-TFY-LOGGING-CONFIG": json.dumps({"enabled": True}),
        },
    )
    return response.choices[0].message.content


def parse_llm_response(raw: str) -> Optional[dict]:
    """Parse LLM response, handling markdown fences and loose text."""
    if raw is None:
        return None

    text = raw.strip()
    # Strip ```json ... ``` fences
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Extract first JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
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


def extract_email(email_dict: dict, max_retries: int = 2) -> Optional[dict]:
    """Extract entities and claims from a single email.

    Uses hybrid chain-of-thought prompt (entities first, then claims).
    Validates output with Pydantic and retries once on failure.
    Returns the raw dict (not Pydantic objects) for JSON-serialisable storage.
    """
    prompt = build_prompt(email_dict)

    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt)
            parsed = parse_llm_response(raw)
            if parsed is None:
                logger.warning(f"Failed to parse LLM response on attempt {attempt + 1}")
                continue

            errors = validate_extraction(parsed)
            if errors:
                logger.warning(f"Pydantic validation errors attempt {attempt + 1}: {errors}")
                if attempt < max_retries - 1:
                    continue

            return parsed

        except Exception as e:
            logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


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
