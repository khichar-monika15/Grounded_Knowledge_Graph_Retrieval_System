"""LLM-based structured extraction from emails."""
import json
import re
import time
import logging
import os
from typing import Optional

from config import OPENAI_API_KEY, BASE_URL, MODEL, EXTRACTIONS_DIR

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an information extraction system. Given an email, extract structured knowledge.

Return ONLY valid JSON with this structure:
{{
  "entities": [
    {{"name": "...", "type": "person|organization|project|topic|location|role", "aliases": []}}
  ],
  "claims": [
    {{
      "claim_type": "works_at|reports_to|participated_in|decided|requested|mentioned|discussed|role_assignment|status_change|opinion|sent_to",
      "subject": "entity name (who/what)",
      "object": "entity name or value (target)",
      "confidence": 0.0-1.0,
      "supporting_excerpt": "EXACT quote from the email that supports this claim"
    }}
  ]
}}

Rules:
- Only extract claims that are directly supported by the email text
- Include the exact excerpt that supports each claim
- confidence: 1.0 = explicitly stated, 0.7 = strongly implied, 0.4 = weakly implied
- Do not invent information not in the email
- Extract sender/recipient relationships as claims too

EMAIL:
From: {sender}
To: {recipient}
Date: {date}

{body}
"""


def build_prompt(email_dict: dict) -> str:
    """Build extraction prompt from email dict."""
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
    """Call LLM via TrueFoundry OpenAI-compatible endpoint."""
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an information extraction system. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=False,
        extra_headers={
            "X-TFY-METADATA": json.dumps({"purpose": "enron-extraction"}),
            "X-TFY-LOGGING-CONFIG": json.dumps({"enabled": True}),
        },
    )
    return response.choices[0].message.content


def parse_llm_response(raw: str) -> Optional[dict]:
    """Parse LLM response, handling markdown fences and malformed JSON."""
    if raw is None:
        return None

    # Strip markdown fences
    text = raw.strip()
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def validate_extraction(data: dict) -> list[str]:
    """Validate extraction output structure. Returns list of error messages."""
    errors = []

    if "entities" not in data:
        errors.append("Missing 'entities' key")
    if "claims" not in data:
        errors.append("Missing 'claims' key")
        return errors

    for i, entity in enumerate(data.get("entities", [])):
        if "type" not in entity:
            errors.append(f"Entity {i} missing 'type'")
        if "name" not in entity:
            errors.append(f"Entity {i} missing 'name'")

    for i, claim in enumerate(data.get("claims", [])):
        excerpt = claim.get("supporting_excerpt", "")
        if not excerpt or not str(excerpt).strip():
            errors.append(f"Claim {i} missing or empty 'supporting_excerpt'")
        if "claim_type" not in claim:
            errors.append(f"Claim {i} missing 'claim_type'")

    return errors


def extract_email(email_dict: dict, max_retries: int = 2) -> Optional[dict]:
    """Extract entities and claims from a single email. Retries once on failure."""
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
                logger.warning(f"Validation errors on attempt {attempt + 1}: {errors}")
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
    """Extract from a batch of emails with rate limiting."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, email in enumerate(emails):
        source_id = str(email.get("source_id", i))
        output_path = os.path.join(output_dir, f"{source_id}.json")

        # Skip if already extracted
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
