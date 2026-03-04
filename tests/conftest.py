import pytest
from datetime import datetime


@pytest.fixture
def sample_email_raw():
    """A single raw email dict as parsed from CSV."""
    return {
        "date": "2001-05-14",
        "sender": "jeff.skilling@enron.com",
        "recipient": "ken.lay@enron.com",
        "body": "Ken, we need to discuss the California situation. The board meeting is set for Friday. Please have Andy prepare the Raptor documents. Jeff"
    }


@pytest.fixture
def sample_email_short():
    """A trivially short email for edge case testing."""
    return {
        "date": "2001-06-01",
        "sender": "jeff.skilling@enron.com",
        "recipient": "ken.lay@enron.com",
        "body": "ok"
    }


@pytest.fixture
def sample_email_quoted():
    """An email with forwarded/quoted content."""
    return {
        "date": "2001-05-15",
        "sender": "ken.lay@enron.com",
        "recipient": "andy.fastow@enron.com",
        "body": "Andy please see below.\n> Ken, we need to discuss the California situation.\n> The board meeting is set for Friday.\n> Please have Andy prepare the Raptor documents.\n> Jeff"
    }


@pytest.fixture
def sample_extraction_response():
    """What a correct LLM extraction response looks like."""
    return {
        "entities": [
            {"name": "Jeff Skilling", "type": "person", "aliases": ["Skilling", "Jeff"]},
            {"name": "Ken Lay", "type": "person", "aliases": ["Kenneth Lay", "Ken"]},
            {"name": "Andy Fastow", "type": "person", "aliases": ["Andrew Fastow"]},
            {"name": "Raptor", "type": "project", "aliases": []},
            {"name": "Enron Board", "type": "organization", "aliases": ["the board"]}
        ],
        "claims": [
            {
                "claim_type": "discussed",
                "subject": "Jeff Skilling",
                "object": "California situation",
                "confidence": 0.9,
                "supporting_excerpt": "we need to discuss the California situation"
            },
            {
                "claim_type": "requested",
                "subject": "Jeff Skilling",
                "object": "Andy Fastow",
                "confidence": 0.85,
                "supporting_excerpt": "Please have Andy prepare the Raptor documents"
            },
            {
                "claim_type": "sent_to",
                "subject": "Jeff Skilling",
                "object": "Ken Lay",
                "confidence": 1.0,
                "supporting_excerpt": "Ken, we need to discuss"
            }
        ]
    }


@pytest.fixture
def sample_malformed_llm_response():
    """Malformed JSON the LLM might return."""
    return '{"entities": [{"name": "Jeff Skilling", "type": "person"}], "claims": [BROKEN JSON'


@pytest.fixture
def sample_valid_llm_response(sample_extraction_response):
    """Valid JSON string from LLM."""
    import json
    return json.dumps(sample_extraction_response)


@pytest.fixture
def duplicate_entities():
    """Entities that should be merged during dedup."""
    from schema import Entity, EntityType
    return [
        Entity(entity_id="e1", canonical_name="Jeff Skilling", entity_type=EntityType.PERSON, aliases=["Skilling"]),
        Entity(entity_id="e2", canonical_name="Skilling, Jeff", entity_type=EntityType.PERSON, aliases=["J. Skilling"]),
        Entity(entity_id="e3", canonical_name="Jeffrey Skilling", entity_type=EntityType.PERSON, aliases=["jeff.skilling@enron.com"]),
        Entity(entity_id="e4", canonical_name="Ken Lay", entity_type=EntityType.PERSON, aliases=["Kenneth Lay"]),
    ]


@pytest.fixture
def duplicate_claims():
    """Claims that should be deduplicated."""
    from schema import Claim, ClaimType, ClaimStatus
    return [
        Claim(claim_id="c1", claim_type=ClaimType.WORKS_AT, subject_entity_id="e1",
              object_entity_id="org1", object_value=None, confidence=0.9,
              evidence_ids=["ev1"]),
        Claim(claim_id="c2", claim_type=ClaimType.WORKS_AT, subject_entity_id="e1",
              object_entity_id="org1", object_value=None, confidence=0.8,
              evidence_ids=["ev2"]),
    ]


@pytest.fixture
def conflicting_claims():
    """Claims that conflict — same subject+predicate, different object."""
    from schema import Claim, ClaimType, ClaimStatus
    from datetime import datetime
    return [
        Claim(claim_id="c1", claim_type=ClaimType.REPORTS_TO, subject_entity_id="e1",
              object_entity_id="e4", confidence=0.9, evidence_ids=["ev1"],
              valid_from=datetime(2001, 1, 1)),
        Claim(claim_id="c2", claim_type=ClaimType.REPORTS_TO, subject_entity_id="e1",
              object_entity_id="e5", confidence=0.95, evidence_ids=["ev2"],
              valid_from=datetime(2001, 6, 1)),
    ]


@pytest.fixture
def sample_graph_data():
    """Pre-built entities, claims, and evidence for graph/retrieval tests."""
    from schema import Entity, Claim, Evidence, EntityType, ClaimType
    from datetime import datetime
    entities = [
        Entity(entity_id="e1", canonical_name="Jeff Skilling", entity_type=EntityType.PERSON, aliases=["Skilling"]),
        Entity(entity_id="e2", canonical_name="Ken Lay", entity_type=EntityType.PERSON, aliases=["Kenneth Lay"]),
        Entity(entity_id="e3", canonical_name="Enron", entity_type=EntityType.ORGANIZATION, aliases=["Enron Corp"]),
        Entity(entity_id="e4", canonical_name="Raptor", entity_type=EntityType.PROJECT, aliases=[]),
    ]
    evidence = [
        Evidence(evidence_id="ev1", source_id="email_001", excerpt="we need to discuss the California situation",
                 timestamp=datetime(2001, 5, 14), sender="jeff.skilling@enron.com"),
        Evidence(evidence_id="ev2", source_id="email_002", excerpt="Please have Andy prepare the Raptor documents",
                 timestamp=datetime(2001, 5, 14), sender="jeff.skilling@enron.com"),
    ]
    claims = [
        Claim(claim_id="c1", claim_type=ClaimType.WORKS_AT, subject_entity_id="e1",
              object_entity_id="e3", confidence=0.95, evidence_ids=["ev1"]),
        Claim(claim_id="c2", claim_type=ClaimType.DISCUSSED, subject_entity_id="e1",
              object_value="California situation", confidence=0.9, evidence_ids=["ev1"]),
        Claim(claim_id="c3", claim_type=ClaimType.PARTICIPATED_IN, subject_entity_id="e1",
              object_entity_id="e4", confidence=0.85, evidence_ids=["ev2"]),
    ]
    return {"entities": entities, "claims": claims, "evidence": evidence}
