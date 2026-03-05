from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
from datetime import datetime
from enum import Enum


class Evidence(BaseModel):
    """Every claim MUST link to one or more Evidence objects."""
    evidence_id: str
    source_id: str
    source_type: str = "email"
    excerpt: str
    timestamp: Optional[datetime] = None
    subject: Optional[str] = None
    sender: Optional[str] = None
    recipients: Optional[list[str]] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    @field_validator('excerpt')
    @classmethod
    def excerpt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('excerpt cannot be empty')
        return v


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TOPIC = "topic"
    LOCATION = "location"
    ROLE = "role"


class Entity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = []
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    metadata: dict = {}
    merge_history: list[dict] = []


class ClaimType(str, Enum):
    WORKS_AT = "works_at"
    REPORTS_TO = "reports_to"
    PARTICIPATED_IN = "participated_in"
    DECIDED = "decided"
    REQUESTED = "requested"
    MENTIONED = "mentioned"
    DISCUSSED = "discussed"
    SENT_TO = "sent_to"
    ROLE_ASSIGNMENT = "role_assignment"
    STATUS_CHANGE = "status_change"
    OPINION = "opinion"
    # Added via data-driven corpus discovery (Batch 5)
    APPROVED = "approved"
    REJECTED = "rejected"
    INFORMED = "informed"
    PROPOSED = "proposed"
    AGREED = "agreed"
    AUTHORIZED = "authorized"


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    UNCERTAIN = "uncertain"


class Claim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    subject_entity_id: str
    object_entity_id: Optional[str] = None
    object_value: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.ACTIVE
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    evidence_ids: list[str] = []
    superseded_by: Optional[str] = None
    extraction_version: str = "v1"
    merge_history: list[dict] = []


# ---------------------------------------------------------------------------
# LLM extraction output models — Pydantic validation of raw LLM JSON
# ---------------------------------------------------------------------------

class RawEntityExtraction(BaseModel):
    """Pydantic model for a single entity as returned by the LLM."""
    name: str
    type: str
    aliases: list[str] = []

    @field_validator('type')
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        valid = {e.value for e in EntityType}
        if v.lower() not in valid:
            raise ValueError(f"Invalid entity type '{v}'. Must be one of: {sorted(valid)}")
        return v.lower()

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('entity name cannot be empty')
        return v.strip()


class RawClaimExtraction(BaseModel):
    """Pydantic model for a single claim as returned by the LLM."""
    claim_type: str
    subject: str
    object: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    supporting_excerpt: str

    @field_validator('claim_type')
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        valid = {ct.value for ct in ClaimType}
        if v.lower() not in valid:
            raise ValueError(f"Invalid claim type '{v}'. Must be one of: {sorted(valid)}")
        return v.lower()

    @field_validator('subject')
    @classmethod
    def subject_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('claim subject cannot be empty')
        return v.strip()

    @field_validator('supporting_excerpt')
    @classmethod
    def excerpt_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('supporting_excerpt cannot be empty — every claim must be grounded')
        return v.strip()


class ExtractionResult(BaseModel):
    """Top-level Pydantic model for the full LLM extraction response."""
    entities: list[RawEntityExtraction]
    claims: list[RawClaimExtraction]
