from pydantic import BaseModel, Field, field_validator
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
    confidence: float = 0.5
    status: ClaimStatus = ClaimStatus.ACTIVE
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    evidence_ids: list[str] = []
    superseded_by: Optional[str] = None
    extraction_version: str = "v1"
    merge_history: list[dict] = []
