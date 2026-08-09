"""
MediLex India — Request/Response schemas.

Two schemas coexist intentionally:

  CaseInput        — The 4-field "simple" schema that the /api/analyze
                     endpoint actually uses today.  Matches what the
                     frontend sends and is easy to test with curl.

  MediLexIntake    — The legally rigorous 30+ field schema originally
                     designed in mediLex.py.  Captures
                     perpetrator relationship, gestational age, consent
                     capacity, guardian checks, and other fields that
                     are *legally significant* for Indian medico-legal
                     protocols.  The frontend already collects these
                     fields; they are preserved here for the Phase-2
                     schema upgrade.

Why keep both?
  - CaseInput is the working API contract today — changing it would
    break the frontend and test scripts simultaneously.
  - MediLexIntake is too valuable to discard; its model_validators
    encode real Indian-law rules (POCSO minor checks, MTP gestational
    limits, accused-as-guardian escalation).  Merging it in is a
    future step, not a cleanup step.

Tech stack:  Pydantic v2 — chosen because FastAPI requires it for
request validation, and the model_validator / field_validator
features let us encode domain-specific legal rules declaratively.
"""

from pydantic import BaseModel, Field
import config


class CaseInput(BaseModel):
    """
    Simple intake schema — what /api/analyze actually reads.

    The frontend sends additional nested fields (sexual_offense,
    pregnancy, consent_and_guardianship, etc.) but FastAPI/Pydantic
    silently ignores them because they are not declared here.
    This is intentional for now — keeps the API surface small and
    testable while the richer schema is integrated incrementally.
    """
    patient_age: int = Field(
        ..., ge=config.MIN_PATIENT_AGE, le=config.MAX_PATIENT_AGE,
        description="Patient age in years."
    )
    gender: str = Field(
        ..., description="Patient gender (male / female / intersex)."
    )
    case_type: str = Field(
        ..., description="Case category, e.g. 'pocso', 'acid_burn', 'road_accident'."
    )
    symptoms: str = Field(
        ..., max_length=config.MAX_SYMPTOMS_LENGTH,
        description="Free-text clinical presentation from the doctor."
    )


class AnalyzeResponse(BaseModel):
    """
    Shape of the /api/analyze JSON response.

    Note: main.py's route doesn't currently declare response_model=AnalyzeResponse,
    so this isn't runtime-enforced today — it documents the contract for callers
    (and for whoever wires response_model in later) rather than validating output.
    """
    session_id: int
    is_minor: bool
    laws_retrieved: list[str]
    checklist: dict | None
    abstained: bool = False
    abstain_reason: str | None = None
    confidence_score: float | None = None
