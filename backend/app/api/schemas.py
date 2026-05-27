"""VideoUnit API Pydantic schemas."""

from pydantic import BaseModel, Field
from typing import Optional


class ContractGenerateRequest(BaseModel):
    """Request to generate a test contract from a text prompt."""
    prompt: str = Field(..., description="Text description of expected video behavior")
    provider: str = Field(default="minimax", description="AI provider: minimax or gemma4")


class ContractGenerateResponse(BaseModel):
    """Response containing generated contract YAML and metadata."""
    contract_yaml: str = Field(..., description="YAML contract string")
    objects: list[str] = Field(..., description="Extracted object names")
    assertions: int = Field(..., description="Number of assertions in contract")


class EvaluateRequest(BaseModel):
    """Request to evaluate a video against a contract."""
    contract_yaml: str = Field(..., description="YAML contract string")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")


class EvaluateResponse(BaseModel):
    """Response from starting a video evaluation."""
    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Initial status: running")


class RunStatusResponse(BaseModel):
    """Status of a VideoUnit evaluation run."""
    run_id: str
    status: str = Field(..., description="running | completed | failed")
    progress: float = Field(default=0.0, description="Progress 0.0-1.0")
    result: Optional[dict] = Field(None, description="Final result if completed")
    failures: Optional[list[dict]] = Field(None, description="Failure list if completed")
    overall_score: Optional[float] = Field(None, description="Overall score 0-100")
    categories: Optional[dict[str, float]] = Field(None, description="Per-category scores")


class ReportResponse(BaseModel):
    """Report generation response."""
    run_id: str
    format: str
    report_path: str
    summary: dict
