from typing import TypedDict, List
from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    risk_level: str = Field(description="HIGH, MEDIUM, or LOW")
    overlap_score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="short explanation")


class ReportAssessment(BaseModel):
    patent_id: str = Field(description="patent id")
    risk_level: str = Field(description="HIGH, MEDIUM, or LOW")
    overlap_score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="reasoning text")


class FTOReport(BaseModel):
    report: str = Field(description="markdown report")
    assessments: List[ReportAssessment] = Field(description="risk list")


class FTOState(TypedDict):
    user_idea: str
    sub_queries: List[str]
    raw_patents: List[dict]
    decomposed_claims: List[dict]
    risk_assessments: List[dict]
    cleared_patents: List[dict]
    retrieval_quality_score: float
    retry_count: int
    human_approved: bool
    final_report: str
