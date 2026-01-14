"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel
from typing import List, Optional


# ============ Request Models ============

class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint"""
    text: str


class CheckpointLoadRequest(BaseModel):
    """Request body for /checkpoints/load endpoint"""
    checkpoint_name: str


# ============ Response Models ============

class ParagraphInfo(BaseModel):
    """Information about a single paragraph"""
    index: int
    text: str
    ai_prob: float
    importance: float


class TopParagraph(BaseModel):
    """Information about the most suspicious paragraph"""
    index: int
    text: str
    ai_prob: float


class LimeToken(BaseModel):
    """LIME token with importance score"""
    word: str
    score: float  # Positive = AI indicator, Negative = Human indicator


class LimeResult(BaseModel):
    """LIME analysis result"""
    tokens: List[LimeToken]


class DeletionTest(BaseModel):
    """Deletion test result for reliability verification"""
    original_prob: float
    modified_prob: float
    drop: float
    reliability: str  # "high", "medium", "low"
    removed_tokens: List[str]


class AnalyzeResponse(BaseModel):
    """Full analysis response"""
    prediction: str  # "AI" or "Human"
    confidence: float
    paragraphs: List[ParagraphInfo]
    top_paragraph: TopParagraph
    lime_result: LimeResult
    deletion_test: DeletionTest


class CheckpointListResponse(BaseModel):
    """Response for /checkpoints endpoint"""
    checkpoints: List[str]


class CheckpointLoadResponse(BaseModel):
    """Response for /checkpoints/load endpoint"""
    status: str
    model_name: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
