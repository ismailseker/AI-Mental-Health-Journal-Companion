from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class EmotionType(str, Enum):
    """Fine-grained emotion types"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    STRESSED = "stressed"
    EXCITED = "excited"
    TIRED = "tired"
    PEACEFUL = "peaceful"
    FRUSTRATED = "frustrated"
    GRATEFUL = "grateful"
    LONELY = "lonely"
    CONFIDENT = "confident"
    OVERWHELMED = "overwhelmed"
    CONTENT = "content"
    WORRIED = "worried"

class SentimentType(str, Enum):
    """Basic sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class JournalEntryCreate(BaseModel):
    """Schema for creating a new journal entry"""
    content: str = Field(..., min_length=1, max_length=5000, description="Journal entry content")
    mood_rating: Optional[int] = Field(None, ge=1, le=10, description="Self-reported mood rating (1-10)")
    tags: Optional[List[str]] = Field(default=[], description="Optional tags for the entry")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Bugün işte çok stresli bir gün geçirdim. Proje deadline'ı yaklaşıyor ve henüz bitiremedim. Ama akşam yürüyüşe çıktım ve biraz rahatladım.",
                "mood_rating": 6,
                "tags": ["iş", "stres", "yürüyüş"]
            }
        }

class JournalEntryUpdate(BaseModel):
    """Schema for updating a journal entry"""
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    mood_rating: Optional[int] = Field(None, ge=1, le=10)
    tags: Optional[List[str]] = None

class SentimentAnalysis(BaseModel):
    """Sentiment analysis results"""
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)

class EmotionAnalysis(BaseModel):
    """Fine-grained emotion analysis results"""
    primary_emotion: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    emotion_scores: Dict[EmotionType, float] = Field(..., description="Scores for all emotions")

class StressAnalysis(BaseModel):
    """Stress level analysis results"""
    stress_level: float = Field(..., ge=0.0, le=1.0, description="Stress level (0=no stress, 1=high stress)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    stress_indicators: List[str] = Field(default=[], description="Detected stress indicators")

class SafetyAnalysis(BaseModel):
    """Safety and self-harm detection results"""
    is_safe: bool = Field(..., description="Whether content is safe")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    concerns: List[str] = Field(default=[], description="Detected concerns")
    requires_attention: bool = Field(..., description="Whether content requires immediate attention")

class AIAnalysis(BaseModel):
    """Complete AI analysis results"""
    sentiment: SentimentAnalysis
    emotions: EmotionAnalysis
    stress: StressAnalysis
    safety: SafetyAnalysis
    analysis_timestamp: datetime
    model_version: str

class JournalEntry(BaseModel):
    """Complete journal entry with AI analysis"""
    id: str
    user_id: str
    content: str
    mood_rating: Optional[int] = None
    tags: List[str] = []
    created_at: datetime
    updated_at: datetime
    ai_analysis: Optional[AIAnalysis] = None
    
    class Config:
        from_attributes = True

class JournalEntryResponse(BaseModel):
    """Response schema for journal entries"""
    entry: JournalEntry
    message: str = "Journal entry processed successfully"

class JournalEntriesList(BaseModel):
    """Response schema for multiple journal entries"""
    entries: List[JournalEntry]
    total_count: int
    page: int
    page_size: int

class AnalyticsSummary(BaseModel):
    """Analytics summary for a time period"""
    period_start: datetime
    period_end: datetime
    total_entries: int
    average_mood: Optional[float] = None
    dominant_emotions: List[EmotionType]
    stress_trend: str  # "increasing", "decreasing", "stable"
    sentiment_distribution: Dict[SentimentType, int]
    insights: List[str] = Field(default=[], description="AI-generated insights")
