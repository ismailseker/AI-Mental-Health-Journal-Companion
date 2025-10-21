"""
Journal routes for the AI Mental Health Journal Companion.
Handles CRUD operations for journal entries and AI analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
import uuid
import logging

from ..schemas.journal_schema import (
    JournalEntryCreate, 
    JournalEntryUpdate, 
    JournalEntry, 
    JournalEntryResponse,
    JournalEntriesList,
    AIAnalysis
)
from ...models.sentiment_model import get_sentiment_analyzer
from ...models.advanced_sentiment_model import get_advanced_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# In-memory storage for demo (will be replaced with database later)
journal_entries_db = {}

@router.post("/entries", response_model=JournalEntryResponse)
async def create_journal_entry(
    entry_data: JournalEntryCreate,
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Create a new journal entry with AI analysis.
    """
    try:
        # Generate unique ID
        entry_id = str(uuid.uuid4())
        
        # Get current timestamp
        now = datetime.now()
        
        # Perform AI analysis
        logger.info(f"Analyzing journal entry for user {user_id}")
        
        # Try advanced analyzer first, fallback to simple analyzer
        try:
            advanced_analyzer = get_advanced_analyzer()
            if advanced_analyzer:
                ai_analysis_data = advanced_analyzer.analyze_complete(entry_data.content)
            else:
                raise Exception("Advanced analyzer not available")
        except Exception as e:
            logger.warning(f"Advanced analyzer failed: {e}, using simple analyzer")
            analyzer = get_sentiment_analyzer()
            ai_analysis_data = analyzer.analyze_complete(entry_data.content)
        
        # Create AI analysis object
        ai_analysis = AIAnalysis(
            sentiment=ai_analysis_data["sentiment"],
            emotions=ai_analysis_data["emotions"],
            stress=ai_analysis_data["stress"],
            safety=ai_analysis_data["safety"],
            analysis_timestamp=datetime.fromisoformat(ai_analysis_data["analysis_timestamp"]),
            model_version=ai_analysis_data["model_version"]
        )
        
        # Create journal entry
        journal_entry = JournalEntry(
            id=entry_id,
            user_id=user_id,
            content=entry_data.content,
            mood_rating=entry_data.mood_rating,
            tags=entry_data.tags or [],
            created_at=now,
            updated_at=now,
            ai_analysis=ai_analysis
        )
        
        # Store in database (in-memory for now)
        journal_entries_db[entry_id] = journal_entry
        
        logger.info(f"Created journal entry {entry_id} for user {user_id}")
        
        return JournalEntryResponse(
            entry=journal_entry,
            message="Journal entry created and analyzed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating journal entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create journal entry: {str(e)}")

@router.get("/entries/{entry_id}", response_model=JournalEntryResponse)
async def get_journal_entry(
    entry_id: str,
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Get a specific journal entry by ID.
    """
    if entry_id not in journal_entries_db:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    
    entry = journal_entries_db[entry_id]
    
    # Check if user owns this entry
    if entry.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return JournalEntryResponse(
        entry=entry,
        message="Journal entry retrieved successfully"
    )

@router.get("/entries", response_model=JournalEntriesList)
async def get_journal_entries(
    user_id: str = "demo_user",  # TODO: Get from authentication
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of entries per page"),
    start_date: Optional[datetime] = Query(None, description="Filter entries from this date"),
    end_date: Optional[datetime] = Query(None, description="Filter entries until this date")
):
    """
    Get journal entries for a user with pagination and date filtering.
    """
    try:
        # Filter entries by user
        user_entries = [
            entry for entry in journal_entries_db.values() 
            if entry.user_id == user_id
        ]
        
        # Apply date filters if provided
        if start_date:
            user_entries = [entry for entry in user_entries if entry.created_at >= start_date]
        
        if end_date:
            user_entries = [entry for entry in user_entries if entry.created_at <= end_date]
        
        # Sort by creation date (newest first)
        user_entries.sort(key=lambda x: x.created_at, reverse=True)
        
        # Calculate pagination
        total_count = len(user_entries)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_entries = user_entries[start_idx:end_idx]
        
        return JournalEntriesList(
            entries=paginated_entries,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error retrieving journal entries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve journal entries: {str(e)}")

@router.put("/entries/{entry_id}", response_model=JournalEntryResponse)
async def update_journal_entry(
    entry_id: str,
    entry_update: JournalEntryUpdate,
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Update an existing journal entry.
    """
    if entry_id not in journal_entries_db:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    
    entry = journal_entries_db[entry_id]
    
    # Check if user owns this entry
    if entry.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Update fields if provided
        if entry_update.content is not None:
            entry.content = entry_update.content
            # Re-analyze if content changed
            analyzer = get_sentiment_analyzer()
            ai_analysis_data = analyzer.analyze_complete(entry.content)
            
            entry.ai_analysis = AIAnalysis(
                sentiment=ai_analysis_data["sentiment"],
                emotions=ai_analysis_data["emotions"],
                stress=ai_analysis_data["stress"],
                safety=ai_analysis_data["safety"],
                analysis_timestamp=datetime.fromisoformat(ai_analysis_data["analysis_timestamp"]),
                model_version=ai_analysis_data["model_version"]
            )
        
        if entry_update.mood_rating is not None:
            entry.mood_rating = entry_update.mood_rating
        
        if entry_update.tags is not None:
            entry.tags = entry_update.tags
        
        # Update timestamp
        entry.updated_at = datetime.now()
        
        # Save back to database
        journal_entries_db[entry_id] = entry
        
        logger.info(f"Updated journal entry {entry_id} for user {user_id}")
        
        return JournalEntryResponse(
            entry=entry,
            message="Journal entry updated successfully"
        )
        
    except Exception as e:
        logger.error(f"Error updating journal entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update journal entry: {str(e)}")

@router.delete("/entries/{entry_id}")
async def delete_journal_entry(
    entry_id: str,
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Delete a journal entry.
    """
    if entry_id not in journal_entries_db:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    
    entry = journal_entries_db[entry_id]
    
    # Check if user owns this entry
    if entry.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Delete from database
        del journal_entries_db[entry_id]
        
        logger.info(f"Deleted journal entry {entry_id} for user {user_id}")
        
        return {"message": "Journal entry deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting journal entry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete journal entry: {str(e)}")

@router.post("/analyze")
async def analyze_text(
    text: str,
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Analyze text without creating a journal entry.
    Useful for real-time analysis or testing.
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        analyzer = get_sentiment_analyzer()
        analysis_result = analyzer.analyze_complete(text)
        
        return {
            "analysis": analysis_result,
            "message": "Text analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze text: {str(e)}")

@router.get("/stats")
async def get_journal_stats(
    user_id: str = "demo_user"  # TODO: Get from authentication
):
    """
    Get basic statistics for user's journal entries.
    """
    try:
        # Filter entries by user
        user_entries = [
            entry for entry in journal_entries_db.values() 
            if entry.user_id == user_id
        ]
        
        if not user_entries:
            return {
                "total_entries": 0,
                "message": "No journal entries found"
            }
        
        # Calculate basic stats
        total_entries = len(user_entries)
        
        # Sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        mood_ratings = []
        
        for entry in user_entries:
            if entry.ai_analysis:
                sentiment = entry.ai_analysis.sentiment.sentiment
                sentiment_counts[sentiment] += 1
            
            if entry.mood_rating:
                mood_ratings.append(entry.mood_rating)
        
        # Calculate average mood
        average_mood = sum(mood_ratings) / len(mood_ratings) if mood_ratings else None
        
        return {
            "total_entries": total_entries,
            "sentiment_distribution": sentiment_counts,
            "average_mood_rating": average_mood,
            "entries_with_mood_rating": len(mood_ratings),
            "message": "Statistics calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error calculating stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")
