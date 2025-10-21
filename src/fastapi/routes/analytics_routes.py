"""
Analytics routes for the AI Mental Health Journal Companion.
Handles analytics, insights, and trend analysis.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from ..schemas.journal_schema import AnalyticsSummary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Import journal entries database (in-memory for now)
from .journal_routes import journal_entries_db

@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    user_id: str = "demo_user",  # TODO: Get from authentication
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get analytics summary for a specific time period.
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter entries by user and date range
        user_entries = [
            entry for entry in journal_entries_db.values()
            if entry.user_id == user_id and start_date <= entry.created_at <= end_date
        ]
        
        if not user_entries:
            return AnalyticsSummary(
                period_start=start_date,
                period_end=end_date,
                total_entries=0,
                dominant_emotions=[],
                stress_trend="stable",
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
                insights=["No entries found for this period"]
            )
        
        # Calculate statistics
        total_entries = len(user_entries)
        
        # Sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        mood_ratings = []
        stress_levels = []
        emotion_counts = {}
        
        for entry in user_entries:
            if entry.ai_analysis:
                # Sentiment
                sentiment = entry.ai_analysis.sentiment.sentiment
                sentiment_counts[sentiment] += 1
                
                # Stress levels
                stress_levels.append(entry.ai_analysis.stress.stress_level)
                
                # Emotions
                primary_emotion = entry.ai_analysis.emotions.primary_emotion
                emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
            
            if entry.mood_rating:
                mood_ratings.append(entry.mood_rating)
        
        # Calculate average mood
        average_mood = sum(mood_ratings) / len(mood_ratings) if mood_ratings else None
        
        # Find dominant emotions (top 3)
        dominant_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        dominant_emotions = [emotion for emotion, count in dominant_emotions]
        
        # Calculate stress trend
        if len(stress_levels) >= 2:
            recent_stress = sum(stress_levels[-len(stress_levels)//2:]) / (len(stress_levels)//2)
            older_stress = sum(stress_levels[:len(stress_levels)//2]) / (len(stress_levels)//2)
            
            if recent_stress > older_stress * 1.1:
                stress_trend = "increasing"
            elif recent_stress < older_stress * 0.9:
                stress_trend = "decreasing"
            else:
                stress_trend = "stable"
        else:
            stress_trend = "stable"
        
        # Generate insights
        insights = []
        
        if sentiment_counts["positive"] > sentiment_counts["negative"]:
            insights.append("Your overall sentiment has been positive recently! 🌟")
        elif sentiment_counts["negative"] > sentiment_counts["positive"]:
            insights.append("You've been experiencing more negative emotions lately. Consider reaching out for support. 💙")
        
        if stress_trend == "increasing":
            insights.append("Your stress levels have been increasing. Consider stress management techniques. 🧘")
        elif stress_trend == "decreasing":
            insights.append("Great news! Your stress levels have been decreasing. 🎉")
        
        if average_mood and average_mood < 5:
            insights.append("Your mood ratings have been low. Consider activities that bring you joy. 😊")
        elif average_mood and average_mood > 7:
            insights.append("You've been in a great mood! Keep up the positive energy! ✨")
        
        if total_entries < days // 7:  # Less than 1 entry per week
            insights.append("Consider journaling more regularly for better insights. 📝")
        
        return AnalyticsSummary(
            period_start=start_date,
            period_end=end_date,
            total_entries=total_entries,
            average_mood=average_mood,
            dominant_emotions=dominant_emotions,
            stress_trend=stress_trend,
            sentiment_distribution=sentiment_counts,
            insights=insights
        )
        
    except Exception as e:
        logger.error(f"Error generating analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics summary: {str(e)}")

@router.get("/trends")
async def get_trends(
    user_id: str = "demo_user",  # TODO: Get from authentication
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get detailed trends for mood, stress, and emotions over time.
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter entries by user and date range
        user_entries = [
            entry for entry in journal_entries_db.values()
            if entry.user_id == user_id and start_date <= entry.created_at <= end_date
        ]
        
        if not user_entries:
            return {
                "message": "No entries found for this period",
                "trends": []
            }
        
        # Sort entries by date
        user_entries.sort(key=lambda x: x.created_at)
        
        # Generate trend data
        trends = []
        for entry in user_entries:
            if entry.ai_analysis:
                trend_point = {
                    "date": entry.created_at.isoformat(),
                    "sentiment": entry.ai_analysis.sentiment.sentiment,
                    "sentiment_confidence": entry.ai_analysis.sentiment.confidence,
                    "primary_emotion": entry.ai_analysis.emotions.primary_emotion,
                    "emotion_confidence": entry.ai_analysis.emotions.confidence,
                    "stress_level": entry.ai_analysis.stress.stress_level,
                    "mood_rating": entry.mood_rating
                }
                trends.append(trend_point)
        
        return {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_data_points": len(trends),
            "trends": trends,
            "message": "Trends calculated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error calculating trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate trends: {str(e)}")

@router.get("/insights")
async def get_insights(
    user_id: str = "demo_user",  # TODO: Get from authentication
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get AI-generated insights and recommendations.
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter entries by user and date range
        user_entries = [
            entry for entry in journal_entries_db.values()
            if entry.user_id == user_id and start_date <= entry.created_at <= end_date
        ]
        
        if not user_entries:
            return {
                "insights": ["No entries found for this period. Start journaling to get personalized insights!"],
                "recommendations": ["Consider writing your first journal entry to begin your mental health journey."]
            }
        
        # Analyze patterns and generate insights
        insights = []
        recommendations = []
        
        # Sentiment analysis
        positive_count = sum(1 for entry in user_entries 
                           if entry.ai_analysis and entry.ai_analysis.sentiment.sentiment == "positive")
        negative_count = sum(1 for entry in user_entries 
                           if entry.ai_analysis and entry.ai_analysis.sentiment.sentiment == "negative")
        
        if positive_count > negative_count * 1.5:
            insights.append("You've been maintaining a positive outlook recently! 🌟")
            recommendations.append("Continue your positive mindset and consider sharing your positivity with others.")
        elif negative_count > positive_count * 1.5:
            insights.append("You've been experiencing more challenging emotions lately.")
            recommendations.append("Consider practicing mindfulness, reaching out to friends, or seeking professional support.")
        
        # Stress analysis
        stress_levels = [entry.ai_analysis.stress.stress_level for entry in user_entries 
                        if entry.ai_analysis]
        if stress_levels:
            avg_stress = sum(stress_levels) / len(stress_levels)
            if avg_stress > 0.7:
                insights.append("Your stress levels have been consistently high.")
                recommendations.append("Try stress management techniques like deep breathing, meditation, or physical exercise.")
            elif avg_stress < 0.3:
                insights.append("You've been managing stress well!")
                recommendations.append("Keep up your stress management practices.")
        
        # Mood analysis
        mood_ratings = [entry.mood_rating for entry in user_entries if entry.mood_rating]
        if mood_ratings:
            avg_mood = sum(mood_ratings) / len(mood_ratings)
            if avg_mood < 5:
                insights.append("Your mood ratings have been on the lower side.")
                recommendations.append("Consider activities that bring you joy, like hobbies, socializing, or spending time in nature.")
            elif avg_mood > 7:
                insights.append("You've been in great spirits!")
                recommendations.append("Your positive mood is wonderful! Consider journaling about what's contributing to your happiness.")
        
        # Consistency analysis
        if len(user_entries) < days // 3:  # Less than 1 entry every 3 days
            insights.append("Your journaling frequency could be more consistent.")
            recommendations.append("Try to journal at least every other day for better insights and mental health benefits.")
        
        # Default insights if none generated
        if not insights:
            insights.append("Keep up your journaling practice! Regular reflection is beneficial for mental health.")
        
        if not recommendations:
            recommendations.append("Continue your journaling journey and consider exploring new self-care activities.")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "analysis_period": f"{days} days",
            "total_entries_analyzed": len(user_entries)
        }
        
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")
