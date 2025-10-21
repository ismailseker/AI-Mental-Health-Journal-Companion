"""
Advanced sentiment analysis model that combines BERT and simple models.
This will be the main model used in the API.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
from pathlib import Path

# Import our models
from .bert_sentiment_model import BERTMentalHealthTrainer
from .sentiment_model import SimpleSentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMentalHealthAnalyzer:
    """
    Advanced mental health analyzer that combines BERT and rule-based models.
    """
    
    def __init__(self, bert_model_path: Optional[str] = None):
        self.bert_trainer = None
        self.simple_analyzer = SimpleSentimentAnalyzer()
        self.bert_available = False
        
        # Try to load BERT model if path provided
        if bert_model_path and Path(bert_model_path).exists():
            try:
                self.bert_trainer = BERTMentalHealthTrainer()
                self.bert_trainer.load_model(bert_model_path)
                self.bert_available = True
                logger.info("✅ BERT model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load BERT model: {e}")
                logger.info("🔄 Falling back to simple model")
        else:
            logger.info("🔄 Using simple model (BERT not available)")
    
    def analyze_sentiment_bert(self, text: str) -> Dict:
        """Analyze sentiment using BERT model."""
        if not self.bert_available:
            return None
        
        try:
            result = self.bert_trainer.predict(text)
            return {
                "sentiment": result['prediction'],
                "confidence": result['confidence'],
                "model": "bert"
            }
        except Exception as e:
            logger.error(f"BERT prediction error: {e}")
            return None
    
    def analyze_sentiment_simple(self, text: str) -> Dict:
        """Analyze sentiment using simple model."""
        result = self.simple_analyzer.analyze_sentiment(text)
        return {
            "sentiment": result['sentiment'],
            "confidence": result['confidence'],
            "model": "simple"
        }
    
    def analyze_sentiment_ensemble(self, text: str) -> Dict:
        """Analyze sentiment using ensemble of models."""
        
        # Get predictions from both models
        bert_result = self.analyze_sentiment_bert(text)
        simple_result = self.analyze_sentiment_simple(text)
        
        if bert_result and simple_result:
            # Ensemble prediction
            bert_weight = 0.7  # BERT gets more weight
            simple_weight = 0.3
            
            # Combine predictions
            if bert_result['sentiment'] == simple_result['sentiment']:
                # Both models agree
                final_sentiment = bert_result['sentiment']
                final_confidence = (bert_result['confidence'] * bert_weight + 
                                  simple_result['confidence'] * simple_weight)
            else:
                # Models disagree, trust BERT more
                final_sentiment = bert_result['sentiment']
                final_confidence = bert_result['confidence'] * 0.8  # Slightly reduce confidence
        elif bert_result:
            # Only BERT available
            final_sentiment = bert_result['sentiment']
            final_confidence = bert_result['confidence']
        else:
            # Only simple model available
            final_sentiment = simple_result['sentiment']
            final_confidence = simple_result['confidence']
        
        return {
            "sentiment": final_sentiment,
            "confidence": final_confidence,
            "model": "ensemble",
            "bert_result": bert_result,
            "simple_result": simple_result
        }
    
    def analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions using simple model (will be enhanced with BERT later)."""
        return self.simple_analyzer.analyze_emotions(text)
    
    def analyze_stress(self, text: str) -> Dict:
        """Analyze stress level using simple model."""
        return self.simple_analyzer.analyze_stress(text)
    
    def analyze_safety(self, text: str) -> Dict:
        """Analyze safety using simple model."""
        return self.simple_analyzer.analyze_safety(text)
    
    def analyze_complete(self, text: str) -> Dict:
        """
        Perform complete analysis using the best available models.
        """
        logger.info(f"🔍 Analyzing text: {text[:100]}...")
        
        # Get sentiment analysis (ensemble if BERT available)
        if self.bert_available:
            sentiment_result = self.analyze_sentiment_ensemble(text)
        else:
            sentiment_result = self.analyze_sentiment_simple(text)
        
        # Get other analyses (using simple model for now)
        emotion_result = self.analyze_emotions(text)
        stress_result = self.analyze_stress(text)
        safety_result = self.analyze_safety(text)
        
        # Format sentiment result to match schema
        formatted_sentiment = {
            "sentiment": sentiment_result['sentiment'],
            "confidence": sentiment_result['confidence'],
            "positive_score": 1.0 if sentiment_result['sentiment'] == 'positive' else 0.0,
            "negative_score": 1.0 if sentiment_result['sentiment'] == 'negative' else 0.0,
            "neutral_score": 1.0 if sentiment_result['sentiment'] == 'neutral' else 0.0
        }
        
        return {
            "sentiment": formatted_sentiment,
            "emotions": emotion_result,
            "stress": stress_result,
            "safety": safety_result,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_version": "advanced_v1.0"
        }

# Global instance
advanced_analyzer = None

def get_advanced_analyzer(bert_model_path: Optional[str] = None) -> AdvancedMentalHealthAnalyzer:
    """Get the global advanced analyzer instance."""
    global advanced_analyzer
    
    if advanced_analyzer is None:
        advanced_analyzer = AdvancedMentalHealthAnalyzer(bert_model_path)
    
    return advanced_analyzer

def initialize_advanced_analyzer(bert_model_path: Optional[str] = None):
    """Initialize the advanced analyzer with BERT model."""
    global advanced_analyzer
    
    # Default BERT model path
    if bert_model_path is None:
        bert_model_path = "/Users/sekerismail/Desktop/AIMentalHealthJournalCompanion/models/bert_mental_health"
    
    advanced_analyzer = AdvancedMentalHealthAnalyzer(bert_model_path)
    return advanced_analyzer
