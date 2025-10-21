"""
Simple sentiment analysis model for the AI Mental Health Journal Companion.
This is a basic implementation that will be enhanced with more sophisticated models later.
"""

import re
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSentimentAnalyzer:
    """
    A simple rule-based sentiment analyzer for Turkish and English text.
    This will be replaced with a more sophisticated model (BERT, RoBERTa) later.
    """
    
    def __init__(self):
        # Turkish positive words
        self.positive_words_tr = {
            'mutlu', 'sevinçli', 'neşeli', 'keyifli', 'huzurlu', 'rahat', 'iyi', 'güzel',
            'harika', 'mükemmel', 'süper', 'muhteşem', 'gurur', 'başarı', 'başarılı',
            'güven', 'umut', 'umutlu', 'pozitif', 'iyimser', 'sevgi', 'aşk', 'gülümseme',
            'gülmek', 'kahkaha', 'eğlence', 'zevk', 'haz', 'tatmin', 'memnun', 'hoşnut',
            'şükür', 'minnettar', 'grateful', 'happy', 'joy', 'excited', 'wonderful',
            'amazing', 'great', 'good', 'excellent', 'fantastic', 'love', 'smile'
        }
        
        # Turkish negative words
        self.negative_words_tr = {
            'üzgün', 'kederli', 'mutsuz', 'hüzünlü', 'kızgın', 'sinirli', 'öfkeli',
            'stresli', 'gergin', 'endişeli', 'kaygılı', 'korku', 'korkmuş', 'panik',
            'depresif', 'bunalım', 'yorgun', 'bitkin', 'tükenmiş', 'çaresiz', 'umutsuz',
            'hayal kırıklığı', 'hayal kırıklığına uğramış', 'kızgın', 'sinir', 'öfke',
            'kıskanç', 'kıskançlık', 'nefret', 'tiksinti', 'iğrenme', 'üzüntü', 'acı',
            'sad', 'angry', 'stressed', 'anxious', 'worried', 'tired', 'exhausted',
            'depressed', 'frustrated', 'hopeless', 'scared', 'fear', 'hate', 'disgust'
        }
        
        # Stress indicators
        self.stress_indicators = {
            'stres', 'stresli', 'gergin', 'baskı', 'zorlanma', 'tükenmiş', 'bitkin',
            'yorgun', 'bunalım', 'sıkışmış', 'çaresiz', 'umutsuz', 'kaygı', 'endişe',
            'panik', 'stress', 'stressed', 'pressure', 'overwhelmed', 'exhausted',
            'burnout', 'anxiety', 'worried', 'panic', 'helpless', 'hopeless'
        }
        
        # Self-harm indicators (basic)
        self.safety_concerns = {
            'intihar', 'kendimi öldürmek', 'yaşamak istemiyorum', 'hayatımı bitirmek',
            'kendime zarar', 'kendimi kesmek', 'ölmek istiyorum', 'suicide', 'kill myself',
            'end my life', 'hurt myself', 'cut myself', 'want to die', 'self harm'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Turkish characters
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of the given text.
        Returns sentiment analysis results.
        """
        if not text:
            return self._get_neutral_sentiment()
        
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        if not words:
            return self._get_neutral_sentiment()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in self.positive_words_tr)
        negative_count = sum(1 for word in words if word in self.negative_words_tr)
        
        total_words = len(words)
        
        # Calculate scores
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1 - positive_score - negative_score
        
        # Determine sentiment
        if positive_score > negative_score and positive_score > 0.1:
            sentiment = "positive"
            confidence = positive_score
        elif negative_score > positive_score and negative_score > 0.1:
            sentiment = "negative"
            confidence = negative_score
        else:
            sentiment = "neutral"
            confidence = neutral_score
        
        return {
            "sentiment": sentiment,
            "confidence": min(confidence, 1.0),
            "positive_score": positive_score,
            "negative_score": negative_score,
            "neutral_score": max(neutral_score, 0.0)
        }
    
    def analyze_emotions(self, text: str) -> Dict:
        """
        Analyze fine-grained emotions in the text.
        This is a simplified version - will be enhanced with ML models later.
        """
        if not text:
            return self._get_neutral_emotion()
        
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Simple emotion detection based on keywords
        emotion_scores = {
            "happy": 0.0, "sad": 0.0, "angry": 0.0, "anxious": 0.0,
            "stressed": 0.0, "excited": 0.0, "tired": 0.0, "peaceful": 0.0,
            "frustrated": 0.0, "grateful": 0.0, "lonely": 0.0, "confident": 0.0,
            "overwhelmed": 0.0, "content": 0.0, "worried": 0.0
        }
        
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "happy": ["mutlu", "sevinçli", "neşeli", "happy", "joy"],
            "sad": ["üzgün", "kederli", "mutsuz", "sad", "sorrow"],
            "angry": ["kızgın", "sinirli", "öfkeli", "angry", "mad"],
            "anxious": ["endişeli", "kaygılı", "anxious", "worried"],
            "stressed": ["stresli", "gergin", "stressed", "tense"],
            "tired": ["yorgun", "bitkin", "tired", "exhausted"],
            "grateful": ["şükür", "minnettar", "grateful", "thankful"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            for word in words:
                if word in keywords:
                    emotion_scores[emotion] += 1.0
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_score
        
        # Find primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        return {
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "emotion_scores": emotion_scores
        }
    
    def analyze_stress(self, text: str) -> Dict:
        """
        Analyze stress level in the text.
        """
        if not text:
            return {"stress_level": 0.0, "confidence": 0.0, "stress_indicators": []}
        
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Count stress indicators
        stress_count = sum(1 for word in words if word in self.stress_indicators)
        total_words = len(words)
        
        # Calculate stress level
        stress_level = min(stress_count / max(total_words, 1) * 5, 1.0)  # Scale to 0-1
        
        # Find specific stress indicators
        found_indicators = [word for word in words if word in self.stress_indicators]
        
        return {
            "stress_level": stress_level,
            "confidence": min(stress_count / max(total_words, 1), 1.0),
            "stress_indicators": found_indicators
        }
    
    def analyze_safety(self, text: str) -> Dict:
        """
        Analyze safety and detect potential self-harm indicators.
        """
        if not text:
            return {
                "is_safe": True,
                "risk_level": "low",
                "concerns": [],
                "requires_attention": False
            }
        
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        # Check for safety concerns
        found_concerns = [word for word in words if word in self.safety_concerns]
        
        if found_concerns:
            return {
                "is_safe": False,
                "risk_level": "high",
                "concerns": found_concerns,
                "requires_attention": True
            }
        
        return {
            "is_safe": True,
            "risk_level": "low",
            "concerns": [],
            "requires_attention": False
        }
    
    def analyze_complete(self, text: str) -> Dict:
        """
        Perform complete analysis of the text.
        Returns all analysis results.
        """
        logger.info(f"Analyzing text: {text[:100]}...")
        
        sentiment_result = self.analyze_sentiment(text)
        emotion_result = self.analyze_emotions(text)
        stress_result = self.analyze_stress(text)
        safety_result = self.analyze_safety(text)
        
        return {
            "sentiment": sentiment_result,
            "emotions": emotion_result,
            "stress": stress_result,
            "safety": safety_result,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_version": "simple_v1.0"
        }
    
    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment when text is empty"""
        return {
            "sentiment": "neutral",
            "confidence": 1.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_score": 1.0
        }
    
    def _get_neutral_emotion(self) -> Dict:
        """Return neutral emotion when text is empty"""
        return {
            "primary_emotion": "content",
            "confidence": 1.0,
            "emotion_scores": {
                "happy": 0.0, "sad": 0.0, "angry": 0.0, "anxious": 0.0,
                "stressed": 0.0, "excited": 0.0, "tired": 0.0, "peaceful": 0.0,
                "frustrated": 0.0, "grateful": 0.0, "lonely": 0.0, "confident": 0.0,
                "overwhelmed": 0.0, "content": 1.0, "worried": 0.0
            }
        }

# Global instance
sentiment_analyzer = SimpleSentimentAnalyzer()

def get_sentiment_analyzer() -> SimpleSentimentAnalyzer:
    """Get the global sentiment analyzer instance"""
    return sentiment_analyzer
