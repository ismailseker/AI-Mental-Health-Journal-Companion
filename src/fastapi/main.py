from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes
from .routes import journal_routes, analytics_routes, auth_routes

# Import models
from ..models.advanced_sentiment_model import initialize_advanced_analyzer

# Global variables for models (will be loaded at startup)
sentiment_model = None
emotion_model = None
advanced_analyzer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting AI Mental Health Journal Companion...")
    
    # Load advanced analyzer
    global advanced_analyzer
    try:
        advanced_analyzer = initialize_advanced_analyzer()
        print("✅ Advanced analyzer loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load advanced analyzer: {e}")
        print("🔄 Using simple models only")
        advanced_analyzer = None
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="AI Mental Health Journal Companion",
    description="AI-powered mental health journaling with sentiment analysis, emotion detection, and personalized insights",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(journal_routes.router, prefix="/api/v1/journal", tags=["journal"])
app.include_router(analytics_routes.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Mental Health Journal Companion API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": {
            "advanced_analyzer": advanced_analyzer is not None,
            "bert_available": advanced_analyzer.bert_available if advanced_analyzer else False
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
