"""
Authentication routes for the AI Mental Health Journal Companion.
Basic authentication setup (will be enhanced later).
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Simple in-memory user storage for demo
users_db = {
    "demo_user": {
        "id": "demo_user",
        "username": "demo_user",
        "email": "demo@example.com",
        "is_active": True
    }
}

@router.post("/login")
async def login(username: str, password: str):
    """
    Simple login endpoint for demo purposes.
    In production, this would use proper authentication with JWT tokens.
    """
    # For demo purposes, accept any username/password combination
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    # Create user if doesn't exist
    if username not in users_db:
        users_db[username] = {
            "id": username,
            "username": username,
            "email": f"{username}@example.com",
            "is_active": True
        }
    
    logger.info(f"User {username} logged in")
    
    return {
        "message": "Login successful",
        "user_id": username,
        "access_token": f"demo_token_{username}",  # Demo token
        "token_type": "bearer"
    }

@router.post("/register")
async def register(username: str, email: str, password: str):
    """
    Simple registration endpoint for demo purposes.
    """
    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Username, email, and password are required")
    
    if username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    users_db[username] = {
        "id": username,
        "username": username,
        "email": email,
        "is_active": True
    }
    
    logger.info(f"New user registered: {username}")
    
    return {
        "message": "Registration successful",
        "user_id": username,
        "access_token": f"demo_token_{username}",  # Demo token
        "token_type": "bearer"
    }

@router.get("/me")
async def get_current_user(user_id: str = "demo_user"):
    """
    Get current user information.
    """
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[user_id]
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "is_active": user["is_active"]
    }

@router.post("/logout")
async def logout():
    """
    Simple logout endpoint.
    """
    return {"message": "Logout successful"}
