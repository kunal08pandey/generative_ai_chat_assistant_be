import time
from fastapi import HTTPException, Request
from collections import defaultdict
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """
    A simple memory-based rate limiter dependency for FastAPI.
    Limits requests per client IP to `max_requests` per `window_seconds`.
    NOTE: As a dependency, a new instance is created per route. Rate limits
    are not shared across different endpoints globally.
    """
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = defaultdict(list)

    async def __call__(self, request: Request):
        if request is None or not hasattr(request, "client"):
            return  # Safety fallback if not used as a FastAPI dependency correctly
            
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean up old timestamps
        self.clients[client_ip] = [
            timestamp for timestamp in self.clients[client_ip] 
            if now - timestamp < self.window_seconds
        ]
        
        # Check limit
        if len(self.clients[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests. Please try again later."
            )
            
        # Add current request
        self.clients[client_ip].append(now)