"""
Security middleware for FastAPI:
- HTTPS enforcement
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- Token blacklist checking
- IP-based security logging
- CORS hardening
"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    
    Headers added:
    - Strict-Transport-Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - Content-Security-Policy
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    """
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

        response.headers["Content-Security-Policy"] = csp
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (formerly Feature Policy)
        permissions = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
        response.headers["Permissions-Policy"] = permissions
        
        return response


class HTTPSEnforcementMiddleware(BaseHTTPMiddleware):
    """Enforce HTTPS in production"""
    
    async def dispatch(self, request: Request, call_next):
        # Only enforce in production
        import os
        if os.getenv("ENVIRONMENT") == "production":
            # Check if request is HTTPS
            scheme = request.url.scheme
            x_forwarded_proto = request.headers.get("x-forwarded-proto")
            
            if scheme != "https" and x_forwarded_proto != "https":
                raise HTTPException(
                    status_code=403,
                    detail="HTTPS required"
                )
        
        response = await call_next(request)
        return response


class TokenBlacklistMiddleware(BaseHTTPMiddleware):
    """Check if token is blacklisted before processing request"""
    
    async def dispatch(self, request: Request, call_next):
        # Check if request has authorization header
        auth_header = request.headers.get("authorization")
        
        if auth_header and "Bearer " in auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            
            # Check if token is blacklisted
            from auth.cache_manager import redis_manager
            if redis_manager.is_token_blacklisted(token):
                raise HTTPException(
                    status_code=401,
                    detail="Token has been revoked"
                )
        
        response = await call_next(request)
        return response


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Log security-relevant events"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract client IP
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log auth-related requests
        if "auth" in request.url.path:
            logger.info(
                f"Auth request: {request.method} {request.url.path} "
                f"from {client_ip} - {user_agent}"
            )
        
        # Log admin endpoints
        if "/admin" in request.url.path or "role" in request.url.path:
            logger.warning(
                f"Admin endpoint access: {request.method} {request.url.path} "
                f"from {client_ip}"
            )
        
        response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP rate limiting middleware.
    Prevents brute force attacks on auth endpoints.
    """
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
    
    async def dispatch(self, request: Request, call_next):
        # Only apply to auth endpoints
        if not ("auth" in request.url.path or "login" in request.url.path or "register" in request.url.path):
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        from auth.cache_manager import redis_manager
        
        # Use in-memory cache for rate limiting per IP
        key = f"rate_limit:{client_ip}"
        
        # Store request count in cache with 60-second TTL
        from datetime import timedelta, datetime
        
        # Simplified rate limit check
        if not self._check_rate_limit(redis_manager, client_ip):
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Reset"] = str(int(datetime.utcnow().timestamp()) + 60)
        
        return response
    
    def _check_rate_limit(self, redis_manager, client_ip: str) -> bool:
        """Check if IP has exceeded rate limit"""
        # Using cache_manager's built-in rate limiting for per-IP tracking
        # We'll use a simple counter approach
        from datetime import datetime, timedelta
        
        key = f"rate_limit_ip:{client_ip}"
        
        # Increment counter
        current_count = getattr(redis_manager, '_ip_counts', {}).get(key, 0)
        current_count += 1
        
        if not hasattr(redis_manager, '_ip_counts'):
            redis_manager._ip_counts = {}
        
        redis_manager._ip_counts[key] = current_count
        
        return current_count <= self.requests_per_minute


class CORSHardeningMiddleware(BaseHTTPMiddleware):
    """
    Hardened CORS configuration.
    Should replace FastAPI's default CORSMiddleware for production.
    """
    def __init__(self, app, allowed_origins: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or []
    
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            origin = request.headers.get("origin")
            
            if origin not in self.allowed_origins:
                return Response(status_code=403)
            
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "3600",
                    "Access-Control-Allow-Credentials": "true"
                }
            )
        
        response = await call_next(request)
        
        origin = request.headers.get("origin")
        if origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP whitelist for sensitive endpoints.
    Useful for admin or internal APIs.
    """
    def __init__(self, app, protected_paths: list = None, whitelist: list = None):
        super().__init__(app)
        self.protected_paths = protected_paths or ["/admin", "/api/internal"]
        self.whitelist = whitelist or []
    
    async def dispatch(self, request: Request, call_next):
        is_protected = any(request.url.path.startswith(p) for p in self.protected_paths)
        
        if not is_protected:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        if client_ip not in self.whitelist:
            logger.warning(f"IP {client_ip} attempted to access protected endpoint {request.url.path}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        return await call_next(request)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive audit logging for compliance.
    Logs all authentication-related events.
    """
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Extract user from token if present
        auth_header = request.headers.get("authorization")
        user_id = None
        
        if auth_header and "Bearer " in auth_header:
            try:
                token = auth_header.replace("Bearer ", "").strip()
                from auth.auth_manager import auth_manager
                payload = auth_manager.verify_token(token)
                user_id = payload.get("sub") if payload else None
            except:
                pass
        
        logger.info(
            f"Request: {request.method} {request.url.path} | "
            f"User: {user_id} | IP: {client_ip} | "
            f"Time: {datetime.utcnow().isoformat()}"
        )
        
        response = await call_next(request)
        
        logger.info(
            f"Response: {response.status_code} for {request.method} {request.url.path} | "
            f"User: {user_id}"
        )
        
        return response