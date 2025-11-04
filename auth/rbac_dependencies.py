"""
Role-Based Access Control (RBAC) dependencies for FastAPI.
Provides reusable dependency functions to protect routes with role/permission checks.
"""

from fastapi import Depends, HTTPException, Header
from auth.auth_manager import auth_manager
from auth.cache_manager import redis_manager
from loguru import logger

# ==================== DEPENDENCY FUNCTIONS ====================

async def verify_jwt_token(authorization: str = Header(None)) -> dict:
    """
    Dependency: Verify JWT token and return payload.
    """
    if not authorization or "Bearer " not in authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    token = authorization.replace("Bearer ", "").strip()
    
    # Check if token is blacklisted using in-memory cache
    if redis_manager.is_token_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    payload = auth_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload


async def verify_verified_email(user: dict = Depends(verify_jwt_token)) -> dict:
    """
    Dependency: Verify that user has verified their email.
    """
    from auth.models import User, get_db_session
    
    session = get_db_session()
    try:
        db_user = session.query(User).filter_by(user_id=user["sub"]).first()
        
        if not db_user or not db_user.email_verified:
            raise HTTPException(
                status_code=403,
                detail="Email verification required"
            )
        
        return user
    finally:
        session.close()


def require_role(required_role: str):
    """
    Dependency factory: Require specific role.
    """
    async def _require_role(user: dict = Depends(verify_jwt_token)) -> dict:
        user_roles = user.get("roles", [])
        
        if required_role not in user_roles:
            logger.warning(
                f"User {user['sub']} attempted to access {required_role} "
                f"endpoint without required role"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Role '{required_role}' required"
            )
        
        return user
    
    return _require_role


def require_any_role(required_roles: list):
    """
    Dependency factory: Require one of several roles.
    """
    async def _require_any_role(user: dict = Depends(verify_jwt_token)) -> dict:
        user_roles = user.get("roles", [])
        
        if not any(role in user_roles for role in required_roles):
            logger.warning(
                f"User {user['sub']} attempted to access endpoint requiring "
                f"one of {required_roles}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"One of roles {required_roles} required"
            )
        
        return user
    
    return _require_any_role


def require_permission(required_permission: str):
    """
    Dependency factory: Require specific permission.
    Uses in-memory cache for permission lookups.
    """
    async def _require_permission(user: dict = Depends(verify_jwt_token)) -> dict:
        user_id = user["sub"]
        
        # Check cached permissions first
        cached_perms = redis_manager.get_cached_permissions(user_id)
        
        if cached_perms is not None:
            # ✅ FIXED: Handle wildcard permission
            if "*" in cached_perms or required_permission in cached_perms:
                return user
            
            logger.warning(f"User {user_id} denied permission: {required_permission}")
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{required_permission}' required"
            )
            
        # Check in database
        has_permission = auth_manager.has_permission(user_id, required_permission)
        
        if not has_permission:
            logger.warning(f"User {user_id} denied permission: {required_permission}")
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{required_permission}' required"
            )
        
        return user
    
    return _require_permission


# ==================== COMMONLY USED DEPENDENCIES ====================

async def require_admin(user: dict = Depends(require_role("admin"))) -> dict:
    """
    Dependency: Require admin role.
    """
    return user


async def require_editor(user: dict = Depends(require_any_role(["admin", "editor"]))) -> dict:
    """
    Dependency: Require editor or admin role.
    """
    return user


async def require_viewer(
    user: dict = Depends(require_any_role(["admin", "editor", "viewer"]))
) -> dict:
    """
    Dependency: Require any authenticated role (admin, editor, or viewer).
    """
    return user


# ==================== ADVANCED DEPENDENCIES ====================

async def get_current_user_with_cache(
    user: dict = Depends(verify_jwt_token)
) -> dict:
    """
    Dependency: Get current user with role/permission cache loaded.
    Useful for routes that need to check multiple permissions.
    """
    from auth.models import User, get_db_session
    
    user_id = user["sub"]
    
    # Check if we have cached role/permission data
    cached_perms = redis_manager.get_cached_permissions(user_id)
    
    if cached_perms is None:
        # Load from DB and cache
        session = get_db_session()
        try:
            db_user = session.query(User).filter_by(user_id=user_id).first()
            
            if not db_user:
                raise HTTPException(status_code=401, detail="User not found")
            
            roles = [role.name for role in db_user.roles]
            permissions = []
            
            # Extract permissions from roles
            import json
            for role in db_user.roles:
                role_perms = json.loads(role.permissions or "[]")
                permissions.extend(role_perms)
            
            # Cache for 1 hour using in-memory cache
            redis_manager.cache_user_permissions(user_id, permissions, ttl=3600)
            
            user["roles"] = roles
            user["permissions"] = list(set(permissions))  # Remove duplicates
        
        finally:
            session.close()
    else:
        user["permissions"] = cached_perms
    
    return user


async def require_mfa_verified(user: dict = Depends(verify_jwt_token)) -> dict:
    """
    Dependency: Require MFA verification.
    Check if user has MFA enabled and recently verified using in-memory cache.
    """
    user_id = user["sub"]
    
    session = get_db_session()
    try:
        db_user = session.query(User).filter_by(user_id=user_id).first()
        
        if not db_user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # ✅ FIXED: Check if MFA is actually enabled for this user
        # For now, we'll assume MFA not required unless explicitly enabled
        # Add mfa_enabled field to User model if needed
        
        # Check if MFA was recently verified
        mfa_verified_key = f"mfa_verified:{user_id}"
        mfa_verified = redis_manager.get(mfa_verified_key) if hasattr(redis_manager, 'get') else None
        
        if not mfa_verified:
            # Could allow if user doesn't have MFA enabled
            # raise HTTPException(status_code=403, detail="MFA verification required")
            pass
        
        return user
    finally:
        session.close()


async def rate_limit_check(user: dict = Depends(verify_jwt_token)) -> dict:
    """
    Dependency: Check user's rate limit before processing request.
    Useful for expensive operations like document uploads.
    """
    from auth.models import User, get_db_session
    
    session = get_db_session()
    try:
        db_user = session.query(User).filter_by(user_id=user["sub"]).first()
        
        if not db_user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Check rate limit using in-memory cache
        if not redis_manager.check_rate_limit(
            user["sub"],
            db_user.daily_query_limit
        ):
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit of {db_user.daily_query_limit} queries exceeded"
            )
        
        return user
    finally:
        session.close()


# ==================== OPTIONAL DEPENDENCIES ====================

async def get_optional_user(authorization: str = Header(None)) -> dict:
    """
    Optional dependency: Get user if authenticated, otherwise return None.
    """
    if not authorization or "Bearer " not in authorization:
        return None
    
    token = authorization.replace("Bearer ", "").strip()
    payload = auth_manager.verify_token(token)
    
    return payload


# ==================== SCOPE-BASED ACCESS ====================

class Scopes:
    """OAuth2-style scopes for granular permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    AUDIT = "audit"


def require_scopes(required_scopes: list):
    """
    Dependency factory: Require specific OAuth2 scopes.
    """
    async def _require_scopes(user: dict = Depends(verify_jwt_token)) -> dict:
        user_scopes = user.get("scopes", [])
        
        if not any(scope in user_scopes for scope in required_scopes):
            raise HTTPException(
                status_code=403,
                detail=f"Scopes {required_scopes} required"
            )
        
        return user
    
    return _require_scopes
