"""
FastAPI authentication endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from auth.auth_manager import auth_manager
from auth.models import User, RefreshToken, get_db_session
from auth.cache_manager import redis_manager
from loguru import logger
import json
import re

from auth.tier_config import validate_tier

router = APIRouter(prefix="/auth", tags=["auth"])

# ==================== REQUEST MODELS ====================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    company: str
    tier: str = Field(default="free", description="Subscription tier (free, basic, pro, enterprise)")
    
    @validator('tier')
    def validate_tier_value(cls, v):
        if not validate_tier(v):
            raise ValueError("Invalid tier. Must be one of: free, basic, pro, enterprise")
        return v.lower()


class LoginRequest(BaseModel):
    email: str
    password: str

class VerifyEmailRequest(BaseModel):
    token: str


class RequestPasswordResetRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    user_id: str
    reset_token: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RoleAssignmentRequest(BaseModel):
    role_name: str

# ==================== HELPER FUNCTIONS ====================

def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    if request.client:
        return request.client.host
    return "unknown"


def get_user_agent(request: Request) -> str:
    """Extract User-Agent from request"""
    return request.headers.get("user-agent", "unknown")

# ==================== EMAIL VERIFICATION ====================

@router.post("/register")
async def register(data: RegisterRequest, request: Request):
    """Register new user with email verification required."""
    try:
        # Validate tier
        if not validate_tier(data.tier):
            raise HTTPException(status_code=400, detail="Invalid tier")
        
        # Register user
        result = auth_manager.register(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            company=data.company,
            tier=data.tier
        )
        
        if "error" in result:
            auth_manager.log_audit_event(
                None, "registration_failed",
                {"email": data.email, "reason": result["error"]},
                status="failure",
                ip_address=get_client_ip(request)
            )
            raise HTTPException(status_code=400, detail=result["error"])
        
        user_id = result["user_id"]
        
        # Generate email verification token
        verification_token = auth_manager.generate_email_verification_token(user_id)
        
        # Send verification email
        # auth_manager.send_verification_email(data.email, data.full_name, verification_token)
        
        # Log audit event
        auth_manager.log_audit_event(
            user_id, "user_registered",
            {"email": data.email, "tier": data.tier},
            ip_address=get_client_ip(request)
        )
        
        logger.info(f"User registered: {data.email}")
        
        return {
            "success": True,
            "message": "Registration successful. Please check your email to verify your account.",
            "user_id": user_id,
            "email": data.email,
            "tier": data.tier
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")



@router.post("/verify-email")
async def verify_email(data: VerifyEmailRequest, request: Request):
    """
    Verify email address using token from email.
    
    User must verify email before they can login.
    """
    try:
        result = auth_manager.verify_email(data.token)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "message": "Email verified successfully",
            "email": result.get("email")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification error: {e}")
        raise HTTPException(status_code=400, detail="Email verification failed")


@router.post("/resend-verification-email")
async def resend_verification_email(email: dict, request: Request):
    """
    Resend verification email if user hasn't verified yet.
    """
    try:
        from models import User, get_db_session
        
        session = get_db_session()
        user = session.query(User).filter_by(email=email.get("email")).first()
        session.close()
        
        if not user:
            # Don't reveal if email exists
            return {"success": True, "message": "If email exists, verification email sent"}
        
        if user.email_verified:
            return {"success": True, "message": "Email already verified"}
        
        # Generate new token and send
        verification_token = auth_manager.generate_email_verification_token(user.user_id)
        auth_manager.send_verification_email(user.email, user.full_name, verification_token)
        
        logger.info(f"Verification email resent to: {user.email}")
        
        return {
            "success": True,
            "message": "Verification email sent"
        }
    
    except Exception as e:
        logger.error(f"Resend verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LOGIN & LOGOUT ====================

@router.post("/login")
async def login(data: LoginRequest, request: Request):
    """
    Login user and return access + refresh tokens.
    
    Email must be verified before login is allowed.
    """
    try:
        # Call auth_manager.login()
        result = auth_manager.login(
            email=data.email,
            password=data.password
        )
        
        if "error" in result:
            auth_manager.log_audit_event(
                None, "login_failed",
                {"email": data.email, "reason": result["error"]},
                status="failure",
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request)
            )
            raise HTTPException(status_code=401, detail=result["error"])
        
        user_id = result["user_id"]
        
        # Generate refresh token
        refresh_token = auth_manager.generate_refresh_token(
            user_id,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )
        
        # Store refresh token in cache
        redis_manager.store_refresh_token(user_id, refresh_token)
        
        # Log audit event
        auth_manager.log_audit_event(
            user_id, "login_success",
            {"email": data.email},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )
        
        logger.info(f"User logged in: {data.email}")
        
        return {
            "success": True,
            "access_token": result["access_token"],
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": result["expires_in"],
            "user_id": user_id,
            "email": result["email"],
            "tier": result["tier"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.post("/logout")
async def logout(authorization: str = Header(None), request: Request = None):
    """
    Logout user and revoke tokens.
    
    Invalidates refresh tokens in Redis and marks as revoked in DB.
    """
    try:
        if not authorization or "Bearer " not in authorization:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        
        token = authorization.replace("Bearer ", "").strip()
        payload = auth_manager.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = payload.get("sub")
        
        # Revoke refresh tokens
        from models import RefreshToken, get_db_session
        session = get_db_session()
        
        tokens = session.query(RefreshToken).filter_by(
            user_id=user_id,
            revoked=False
        ).all()
        
        for token_record in tokens:
            token_record.revoked = True
        
        session.commit()
        session.close()
        
        # Also revoke in Redis
        redis_manager.revoke_user_refresh_tokens(user_id)
        
        # Log audit event
        auth_manager.log_audit_event(
            user_id, "logout",
            {},
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )
        
        logger.info(f"User logged out: {user_id}")
        
        return {"success": True, "message": "Logged out successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PASSWORD RESET ====================

@router.post("/request-password-reset")
async def request_password_reset(data: RequestPasswordResetRequest, request: Request):
    """
    Request password reset. Sends email with reset link.
    
    Does not reveal if email exists (security best practice).
    """
    try:
        result = auth_manager.request_password_reset(
            data.email,
            ip_address=get_client_ip(request)
        )
        
        return {
            "success": True,
            "message": "If email exists, password reset link has been sent"
        }
    
    except Exception as e:
        logger.error(f"Request password reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest, request: Request):
    """
    Reset password using token from email.
    
    Requires user_id, reset_token, and new_password.
    Invalidates all existing sessions after reset.
    """
    try:
        result = auth_manager.reset_password(
            data.user_id,
            data.reset_token,
            data.new_password,
            ip_address=get_client_ip(request)
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "message": "Password reset successfully. Please login with your new password."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== REFRESH TOKENS ====================

@router.post("/refresh-token")
async def refresh_token(data: RefreshTokenRequest, request: Request):
    """
    Refresh access token using refresh token.
    
    Implements token rotation for enhanced security.
    Old refresh token is revoked after new one is issued.
    """
    try:
        # step 1: check cache manager
        user_id = redis_manager.get_refresh_token_user(data.refresh_token)

        if not user_id:
            logger.debug(f"REFRESH Cache miss for token, checking database")
            
            # Query database for the refresh token
            from auth.models import RefreshToken, get_db_session
            session = get_db_session()
            try:
                dbtoken = session.query(RefreshToken).filter_by(
                    refresh_token=data.refresh_token,
                    revoked=False,
                ).first()
                
                if not dbtoken:
                    logger.warning(f"REFRESH Invalid or expired refresh token")
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid refresh token"
                    )
                
                user_id = dbtoken.user.user_id
                
                # Repopulate cache from database
                logger.debug(f"REFRESH Repopulating cache from database for user {user_id}")
                redis_manager.store_refresh_token(
                    user_id=user_id,
                    refresh_token=data.refresh_token,
                    ttl=int((dbtoken.expires_at - datetime.utcnow()).total_seconds())
                )
                
            finally:
                session.close()
        
        # Step 3: Continue with normal refresh flow
        result = auth_manager.refresh_access_token(
            user_id=user_id,
            refresh_token=data.refresh_token,
            ip_address=get_client_ip(request),
        )
        
        if "error" in result:
            raise HTTPException(status_code=401, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


# ==================== ROLE MANAGEMENT ====================

@router.get("/users/{user_id}/roles")
async def get_user_roles(user_id: str, authorization: str = Header(None)):
    """
    Get all roles for a user.
    """
    try:
        if not authorization or "Bearer " not in authorization:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        
        token = authorization.replace("Bearer ", "").strip()
        payload = auth_manager.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check if user can view this user's roles
        # (could be themselves or admin)
        requester_id = payload.get("sub")
        if requester_id != user_id and "admin" not in payload.get("roles", []):
            raise HTTPException(status_code=403, detail="Forbidden")
        
        roles = auth_manager.get_user_roles(user_id)
        
        return {
            "user_id": user_id,
            "roles": roles
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user roles error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/roles")
async def assign_role(user_id: str, data: RoleAssignmentRequest, 
                     authorization: str = Header(None)):
    """
    Assign role to user. Admin only.
    """
    try:
        if not authorization or "Bearer " not in authorization:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        
        token = authorization.replace("Bearer ", "").strip()
        payload = auth_manager.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check admin role
        if "admin" not in payload.get("roles", []):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        admin_id = payload.get("sub")
        result = auth_manager.assign_role(user_id, data.role_name, admin_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assign role error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}/roles/{role_name}")
async def revoke_role(user_id: str, role_name: str, 
                     authorization: str = Header(None)):
    """
    Revoke role from user. Admin only.
    """
    try:
        if not authorization or "Bearer " not in authorization:
            raise HTTPException(status_code=401, detail="Missing authorization token")
        
        token = authorization.replace("Bearer ", "").strip()
        payload = auth_manager.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check admin role
        if "admin" not in payload.get("roles", []):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        admin_id = payload.get("sub")
        result = auth_manager.revoke_role(user_id, role_name, admin_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revoke role error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
