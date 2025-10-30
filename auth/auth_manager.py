"""
Enhanced Authentication Manager with bcrypt hashing
"""

import jwt
import os
import secrets
import json
import bcrypt
from datetime import datetime, timedelta
from auth.models import (
    User, Role, RefreshToken, PasswordReset, QueryLog, AuditLog, get_db_session
)
from auth.cache_manager import redis_manager
from loguru import logger


class AuthManager:
    """Authentication manager"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET")
        self.jwt_expiry = 3600
        self.refresh_token_expiry = 7 * 24 * 3600
        self.email_verification_expiry = 24 * 3600
        self.password_reset_expiry = 1 * 3600
        self.max_refresh_rotations = 5
    
    # ==================== PASSWORD HASHING ====================
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            password_bytes = password.encode('utf-8')[:72]
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password_bytes, salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing error: {e}")
            raise
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            # Handle None case
            if not password_hash:
                logger.error("Password hash is None")
                return False
            
            # Convert password to bytes
            password_bytes = password.encode('utf-8')[:72]
            
            # Handle if password_hash is already bytes
            if isinstance(password_hash, bytes):
                hash_bytes = password_hash
            else:
                hash_bytes = password_hash.encode('utf-8')
            
            return bcrypt.checkpw(password_bytes, hash_bytes)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    # ==================== REGISTRATION ====================
    
    def register(self, email: str, password: str, full_name: str, company: str, tier: str = "free") -> dict:
        """Register new user with tier configuration"""
        from tier_config import validate_tier, get_daily_limit
        
        session = get_db_session()
        try:
            if not validate_tier(tier):
                return {"error": "Invalid tier"}
            
            existing = session.query(User).filter_by(email=email).first()
            if existing:
                return {"error": "Email already registered"}
            
            if len(password) < 8:
                return {"error": "Password must be at least 8 characters"}
            
            daily_limit = get_daily_limit(tier)
            password_hash = self._hash_password(password)
            
            user = User(
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                company=company,
                tier=tier,
                daily_query_limit=daily_limit,
                email_verified=True,
                email_verified_at=datetime.utcnow(),
                is_active=True,
                created_at=datetime.utcnow()
            )
            session.add(user)
            session.commit()
            
            logger.info(f"User registered: {email} with tier {tier}")
            
            return {
                "success": True,
                "user_id": user.user_id,
                "tier": tier,
                "daily_query_limit": daily_limit
            }
        
        except Exception as e:
            logger.error(f"Registration error: {e}")
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()

    # ==================== check query limit =========================

    def check_query_limit(self, user_id: str) -> bool:
        """
        Check if user has remaining queries for today according to their subscription tier.
        Returns True if user can make a query, False if daily limit exceeded.
        """
        from auth.models import get_db_session, User
        from datetime import datetime

        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return False

            today = datetime.utcnow().date()

            # Optional: if using a date field to reset queries_today daily
            if hasattr(user, "queries_today_date") and user.queries_today_date != today:
                user.queries_today = 0
                user.queries_today_date = today
                session.commit()

            # If user.queries_today is less than user's daily_query_limit, allow
            if user.queries_today < user.daily_query_limit:
                return True
            
            return False
        finally:
            session.close()
    
    # ==================== LOGIN ====================
    
    def login(self, email: str, password: str) -> dict:
        """Login user and return access token"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(email=email).first()
            
            if not user:
                return {"error": "Invalid email or password"}
            
            if not self._verify_password(password, user.password_hash):
                return {"error": "Invalid email or password"}
            
            if not user.is_active:
                return {"error": "Account is disabled"}
            
            if not user.email_verified:
                return {"error": "Email not verified"}
            
            user.last_login = datetime.utcnow()
            session.commit()
            
            access_token = jwt.encode(
                {
                    "sub": user.user_id,
                    "email": user.email,
                    "tier": user.tier,
                    "roles": [role.name for role in user.roles],
                    "exp": datetime.utcnow() + timedelta(seconds=self.jwt_expiry)
                },
                self.jwt_secret,
                algorithm="HS256"
            )
            
            logger.info(f"User logged in: {email}")
            
            return {
                "success": True,
                "access_token": access_token,
                "user_id": user.user_id,
                "email": user.email,
                "tier": user.tier,
                "expires_in": self.jwt_expiry
            }
        
        except Exception as e:
            logger.error(f"Login error: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== EMAIL VERIFICATION ====================
    
    def generate_email_verification_token(self, user_id: str) -> str:
        """Generate JWT-based email verification token"""
        token = jwt.encode(
            {
                "sub": user_id,
                "type": "email_verify",
                "exp": datetime.utcnow() + timedelta(seconds=self.email_verification_expiry)
            },
            self.jwt_secret,
            algorithm="HS256"
        )
        return token
    
    def verify_email(self, token: str) -> dict:
        """Verify email using token"""
        session = get_db_session()
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            if payload.get("type") != "email_verify":
                return {"error": "Invalid token type"}
            
            user_id = payload.get("sub")
            user = session.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                return {"error": "User not found"}
            
            user.email_verified = True
            user.email_verified_at = datetime.utcnow()
            session.commit()
            
            self.log_audit_event(user_id, "email_verified", {"email": user.email})
            logger.info(f"Email verified for user: {user.email}")
            
            return {"success": True, "email": user.email}
        
        except jwt.ExpiredSignatureError:
            return {"error": "Email verification link expired"}
        except jwt.InvalidTokenError:
            return {"error": "Invalid email verification token"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== PASSWORD RESET ====================
    
    def request_password_reset(self, email: str, ip_address: str = None) -> dict:
        """Generate password reset token"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(email=email).first()
            
            if not user:
                self.log_audit_event(None, "password_reset_requested", 
                                   {"email": email, "status": "user_not_found"}, 
                                   ip_address=ip_address)
                return {"success": True}
            
            reset_token = secrets.token_urlsafe(32)
            reset_token_hash = self._hash_password(reset_token)
            
            password_reset = PasswordReset(
                user_id=user.user_id,
                reset_token=reset_token_hash,
                expires_at=datetime.utcnow() + timedelta(seconds=self.password_reset_expiry)
            )
            
            session.add(password_reset)
            session.commit()
            
            logger.info(f"Password reset requested for: {email}")
            self.log_audit_event(user.user_id, "password_reset_requested", 
                               {"email": email}, ip_address=ip_address)
            
            return {"success": True}
        
        except Exception as e:
            logger.error(f"Error requesting password reset: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def reset_password(self, user_id: str, reset_token: str, new_password: str, 
                      ip_address: str = None) -> dict:
        """Reset password with token verification"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return {"error": "User not found"}
            
            reset_record = session.query(PasswordReset).filter_by(
                user_id=user_id,
                used=False
            ).filter(PasswordReset.expires_at > datetime.utcnow()).first()
            
            if not reset_record:
                self.log_audit_event(user_id, "password_reset_failed", 
                                   {"reason": "no_valid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid or expired reset token"}
            
            # Verify reset token using bcrypt
            if not self._verify_password(reset_token, reset_record.reset_token):
                self.log_audit_event(user_id, "password_reset_failed", 
                                   {"reason": "invalid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid reset token"}
            
            if len(new_password) < 8:
                return {"error": "Password must be at least 8 characters"}
            
            user.password_hash = self._hash_password(new_password)
            reset_record.used = True
            reset_record.used_at = datetime.utcnow()
            
            session.commit()
            self._revoke_all_refresh_tokens(user_id, session)
            
            self.log_audit_event(user_id, "password_reset_success", 
                               {"email": user.email}, ip_address=ip_address)
            
            logger.info(f"Password reset successful for: {user.email}")
            return {"success": True, "message": "Password reset successful"}
        
        except Exception as e:
            logger.error(f"Error resetting password: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== REFRESH TOKEN ROTATION ====================
    
    def generate_refresh_token(self, user_id: str, ip_address: str = None, 
                              user_agent: str = None) -> str:
        """Generate a long-lived refresh token"""
        refresh_token = secrets.token_urlsafe(64)
        
        session = get_db_session()
        try:
            token_record = RefreshToken(
                user_id=user_id,
                refresh_token=self._hash_password(refresh_token),
                expires_at=datetime.utcnow() + timedelta(seconds=self.refresh_token_expiry),
                ip_address=ip_address,
                user_agent=user_agent,
                last_used_at=datetime.utcnow()
            )
            
            session.add(token_record)
            session.commit()
            
            redis_manager.store_refresh_token(user_id, refresh_token, 
                                             self.refresh_token_expiry)
            
            return refresh_token
        finally:
            session.close()
    
    def refresh_access_token(self, user_id: str, refresh_token: str, 
                            ip_address: str = None) -> dict:
        """Issue new access token from refresh token with rotation"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user or not user.is_active or not user.email_verified:
                return {"error": "User not found or account disabled"}
            
            token_records = session.query(RefreshToken).filter_by(
                user_id=user_id,
                revoked=False
            ).filter(RefreshToken.expires_at > datetime.utcnow()).all()
            
            valid_token_record = None
            for token_record in token_records:
                if self._verify_password(refresh_token, token_record.refresh_token):
                    valid_token_record = token_record
                    break
            
            if not valid_token_record:
                self.log_audit_event(user_id, "token_refresh_failed", 
                                   {"reason": "invalid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid refresh token"}
            
            if valid_token_record.rotation_count >= self.max_refresh_rotations:
                valid_token_record.revoked = True
                valid_token_record.revoked_at = datetime.utcnow()
                session.commit()
                
                self.log_audit_event(user_id, "token_rotation_limit_exceeded", 
                                   {"token_id": valid_token_record.token_id}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Refresh token rotation limit exceeded"}
            
            access_token = jwt.encode(
                {
                    "sub": user.user_id,
                    "email": user.email,
                    "tier": user.tier,
                    "roles": [role.name for role in user.roles],
                    "exp": datetime.utcnow() + timedelta(seconds=self.jwt_expiry)
                },
                self.jwt_secret,
                algorithm="HS256"
            )
            
            new_refresh_token = secrets.token_urlsafe(64)
            
            new_token_record = RefreshToken(
                user_id=user_id,
                refresh_token=self._hash_password(new_refresh_token),
                access_token=access_token,
                expires_at=datetime.utcnow() + timedelta(seconds=self.refresh_token_expiry),
                ip_address=ip_address,
                last_used_at=datetime.utcnow()
            )
            
            valid_token_record.revoked = True
            valid_token_record.revoked_at = datetime.utcnow()
            valid_token_record.rotation_count += 1
            
            session.add(new_token_record)
            session.commit()
            
            self.log_audit_event(user_id, "token_refreshed", 
                               {"rotation_count": valid_token_record.rotation_count}, 
                               ip_address=ip_address)
            
            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": self.jwt_expiry
            }
        
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== RBAC ====================
    
    def assign_role(self, user_id: str, role_name: str, admin_user_id: str = None) -> dict:
        """Assign role to user"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return {"error": "User not found"}
            
            role = session.query(Role).filter_by(name=role_name).first()
            if not role:
                return {"error": f"Role '{role_name}' not found"}
            
            if role in user.roles:
                return {"error": "User already has this role"}
            
            user.roles.append(role)
            session.commit()
            
            self.log_audit_event(admin_user_id, "role_assigned", 
                               {"user_id": user_id, "role": role_name})
            
            logger.info(f"Role '{role_name}' assigned to user {user_id}")
            return {"success": True}
        
        except Exception as e:
            return {"error": str(e)}
        finally:
            session.close()
    
    def revoke_role(self, user_id: str, role_name: str, admin_user_id: str = None) -> dict:
        """Revoke role from user"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return {"error": "User not found"}
            
            role = session.query(Role).filter_by(name=role_name).first()
            if not role:
                return {"error": f"Role '{role_name}' not found"}
            
            if role not in user.roles:
                return {"error": "User does not have this role"}
            
            user.roles.remove(role)
            session.commit()
            
            self.log_audit_event(admin_user_id, "role_revoked", 
                               {"user_id": user_id, "role": role_name})
            
            logger.info(f"Role '{role_name}' revoked from user {user_id}")
            return {"success": True}
        
        except Exception as e:
            return {"error": str(e)}
        finally:
            session.close()
    
    def get_user_roles(self, user_id: str) -> list:
        """Get all roles for a user"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            return [role.name for role in user.roles] if user else []
        finally:
            session.close()
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                return False
            
            for role in user.roles:
                permissions = json.loads(role.permissions or "[]")
                if "*" in permissions or permission in permissions:
                    return True
            
            return False
        finally:
            session.close()
    
    # ==================== AUDIT LOGGING ====================
    
    def log_audit_event(self, user_id: str, event_type: str, event_details: dict = None, 
                       status: str = "success", ip_address: str = None, user_agent: str = None):
        """Log security audit event"""
        session = get_db_session()
        try:
            audit_log = AuditLog(
                user_id=user_id,
                event_type=event_type,
                event_details=json.dumps(event_details or {}),
                ip_address=ip_address,
                user_agent=user_agent,
                status=status
            )
            
            session.add(audit_log)
            session.commit()
            
            logger.info(f"Audit: {event_type} for user {user_id} - {status}")
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
        finally:
            session.close()
    
    def _revoke_all_refresh_tokens(self, user_id: str, session):
        """Revoke all refresh tokens for a user"""
        tokens = session.query(RefreshToken).filter_by(
            user_id=user_id,
            revoked=False
        ).all()
        
        for token in tokens:
            token.revoked = True
            token.revoked_at = datetime.utcnow()
        
        session.commit()
        redis_manager.revoke_user_refresh_tokens(user_id)
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


# Global instance
auth_manager = AuthManager()
