"""
Enhanced Authentication Manager with bcrypt hashing - WITH DETAILED LOGGING
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
        logger.info("AuthManager initialized")
    
    # ==================== PASSWORD HASHING ====================
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            logger.debug(f"[HASH] Starting password hash - password length: {len(password)}")
            
            password_bytes = password.encode('utf-8')[:72]
            logger.debug(f"[HASH] Password encoded to bytes, length: {len(password_bytes)}")
            
            salt = bcrypt.gensalt(rounds=12)
            logger.debug(f"[HASH] Salt generated")
            
            hashed = bcrypt.hashpw(password_bytes, salt)
            logger.debug(f"[HASH] Password hashed successfully, result type: {type(hashed)}")
            
            result = hashed.decode('utf-8')
            logger.debug(f"[HASH] Hash decoded to UTF-8, length: {len(result)}")
            
            return result
        except Exception as e:
            logger.error(f"[HASH] Password hashing error: {type(e).__name__}: {e}")
            raise
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            logger.debug(f"[VERIFY] Starting password verification")
            logger.debug(f"[VERIFY] Input password type: {type(password)}, length: {len(password) if password else 'None'}")
            logger.debug(f"[VERIFY] Input hash type: {type(password_hash)}, value_preview: {str(password_hash)[:30] if password_hash else 'None'}")
            
            # Handle None case
            if not password_hash:
                logger.error("[VERIFY] Password hash is None or empty")
                return False
            
            logger.debug(f"[VERIFY] Password hash is valid, not None")
            
            # Convert password to bytes
            password_bytes = password.encode('utf-8')[:72]
            logger.debug(f"[VERIFY] Password encoded to bytes: {type(password_bytes)}")
            
            # Handle if password_hash is already bytes
            if isinstance(password_hash, bytes):
                logger.debug(f"[VERIFY] Hash is bytes type, using directly")
                hash_bytes = password_hash
            else:
                logger.debug(f"[VERIFY] Hash is str type, encoding to bytes")
                hash_bytes = password_hash.encode('utf-8')
            
            logger.debug(f"[VERIFY] Hash converted to bytes: {type(hash_bytes)}")
            logger.debug(f"[VERIFY] Calling bcrypt.checkpw() - password_bytes: {len(password_bytes)} bytes, hash_bytes: {len(hash_bytes)} bytes")
            
            result = bcrypt.checkpw(password_bytes, hash_bytes)
            logger.debug(f"[VERIFY] bcrypt.checkpw() result: {result}")
            
            return result
            
        except TypeError as e:
            logger.error(f"[VERIFY] TypeError in password verification: {e}")
            logger.error(f"[VERIFY] password type: {type(password)}")
            logger.error(f"[VERIFY] password_hash type: {type(password_hash)}")
            logger.error(f"[VERIFY] password_hash value: {repr(password_hash)[:100]}")
            return False
        except Exception as e:
            logger.error(f"[VERIFY] Exception in password verification: {type(e).__name__}: {e}")
            logger.error(f"[VERIFY] password_hash value: {repr(password_hash)[:100]}")
            return False
    
    # ==================== REGISTRATION ====================
    
    def register(self, email: str, password: str, full_name: str, company: str, tier: str = "free") -> dict:
        """Register new user with tier configuration"""
        from tier_config import validate_tier, get_daily_limit
        
        session = get_db_session()
        try:
            logger.info(f"[REGISTER] Starting registration for email: {email}")
            
            if not validate_tier(tier):
                logger.warning(f"[REGISTER] Invalid tier: {tier}")
                return {"error": "Invalid tier"}
            
            existing = session.query(User).filter_by(email=email).first()
            if existing:
                logger.warning(f"[REGISTER] Email already exists: {email}")
                return {"error": "Email already registered"}
            
            if len(password) < 8:
                logger.warning(f"[REGISTER] Password too short for email: {email}")
                return {"error": "Password must be at least 8 characters"}
            
            daily_limit = get_daily_limit(tier)
            logger.debug(f"[REGISTER] Daily limit for tier {tier}: {daily_limit}")
            
            password_hash = self._hash_password(password)
            logger.debug(f"[REGISTER] Password hashed successfully")
            
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
            
            logger.info(f"[REGISTER] User registered successfully: {email} with tier {tier}")
            
            return {
                "success": True,
                "user_id": user.user_id,
                "tier": tier,
                "daily_query_limit": daily_limit
            }
        
        except Exception as e:
            logger.error(f"[REGISTER] Registration error: {type(e).__name__}: {e}")
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
            logger.debug(f"[QUERY_LIMIT] Checking query limit for user: {user_id}")
            
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.warning(f"[QUERY_LIMIT] User not found: {user_id}")
                return False

            today = datetime.utcnow().date()
            logger.debug(f"[QUERY_LIMIT] Today's date: {today}")

            # Optional: if using a date field to reset queries_today daily
            if hasattr(user, "queries_today_date") and user.queries_today_date != today:
                logger.debug(f"[QUERY_LIMIT] Resetting daily queries for {user_id}")
                user.queries_today = 0
                user.queries_today_date = today
                session.commit()

            logger.debug(f"[QUERY_LIMIT] User queries today: {user.queries_today}, limit: {user.daily_query_limit}")

            # If user.queries_today is less than user's daily_query_limit, allow
            if user.queries_today < user.daily_query_limit:
                logger.debug(f"[QUERY_LIMIT] Query allowed for {user_id}")
                return True
            
            logger.warning(f"[QUERY_LIMIT] Query limit exceeded for {user_id}")
            return False
        except Exception as e:
            logger.error(f"[QUERY_LIMIT] Error checking query limit: {type(e).__name__}: {e}")
            return False
        finally:
            session.close()
    
    # ==================== LOGIN ====================
    
    def login(self, email: str, password: str) -> dict:
        """Login user and return access token"""
        session = get_db_session()
        try:
            logger.info(f"[LOGIN] Starting login for email: {email}")
            
            user = session.query(User).filter_by(email=email).first()
            
            if not user:
                logger.warning(f"[LOGIN] User not found: {email}")
                return {"error": "Invalid email or password"}
            
            logger.debug(f"[LOGIN] User found: {email}, user_id: {user.user_id}")
            logger.debug(f"[LOGIN] User password_hash type from DB: {type(user.password_hash)}")
            logger.debug(f"[LOGIN] User password_hash preview: {str(user.password_hash)[:50] if user.password_hash else 'None'}")
            logger.debug(f"[LOGIN] User is_active: {user.is_active}")
            logger.debug(f"[LOGIN] User email_verified: {user.email_verified}")
            
            logger.debug(f"[LOGIN] Calling _verify_password for user: {email}")
            if not self._verify_password(password, user.password_hash):
                logger.warning(f"[LOGIN] Password verification failed for: {email}")
                return {"error": "Invalid email or password"}
            
            logger.debug(f"[LOGIN] Password verification succeeded for: {email}")
            
            if not user.is_active:
                logger.warning(f"[LOGIN] Account is disabled for: {email}")
                return {"error": "Account is disabled"}
            
            if not user.email_verified:
                logger.warning(f"[LOGIN] Email not verified for: {email}")
                return {"error": "Email not verified"}
            
            user.last_login = datetime.utcnow()
            session.commit()
            logger.debug(f"[LOGIN] Updated last_login for: {email}")
            
            logger.debug(f"[LOGIN] Encoding JWT token for user: {user.user_id}")
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
            logger.debug(f"[LOGIN] JWT token encoded successfully")
            
            logger.info(f"[LOGIN] User logged in successfully: {email}")
            
            return {
                "success": True,
                "access_token": access_token,
                "user_id": user.user_id,
                "email": user.email,
                "tier": user.tier,
                "expires_in": self.jwt_expiry
            }
        
        except Exception as e:
            logger.error(f"[LOGIN] Login error: {type(e).__name__}: {e}")
            logger.error(f"[LOGIN] Full exception: {str(e)}")
            import traceback
            logger.error(f"[LOGIN] Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== EMAIL VERIFICATION ====================
    
    def generate_email_verification_token(self, user_id: str) -> str:
        """Generate JWT-based email verification token"""
        logger.debug(f"[EMAIL_VERIFY] Generating email verification token for user: {user_id}")
        token = jwt.encode(
            {
                "sub": user_id,
                "type": "email_verify",
                "exp": datetime.utcnow() + timedelta(seconds=self.email_verification_expiry)
            },
            self.jwt_secret,
            algorithm="HS256"
        )
        logger.debug(f"[EMAIL_VERIFY] Token generated successfully")
        return token
    
    def verify_email(self, token: str) -> dict:
        """Verify email using token"""
        session = get_db_session()
        try:
            logger.debug(f"[EMAIL_VERIFY] Starting email verification with token")
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            logger.debug(f"[EMAIL_VERIFY] Token decoded successfully")
            
            if payload.get("type") != "email_verify":
                logger.warning(f"[EMAIL_VERIFY] Invalid token type: {payload.get('type')}")
                return {"error": "Invalid token type"}
            
            user_id = payload.get("sub")
            logger.debug(f"[EMAIL_VERIFY] User ID from token: {user_id}")
            
            user = session.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                logger.warning(f"[EMAIL_VERIFY] User not found: {user_id}")
                return {"error": "User not found"}
            
            user.email_verified = True
            user.email_verified_at = datetime.utcnow()
            session.commit()
            
            self.log_audit_event(user_id, "email_verified", {"email": user.email})
            logger.info(f"[EMAIL_VERIFY] Email verified for user: {user.email}")
            
            return {"success": True, "email": user.email}
        
        except jwt.ExpiredSignatureError:
            logger.warning(f"[EMAIL_VERIFY] Email verification link expired")
            return {"error": "Email verification link expired"}
        except jwt.InvalidTokenError as e:
            logger.warning(f"[EMAIL_VERIFY] Invalid email verification token: {e}")
            return {"error": "Invalid email verification token"}
        except Exception as e:
            logger.error(f"[EMAIL_VERIFY] Error: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== PASSWORD RESET ====================
    
    def request_password_reset(self, email: str, ip_address: str = None) -> dict:
        """Generate password reset token"""
        session = get_db_session()
        try:
            logger.info(f"[PASSWORD_RESET] Password reset requested for email: {email}")
            
            user = session.query(User).filter_by(email=email).first()
            
            if not user:
                logger.warning(f"[PASSWORD_RESET] User not found: {email}")
                self.log_audit_event(None, "password_reset_requested", 
                                   {"email": email, "status": "user_not_found"}, 
                                   ip_address=ip_address)
                return {"success": True}
            
            reset_token = secrets.token_urlsafe(32)
            logger.debug(f"[PASSWORD_RESET] Reset token generated for user: {user.user_id}")
            
            reset_token_hash = self._hash_password(reset_token)
            logger.debug(f"[PASSWORD_RESET] Reset token hashed")
            
            password_reset = PasswordReset(
                user_id=user.user_id,
                reset_token=reset_token_hash,
                expires_at=datetime.utcnow() + timedelta(seconds=self.password_reset_expiry)
            )
            
            session.add(password_reset)
            session.commit()
            
            logger.info(f"[PASSWORD_RESET] Password reset requested successfully for: {email}")
            self.log_audit_event(user.user_id, "password_reset_requested", 
                               {"email": email}, ip_address=ip_address)
            
            return {"success": True}
        
        except Exception as e:
            logger.error(f"[PASSWORD_RESET] Error requesting password reset: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def reset_password(self, user_id: str, reset_token: str, new_password: str, 
                      ip_address: str = None) -> dict:
        """Reset password with token verification"""
        session = get_db_session()
        try:
            logger.info(f"[RESET_PWD] Password reset initiated for user: {user_id}")
            
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.warning(f"[RESET_PWD] User not found: {user_id}")
                return {"error": "User not found"}
            
            reset_record = session.query(PasswordReset).filter_by(
                user_id=user_id,
                used=False
            ).filter(PasswordReset.expires_at > datetime.utcnow()).first()
            
            if not reset_record:
                logger.warning(f"[RESET_PWD] No valid reset token found for user: {user_id}")
                self.log_audit_event(user_id, "password_reset_failed", 
                                   {"reason": "no_valid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid or expired reset token"}
            
            # Verify reset token using bcrypt
            if not self._verify_password(reset_token, reset_record.reset_token):
                logger.warning(f"[RESET_PWD] Reset token verification failed for user: {user_id}")
                self.log_audit_event(user_id, "password_reset_failed", 
                                   {"reason": "invalid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid reset token"}
            
            if len(new_password) < 8:
                logger.warning(f"[RESET_PWD] New password too short for user: {user_id}")
                return {"error": "Password must be at least 8 characters"}
            
            user.password_hash = self._hash_password(new_password)
            reset_record.used = True
            reset_record.used_at = datetime.utcnow()
            
            session.commit()
            self._revoke_all_refresh_tokens(user_id, session)
            
            self.log_audit_event(user_id, "password_reset_success", 
                               {"email": user.email}, ip_address=ip_address)
            
            logger.info(f"[RESET_PWD] Password reset successful for: {user.email}")
            return {"success": True, "message": "Password reset successful"}
        
        except Exception as e:
            logger.error(f"[RESET_PWD] Error resetting password: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== REFRESH TOKEN ROTATION ====================
    
    def generate_refresh_token(self, user_id: str, ip_address: str = None, 
                              user_agent: str = None) -> str:
        """Generate a long-lived refresh token"""
        logger.debug(f"[REFRESH_TOKEN] Generating refresh token for user: {user_id}")
        
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
            
            logger.debug(f"[REFRESH_TOKEN] Refresh token stored in DB for user: {user_id}")
            
            redis_manager.store_refresh_token(user_id, refresh_token, 
                                             self.refresh_token_expiry)
            
            logger.debug(f"[REFRESH_TOKEN] Refresh token stored in Redis for user: {user_id}")
            
            return refresh_token
        finally:
            session.close()
    
    def refresh_access_token(self, user_id: str, refresh_token: str, 
                            ip_address: str = None) -> dict:
        """Issue new access token from refresh token with rotation"""
        logger.debug(f"[REFRESH_ACCESS] Refreshing access token for user: {user_id}")
        
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user or not user.is_active or not user.email_verified:
                logger.warning(f"[REFRESH_ACCESS] User not found or account disabled: {user_id}")
                return {"error": "User not found or account disabled"}
            
            token_records = session.query(RefreshToken).filter_by(
                user_id=user_id,
                revoked=False
            ).filter(RefreshToken.expires_at > datetime.utcnow()).all()
            
            logger.debug(f"[REFRESH_ACCESS] Found {len(token_records)} valid refresh tokens for user: {user_id}")
            
            valid_token_record = None
            for token_record in token_records:
                if self._verify_password(refresh_token, token_record.refresh_token):
                    valid_token_record = token_record
                    logger.debug(f"[REFRESH_ACCESS] Valid refresh token found for user: {user_id}")
                    break
            
            if not valid_token_record:
                logger.warning(f"[REFRESH_ACCESS] No valid refresh token found for user: {user_id}")
                self.log_audit_event(user_id, "token_refresh_failed", 
                                   {"reason": "invalid_token"}, 
                                   status="failure", ip_address=ip_address)
                return {"error": "Invalid refresh token"}
            
            if valid_token_record.rotation_count >= self.max_refresh_rotations:
                logger.warning(f"[REFRESH_ACCESS] Rotation limit exceeded for user: {user_id}")
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
            
            logger.debug(f"[REFRESH_ACCESS] New access token generated")
            
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
            
            logger.info(f"[REFRESH_ACCESS] Access token refreshed successfully for user: {user_id}")
            
            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": self.jwt_expiry
            }
        
        except Exception as e:
            logger.error(f"[REFRESH_ACCESS] Error refreshing token: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    # ==================== RBAC ====================
    
    def assign_role(self, user_id: str, role_name: str, admin_user_id: str = None) -> dict:
        """Assign role to user"""
        logger.info(f"[ROLE] Assigning role '{role_name}' to user: {user_id}")
        
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.warning(f"[ROLE] User not found: {user_id}")
                return {"error": "User not found"}
            
            role = session.query(Role).filter_by(name=role_name).first()
            if not role:
                logger.warning(f"[ROLE] Role not found: {role_name}")
                return {"error": f"Role '{role_name}' not found"}
            
            if role in user.roles:
                logger.warning(f"[ROLE] User {user_id} already has role: {role_name}")
                return {"error": "User already has this role"}
            
            user.roles.append(role)
            session.commit()
            
            self.log_audit_event(admin_user_id, "role_assigned", 
                               {"user_id": user_id, "role": role_name})
            
            logger.info(f"[ROLE] Role '{role_name}' assigned to user {user_id}")
            return {"success": True}
        
        except Exception as e:
            logger.error(f"[ROLE] Error assigning role: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def revoke_role(self, user_id: str, role_name: str, admin_user_id: str = None) -> dict:
        """Revoke role from user"""
        logger.info(f"[ROLE] Revoking role '{role_name}' from user: {user_id}")
        
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.warning(f"[ROLE] User not found: {user_id}")
                return {"error": "User not found"}
            
            role = session.query(Role).filter_by(name=role_name).first()
            if not role:
                logger.warning(f"[ROLE] Role not found: {role_name}")
                return {"error": f"Role '{role_name}' not found"}
            
            if role not in user.roles:
                logger.warning(f"[ROLE] User {user_id} does not have role: {role_name}")
                return {"error": "User does not have this role"}
            
            user.roles.remove(role)
            session.commit()
            
            self.log_audit_event(admin_user_id, "role_revoked", 
                               {"user_id": user_id, "role": role_name})
            
            logger.info(f"[ROLE] Role '{role_name}' revoked from user {user_id}")
            return {"success": True}
        
        except Exception as e:
            logger.error(f"[ROLE] Error revoking role: {type(e).__name__}: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def get_user_roles(self, user_id: str) -> list:
        """Get all roles for a user"""
        logger.debug(f"[ROLE] Getting roles for user: {user_id}")
        
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            roles = [role.name for role in user.roles] if user else []
            logger.debug(f"[ROLE] User {user_id} has roles: {roles}")
            return roles
        finally:
            session.close()
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission"""
        logger.debug(f"[PERMISSION] Checking if user {user_id} has permission: {permission}")
        
        session = get_db_session()
        try:
            user = session.query(User).filter_by(user_id=user_id).first()
            if not user:
                logger.warning(f"[PERMISSION] User not found: {user_id}")
                return False
            
            for role in user.roles:
                permissions = json.loads(role.permissions or "[]")
                if "*" in permissions or permission in permissions:
                    logger.debug(f"[PERMISSION] Permission '{permission}' granted for user {user_id}")
                    return True
            
            logger.warning(f"[PERMISSION] Permission '{permission}' denied for user {user_id}")
            return False
        finally:
            session.close()
    
    # ==================== AUDIT LOGGING ====================
    
    def log_audit_event(self, user_id: str, event_type: str, event_details: dict = None, 
                       status: str = "success", ip_address: str = None, user_agent: str = None):
        """Log security audit event"""
        logger.debug(f"[AUDIT] Logging audit event - user: {user_id}, event: {event_type}, status: {status}")
        
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
            
            logger.info(f"[AUDIT] {event_type} for user {user_id} - {status}")
        except Exception as e:
            logger.error(f"[AUDIT] Error logging audit event: {type(e).__name__}: {e}")
        finally:
            session.close()
    
    def _revoke_all_refresh_tokens(self, user_id: str, session):
        """Revoke all refresh tokens for a user"""
        logger.debug(f"[REVOKE] Revoking all refresh tokens for user: {user_id}")
        
        tokens = session.query(RefreshToken).filter_by(
            user_id=user_id,
            revoked=False
        ).all()
        
        logger.debug(f"[REVOKE] Found {len(tokens)} active refresh tokens to revoke")
        
        for token in tokens:
            token.revoked = True
            token.revoked_at = datetime.utcnow()
        
        session.commit()
        redis_manager.revoke_user_refresh_tokens(user_id)
        
        logger.info(f"[REVOKE] All refresh tokens revoked for user: {user_id}")
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token and return payload"""
        logger.debug(f"[TOKEN_VERIFY] Verifying JWT token")
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            logger.debug(f"[TOKEN_VERIFY] Token verified successfully for user: {payload.get('sub')}")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning(f"[TOKEN_VERIFY] Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"[TOKEN_VERIFY] Invalid token: {e}")
            return None



# Global instance
auth_manager = AuthManager()
