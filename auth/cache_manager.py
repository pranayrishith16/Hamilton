"""
In-memory cache for sessions and rate limiting (MVP version).
Replaces Redis with Python dictionaries for development/testing.
WARNING: This is single-instance only and data is lost on restart.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, List
import threading

class InMemoryCacheManager:
    """In-memory cache manager using Python dicts"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.sessions = {}  # session:user_id -> token
        self.refresh_tokens = {}  # refresh_token:token -> user_id
        self.user_refresh_tokens = {}  # user_refresh_tokens:user_id -> set of tokens
        self.blacklist = {}  # blacklist:token -> expiry time
        self.rate_limits = {}  # queries:user_id:day -> (count, expiry)
        self.verification_tokens = {}  # verify_email:email -> (token, expiry)
        self.reset_tokens = {}  # reset_token:user_id -> (token, expiry)
        self.security_events = {}  # security_events:user_id -> list of events
        self.permissions_cache = {}  # permissions:user_id -> (permissions, expiry)
        self.lock = threading.Lock()
        print("✓ In-memory cache initialized (MVP mode)")
    
    def _is_expired(self, expiry_time):
        """Check if timestamp has expired"""
        if expiry_time is None:
            return False
        return datetime.utcnow() > expiry_time
    
    def _cleanup_expired(self):
        """Clean up expired entries (called periodically)"""
        now = datetime.utcnow()
        
        # Clean blacklist
        self.blacklist = {k: v for k, v in self.blacklist.items() if v > now}
        
        # Clean rate limits
        self.rate_limits = {k: v for k, v in self.rate_limits.items() if v[1] > now}
        
        # Clean verification tokens
        self.verification_tokens = {k: v for k, v in self.verification_tokens.items() if v[1] > now}
        
        # Clean reset tokens
        self.reset_tokens = {k: v for k, v in self.reset_tokens.items() if v[1] > now}
        
        # Clean permissions cache
        self.permissions_cache = {k: v for k, v in self.permissions_cache.items() if v[1] > now}
    
    # ==================== SESSION MANAGEMENT ====================
    
    def store_session(self, user_id: str, token: str, ttl: int = 3600):
        """Store user session"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.sessions[f"session:{user_id}"] = (token, expiry)
    
    def get_session(self, user_id: str) -> Optional[str]:
        """Get user session"""
        with self.lock:
            key = f"session:{user_id}"
            if key in self.sessions:
                token, expiry = self.sessions[key]
                if not self._is_expired(expiry):
                    return token
                del self.sessions[key]
            return None
    
    def delete_session(self, user_id: str):
        """Delete session (logout)"""
        with self.lock:
            key = f"session:{user_id}"
            if key in self.sessions:
                del self.sessions[key]
    
    # ==================== REFRESH TOKEN MANAGEMENT ====================
    
    def store_refresh_token(self, user_id: str, refresh_token: str, ttl: int = 604800):
        """Store refresh token with user_id mapping"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.refresh_tokens[f"refresh_token:{refresh_token}"] = (user_id, expiry)
            
            if f"user_refresh_tokens:{user_id}" not in self.user_refresh_tokens:
                self.user_refresh_tokens[f"user_refresh_tokens:{user_id}"] = set()
            self.user_refresh_tokens[f"user_refresh_tokens:{user_id}"].add(refresh_token)
    
    def get_refresh_token_user(self, refresh_token: str) -> Optional[str]:
        """Get user_id associated with refresh token"""
        with self.lock:
            key = f"refresh_token:{refresh_token}"
            if key in self.refresh_tokens:
                user_id, expiry = self.refresh_tokens[key]
                if not self._is_expired(expiry):
                    return user_id
                del self.refresh_tokens[key]
            return None
    
    def revoke_refresh_token(self, refresh_token: str):
        """Revoke single refresh token"""
        with self.lock:
            key = f"refresh_token:{refresh_token}"
            if key in self.refresh_tokens:
                user_id, _ = self.refresh_tokens[key]
                del self.refresh_tokens[key]
                
                user_key = f"user_refresh_tokens:{user_id}"
                if user_key in self.user_refresh_tokens:
                    self.user_refresh_tokens[user_key].discard(refresh_token)
    
    def revoke_user_refresh_tokens(self, user_id: str):
        """Revoke all refresh tokens for a user"""
        with self.lock:
            user_key = f"user_refresh_tokens:{user_id}"
            if user_key in self.user_refresh_tokens:
                tokens = list(self.user_refresh_tokens[user_key])
                for token in tokens:
                    if f"refresh_token:{token}" in self.refresh_tokens:
                        del self.refresh_tokens[f"refresh_token:{token}"]
                del self.user_refresh_tokens[user_key]
    
    # ==================== TOKEN BLACKLIST ====================
    
    def blacklist_token(self, token: str, ttl: int = 3600):
        """Blacklist access token"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.blacklist[f"blacklist:{token}"] = expiry
    
    def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        with self.lock:
            key = f"blacklist:{token}"
            if key in self.blacklist:
                if not self._is_expired(self.blacklist[key]):
                    return True
                del self.blacklist[key]
            return False
    
    # ==================== RATE LIMITING ====================
    
    def check_rate_limit(self, user_id: str, limit: int) -> bool:
        """Check if user exceeded daily rate limit"""
        with self.lock:
            self._cleanup_expired()
            key = f"queries:{user_id}:day"
            
            if key not in self.rate_limits:
                expiry = datetime.utcnow() + timedelta(days=1)
                self.rate_limits[key] = (1, expiry)
                return True
            
            count, expiry = self.rate_limits[key]
            if self._is_expired(expiry):
                expiry = datetime.utcnow() + timedelta(days=1)
                self.rate_limits[key] = (1, expiry)
                return True
            
            if count >= limit:
                return False
            
            self.rate_limits[key] = (count + 1, expiry)
            return True
    
    def get_query_count(self, user_id: str) -> int:
        """Get current query count for today"""
        with self.lock:
            key = f"queries:{user_id}:day"
            if key in self.rate_limits:
                count, expiry = self.rate_limits[key]
                if not self._is_expired(expiry):
                    return count
            return 0
    
    def reset_daily_limit(self, user_id: str):
        """Reset daily query limit"""
        with self.lock:
            key = f"queries:{user_id}:day"
            if key in self.rate_limits:
                del self.rate_limits[key]
    
    # ==================== EMAIL VERIFICATION ====================
    
    def store_verification_token(self, email: str, token: str, ttl: int = 86400):
        """Store email verification token"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.verification_tokens[f"verify_email:{email}"] = (token, expiry)
    
    def get_verification_token(self, email: str) -> Optional[str]:
        """Get verification token for email"""
        with self.lock:
            key = f"verify_email:{email}"
            if key in self.verification_tokens:
                token, expiry = self.verification_tokens[key]
                if not self._is_expired(expiry):
                    return token
                del self.verification_tokens[key]
            return None
    
    def delete_verification_token(self, email: str):
        """Delete verification token"""
        with self.lock:
            key = f"verify_email:{email}"
            if key in self.verification_tokens:
                del self.verification_tokens[key]
    
    # ==================== PASSWORD RESET ====================
    
    def store_reset_token(self, user_id: str, reset_token: str, ttl: int = 3600):
        """Store password reset token"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.reset_tokens[f"reset_token:{user_id}"] = (reset_token, expiry)
    
    def get_reset_token(self, user_id: str) -> Optional[str]:
        """Get reset token for user"""
        with self.lock:
            key = f"reset_token:{user_id}"
            if key in self.reset_tokens:
                token, expiry = self.reset_tokens[key]
                if not self._is_expired(expiry):
                    return token
                del self.reset_tokens[key]
            return None
    
    def delete_reset_token(self, user_id: str):
        """Delete reset token"""
        with self.lock:
            key = f"reset_token:{user_id}"
            if key in self.reset_tokens:
                del self.reset_tokens[key]
    
    # ==================== AUDIT LOG CACHE ====================
    
    def log_security_event(self, user_id: str, event_type: str, details: dict = None, ttl: int = 2592000):
        """Cache security events"""
        with self.lock:
            event_data = {
                "event_type": event_type,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            key = f"security_events:{user_id}"
            if key not in self.security_events:
                self.security_events[key] = []
            
            self.security_events[key].insert(0, event_data)
            self.security_events[key] = self.security_events[key][:1000]  # Keep last 1000
    
    def get_security_events(self, user_id: str, limit: int = 50) -> List[dict]:
        """Get recent security events for user"""
        with self.lock:
            key = f"security_events:{user_id}"
            if key in self.security_events:
                return self.security_events[key][:limit]
            return []
    
    # ==================== PERMISSION CACHE ====================
    
    def cache_user_permissions(self, user_id: str, permissions: list, ttl: int = 3600):
        """Cache user permissions"""
        with self.lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self.permissions_cache[f"permissions:{user_id}"] = (permissions, expiry)
    
    def get_cached_permissions(self, user_id: str) -> Optional[List[str]]:
        """Get cached user permissions"""
        with self.lock:
            key = f"permissions:{user_id}"
            if key in self.permissions_cache:
                permissions, expiry = self.permissions_cache[key]
                if not self._is_expired(expiry):
                    return permissions
                del self.permissions_cache[key]
            return None
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate user permission/role cache"""
        with self.lock:
            perm_key = f"permissions:{user_id}"
            if perm_key in self.permissions_cache:
                del self.permissions_cache[perm_key]

# Global instance
redis_manager = InMemoryCacheManager()
