"""
API Key authentication system for programmatic access.

Features:
  - Generate cryptographically secure API keys
  - Hash keys using bcrypt (never store plain text)
  - Validate keys from Authorization header
  - Track key usage and rate limits
  - Revoke/expire keys
  - Associate keys with users, scopes, and expiration dates

Classes:
  - APIKeyManager: Main key management class
  - APIKey: Data model for keys

Usage:
  manager = APIKeyManager()
  key = manager.generate_key(user_id="user123", scopes=["read:docs"])
  is_valid = manager.validate_key("sk_abc123...")
"""