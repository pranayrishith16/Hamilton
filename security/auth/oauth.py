"""
OAuth 2.0 authorization flow for third-party integrations.

Supported Flows:
  - Authorization Code Flow (most secure, for web apps)
  - Client Credentials Flow (for server-to-server)
  - Refresh Token Flow (long-lived sessions)

Features:
  - Generate authorization codes
  - Exchange codes for access/refresh tokens
  - Validate JWT tokens
  - Scope-based permissions
  - Token revocation
  - Redirect URI validation

Classes:
  - OAuthProvider: Main OAuth server implementation
  - TokenManager: Token generation and validation
  - ScopeValidator: Permission checking

Standards: RFC 6749, RFC 7519 (JWT)
"""