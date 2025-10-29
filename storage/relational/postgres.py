"""
PostgreSQL connection and ORM models for relational data.

Tables:
  - users: User accounts and profiles
  - api_keys: API key hashes and metadata
  - audit_events: Immutable audit log
  - retention_policies: Data retention rules
  - roles: RBAC role definitions

Features:
  - SQLAlchemy ORM models
  - Connection pooling
  - Query optimization with indexes
  - Migration support (Alembic)
  - Backup and restore utilities

Classes:
  - DatabaseManager: Connection management
  - User, APIKey, AuditEvent: ORM models
  - QueryBuilder: Complex query construction

Connection: Connection string from env (POSTGRES_URL)
"""