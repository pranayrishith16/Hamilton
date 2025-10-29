"""
Immutable audit logging for security and compliance.

Logs:
  - Authentication events (login, logout, token generation)
  - Authorization events (access granted/denied)
  - Data access (document reads, queries, exports)
  - Configuration changes (policy updates, role changes)
  - Security incidents (suspicious activity, violations)

Features:
  - Structured logging with metadata
  - PostgreSQL persistence with indexes
  - Tenant isolation (multi-tenant safe)
  - Suspicious activity detection
  - Real-time alerting for critical events
  - Export for compliance audits
  - Integrity hashing (tamper detection)

Classes:
  - AuditLogger: Main logging class
  - EventType: Enum of event types
  - EventSeverity: INFO, WARNING, CRITICAL

Compliance: GDPR, HIPAA, SOC 2, ISO 27001
"""