"""
Multi-tenant isolation for secure data separation.

Features:
  - Tenant ID extraction from request (API key, JWT, header)
  - Automatic tenant_id filtering on all queries
  - Separate schemas or row-level security (RLS)
  - Prevent cross-tenant data access (even for admins)
  - Tenant-specific rate limiting
  - Isolated audit logs per tenant

Data Isolation:
  - Database: PostgreSQL RLS or separate schemas
  - Storage: Separate S3 buckets or folders per tenant
  - Vector DB: Separate collections per tenant
  - Cache: Tenant-prefixed keys

Classes:
  - TenantContext: Extract and validate tenant ID
  - TenantFilter: Apply tenant filtering to queries
  - TenantIsolationMiddleware: FastAPI middleware

Security: Zero-trust multi-tenancy
"""