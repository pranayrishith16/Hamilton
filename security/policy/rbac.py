"""
Role-Based Access Control (RBAC) system.

Roles:
  - Admin: Full system access
  - LawFirmPartner: Can query docs, assign cases, manage associates
  - Associate: Can query docs, limited to assigned cases
  - Reader: Read-only access to documents
  - Guest: Temporary, limited access

Permissions:
  - read:documents, write:queries, export:results
  - admin:users, manage:policies, view:analytics

Features:
  - Role assignment to users
  - Permission checks before actions
  - Role hierarchy (inheritance)
  - Dynamic role updates
  - Audit logging of role changes

Classes:
  - RBACManager: Main access control logic
  - Role: Role definition with permissions
  - PermissionChecker: Fast permission lookup

Standards: NIST RBAC model
"""