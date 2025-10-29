"""
PII (Personally Identifiable Information) detection and redaction.

Detects:
  - Social Security Numbers (XXX-XX-XXXX)
  - Credit card numbers (XXXX-XXXX-XXXX-XXXX)
  - Email addresses, phone numbers
  - Names (NLP-based), addresses
  - Bank accounts, driver's licenses
  - Medical record numbers

Redaction Strategies:
  - Full: Replace with [REDACTED]
  - Partial: Show last 4 digits (****-1234)
  - Masking: Replace with dummy data
  - Context-aware: Keep for authorized users

Features:
  - Regex-based pattern matching
  - NER (Named Entity Recognition) for names
  - Configurable rules per client
  - Audit log of redactions
  - Real-time redaction before LLM input

Classes:
  - PIIDetector: Main detection engine
  - RedactionEngine: Apply redaction strategies
  - PIIRule: Individual detection rule

Compliance: GDPR, CCPA, HIPAA
"""