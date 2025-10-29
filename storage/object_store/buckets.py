"""
S3/Azure Blob Storage operations for document storage.

Operations:
  - Upload documents (PDFs, DOCX, TXT)
  - Download by key
  - List files with filters
  - Delete files
  - Generate presigned URLs (temporary access)
  - Copy/move between buckets

Features:
  - Support S3, Azure Blob, and GCS
  - Automatic retries on failure
  - Progress tracking for large files
  - Multipart uploads for large files
  - Server-side encryption (SSE)

Classes:
  - ObjectStore: Abstract interface
  - S3Store, AzureStore, GCSStore: Implementations
  - UploadManager: Handle large file uploads

Usage: Stores raw legal documents before processing
"""