import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from loguru import logger
import dotenv

dotenv.load_dotenv()

from auth.rbac_dependencies import verify_jwt_token  # Your JWT verification

# ============================================
# CONFIGURATION
# ============================================

router = APIRouter(prefix="/api/documents", tags=["documents"])

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "legal-docs")

# Initialize Azure Blob Storage Client
if AZURE_STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    logger.info("✓ Azure Blob Storage client initialized")
else:
    blob_service_client = None
    logger.error("❌ AZURE_STORAGE_CONNECTION_STRING not configured")

@router.get("/view/{document_id}")
async def view_document_inline(document_id: str, current_user = Depends(verify_jwt_token)):
    """Stream PDF inline without download prompt"""
    try:
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=document_id
        )
        
        download_stream = blob_client.download_blob()
        
        return StreamingResponse(
            download_stream.chunks(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline; filename=document.pdf",
                "Cache-Control": "public, max-age=3600"
            }
        )
    except Exception as e:
        logger.error(f"Error viewing document: {e}")
        raise HTTPException(status_code=404, detail="Document not found")