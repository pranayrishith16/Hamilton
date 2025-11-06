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

router = APIRouter(prefix="/documents", tags=["documents"])

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

@router.get("/view/{blob_path:path}")
async def view_document_inline(
    blob_path: str,
    current_user: dict = Depends(verify_jwt_token)
):
    """
    Stream PDF inline without download prompt.
    Accepts paths with slashes like: case_rcpdfs/sharon_kennell_v._diahann_gates.pdf
    """
    try:
        import urllib.parse
        
        # URL decode the blob_path (in case it's encoded like case_rcpdfs%2Ffile.pdf)
        blob_path = urllib.parse.unquote(blob_path)
        
        logger.info(f"[VIEW] Requesting document: {blob_path}")
        
        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=blob_path
        )
        
        # Verify blob exists before streaming
        try:
            properties = blob_client.get_blob_properties()
            logger.info(f"[VIEW] Found blob: {blob_path} ({properties.size} bytes)")
        except Exception as e:
            logger.error(f"[VIEW] Blob not found: {blob_path} - {str(e)}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Stream the PDF
        download_stream = blob_client.download_blob()
        
        # Extract filename for header
        filename = blob_path.split('/')[-1]
        
        return StreamingResponse(
            download_stream.chunks(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "public, max-age=3600",
                "X-Content-Type-Options": "nosniff",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VIEW] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Document viewing failed")