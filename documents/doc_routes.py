from azure.storage.blob import BlobServiceClient
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
import os
import dotenv

dotenv.load_dotenv()

from auth.rbac_dependencies import verify_jwt_token

router = APIRouter(prefix="/api/documents", tags=["documents"])

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "legal-docs")

# Initialize client
if AZURE_STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    logger.info("✓ Azure Blob Storage client initialized")
else:
    blob_service_client = None
    logger.error("❌ AZURE_STORAGE_CONNECTION_STRING not configured")


# ✅ HELPER FUNCTION: Generator for true streaming
def generate_chunks(blob_client):
    """
    Generator that yields chunks directly from Azure
    Never loads entire file into memory
    
    Yields:
        bytes: 4KB chunks from blob
    """
    try:
        download_stream = blob_client.download_blob()
        
        chunk_count = 0
        for chunk in download_stream.chunks():
            chunk_count += 1
            if chunk_count % 100 == 0:
                logger.debug(f"[STREAM] Sent {chunk_count} chunks...")
            
            # ✅ Yield immediately (don't buffer)
            yield chunk
        
        logger.info(f"[STREAM] Complete: {chunk_count} total chunks")
    
    except Exception as e:
        logger.error(f"[STREAM] Error during chunk generation: {e}")
        raise


# ============ MAIN ENDPOINT ============

@router.get("/view/{blob_path:path}")
async def viewDocument(
    blob_path: str,
    jwt_user: dict = Depends(verify_jwt_token)
):
    """
    ✅ True streaming PDF viewer (no memory bloat)
    
    Security: JWT token must be valid (30-min expiry)
    Viewing: Browser displays inline (not download)
    Memory: ~4KB (one chunk) regardless of file size
    
    Args:
        blob_path: Path to PDF in Azure Blob (e.g., "case_rcpdfs/file.pdf")
        jwt_user: Validated JWT user (dependency injection)
    
    Returns:
        StreamingResponse: PDF chunks streamed to browser
    
    Raises:
        HTTPException: 401 if token invalid, 404 if blob not found, 500 if error
    """
    
    try:
        # ============================================
        # STEP 1: Validate User
        # ============================================
        
        user_id = jwt_user.get("sub")
        if not user_id:
            logger.warning(f"[VIEW] Request without valid JWT")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        logger.info(f"[VIEW] User {user_id} viewing: {blob_path}")
        
        
        # ============================================
        # STEP 2: Get Blob Reference & Verify Exists
        # ============================================
        
        try:
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_CONTAINER_NAME,
                blob=blob_path
            )
            
            # Check blob exists and get size
            properties = blob_client.get_blob_properties()
            blob_size_mb = properties.size / (1024 * 1024)
            logger.info(f"[VIEW] Blob found: {blob_path} ({blob_size_mb:.2f} MB)")
        
        except Exception as e:
            if "BlobNotFound" in str(e):
                logger.warning(f"[VIEW] Blob not found: {blob_path}")
                raise HTTPException(status_code=404, detail="Document not found")
            else:
                logger.error(f"[VIEW] Error checking blob: {e}")
                raise HTTPException(status_code=500, detail="Failed to access document")
        
        
        # ============================================
        # STEP 3: Stream Response (✅ Generator, not buffer)
        # ============================================
        
        # ✅ Pass generator function, not loaded BytesIO
        # StreamingResponse will call generate_chunks() and iterate
        return StreamingResponse(
            generate_chunks(blob_client),  # ✅ Generator (not buffer)
            media_type="application/pdf",
            headers={
                # ✅ Force inline viewing (not download dialog)
                "Content-Disposition": "inline; filename=document.pdf",
                
                # Prevent caching (security - JWT expires in 30 mins)
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                
                # Allow CORS if needed
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[VIEW] Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
