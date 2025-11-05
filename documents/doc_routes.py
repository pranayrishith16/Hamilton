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
TEMP_CONTAINER_NAME = "temp-pdf-views"  # ✅ Temporary cache container

# Initialize Azure Blob Storage Client
if AZURE_STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    logger.info("✓ Azure Blob Storage client initialized")
else:
    blob_service_client = None
    logger.error("❌ AZURE_STORAGE_CONNECTION_STRING not configured")


# ============================================
# STARTUP: Create Temp Container If Not Exists
# ============================================

def create_temp_container():
    """Create temporary container on app startup"""
    try:
        if not blob_service_client:
            logger.error("Blob service client not initialized")
            return

        # Check if temp container exists
        temp_container_client = blob_service_client.get_container_client(
            TEMP_CONTAINER_NAME
        )
        
        try:
            temp_container_client.get_container_properties()
            logger.info(f"✓ Temp container already exists: {TEMP_CONTAINER_NAME}")
        except Exception:
            # Create if doesn't exist
            blob_service_client.create_container(name=TEMP_CONTAINER_NAME)
            logger.info(f"✓ Created temp container: {TEMP_CONTAINER_NAME}")
            
    except Exception as e:
        logger.error(f"Error creating temp container: {e}")


# Call on startup (add to your FastAPI app startup event)
# @app.on_event("startup")
# async def startup():
#     create_temp_container()


# ============================================
# ENDPOINT: Generate Temporary View URL
# ============================================

@router.post("/generate-view-url/{blob_path:path}")
async def generate_view_url(
    blob_path: str,
    jwt_user: dict = Depends(verify_jwt_token)  # ✅ JWT required
):
    """
    Generate temporary SAS URL for viewing PDF inline
    
    Flow:
    1. Validate JWT token
    2. Copy PDF from private blob to temp container
    3. Generate SAS URL with inline display parameter
    4. Schedule deletion after TTL
    5. Return secure temp URL
    
    Args:
        blob_path: Path to PDF in Azure (e.g., "case_rcpdfs/file.pdf")
        jwt_user: Validated JWT user from dependency injection
    
    Returns:
        {
            "url": "https://account.blob.core.windows.net/temp-pdf-views/...",
            "expires_in": 1800,
            "temp_file": "user-uuid-filename.pdf"
        }
    
    Raises:
        401: Invalid JWT token
        404: Document not found
        500: Error during processing
    """
    
    try:
        # ============================================
        # STEP 1: Validate JWT
        # ============================================
        
        user_id = jwt_user.get("sub")
        if not user_id:
            logger.warning("[GENERATE-URL] Request without valid JWT")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        logger.info(f"[GENERATE-URL] User {user_id} requesting view URL for: {blob_path}")
        
        if not blob_service_client:
            logger.error("[GENERATE-URL] Blob service client not initialized")
            raise HTTPException(status_code=500, detail="Storage not configured")
        
        # ============================================
        # STEP 2: Get source blob from private container
        # ============================================
        
        try:
            source_container_client = blob_service_client.get_container_client(
                AZURE_CONTAINER_NAME
            )
            source_blob_client = source_container_client.get_blob_client(blob_path)
            
            # Check if blob exists
            properties = source_blob_client.get_blob_properties()
            blob_size = properties.size
            
            logger.info(
                f"[GENERATE-URL] Source blob found: {blob_path} ({blob_size} bytes)"
            )
            
        except Exception as e:
            logger.warning(f"[GENERATE-URL] Source blob not found: {blob_path}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        # ============================================
        # STEP 3: Create unique temp filename
        # ============================================
        
        # Extract original filename
        original_filename = blob_path.split("/")[-1]
        
        # Create unique temp name: user_id-uuid-original_filename
        temp_filename = f"{user_id}-{uuid.uuid4()}-{original_filename}"
        
        logger.info(f"[GENERATE-URL] Temp filename: {temp_filename}")
        
        # ============================================
        # STEP 4: Copy blob from private to temp container
        # ============================================
        
        try:
            # Get temp container client
            temp_container_client = blob_service_client.get_container_client(
                TEMP_CONTAINER_NAME
            )
            temp_blob_client = temp_container_client.get_blob_client(temp_filename)
            
            # Download from private blob
            download_stream = source_blob_client.download_blob()
            
            # Upload to temp container
            temp_blob_client.upload_blob(
                download_stream,
                overwrite=True,
            )
            
            logger.info(f"[GENERATE-URL] Copied to temp container: {temp_filename}")
            
        except Exception as e:
            logger.error(f"[GENERATE-URL] Error copying blob: {e}")
            raise HTTPException(status_code=500, detail="Error copying document")
        
        # ============================================
        # STEP 5: Generate SAS URL with TTL
        # ============================================
        
        try:
            expiry_time = datetime.utcnow() + timedelta(minutes=30)  # 30 minute TTL
            
            # ✅ Generate SAS with inline parameter
            sas_token = generate_blob_sas(
                account_name=AZURE_STORAGE_ACCOUNT_NAME,
                container_name=TEMP_CONTAINER_NAME,
                blob_name=temp_filename,
                account_key=AZURE_STORAGE_ACCOUNT_KEY,
                permission=BlobSasPermissions(read=True),  # Read-only
                expiry=expiry_time,
                # ✅ CRITICAL: Force inline display instead of download
                response_content_disposition="inline; filename=document.pdf",
                response_content_type="application/pdf"
            )
            
            temp_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{TEMP_CONTAINER_NAME}/{temp_filename}?{sas_token}"
            
            logger.info(
                f"[GENERATE-URL] SAS URL generated for user {user_id}, expires at: {expiry_time}"
            )
            
        except Exception as e:
            logger.error(f"[GENERATE-URL] Error generating SAS URL: {e}")
            raise HTTPException(status_code=500, detail="Error generating SAS URL")
        
        # ============================================
        # STEP 6: Schedule deletion after TTL
        # ============================================
        
        # Delete after 31 minutes (1 minute after SAS expires)
        asyncio.create_task(delete_temp_blob_async(temp_filename, delay_minutes=31))
        
        logger.info(f"[GENERATE-URL] Scheduled deletion for: {temp_filename}")
        
        # ============================================
        # STEP 7: Return response
        # ============================================
        
        return {
            "url": temp_url,
            "expires_in": 1800,  # 30 minutes in seconds
            "temp_file": temp_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[GENERATE-URL] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================
# BACKGROUND TASK: Delete Temp Blob After TTL
# ============================================

async def delete_temp_blob_async(blob_name: str, delay_minutes: int = 31):
    """
    Delete temporary blob after delay (background task)
    
    Args:
        blob_name: Name of temp blob to delete
        delay_minutes: Minutes to wait before deleting (default 31 = 1 min after SAS expires)
    """
    
    try:
        # Wait for delay
        await asyncio.sleep(delay_minutes * 60)
        
        # Delete blob
        temp_container_client = blob_service_client.get_container_client(
            TEMP_CONTAINER_NAME
        )
        temp_blob_client = temp_container_client.get_blob_client(blob_name)
        temp_blob_client.delete_blob()
        
        logger.info(f"[CLEANUP] Deleted temp blob: {blob_name}")
        
    except Exception as e:
        logger.error(f"[CLEANUP] Error deleting temp blob {blob_name}: {e}")


# ============================================
# OPTIONAL: List temp blobs (for debugging)
# ============================================

@router.get("/temp-blobs")
async def list_temp_blobs(jwt_user: dict = Depends(verify_jwt_token)):
    """List all temp blobs (for debugging)"""
    
    try:
        temp_container_client = blob_service_client.get_container_client(
            TEMP_CONTAINER_NAME
        )
        
        blobs = list(temp_container_client.list_blobs())
        
        return {
            "count": len(blobs),
            "blobs": [blob.name for blob in blobs]
        }
        
    except Exception as e:
        logger.error(f"[TEMP-BLOBS] Error: {e}")
        raise HTTPException(status_code=500, detail="Error listing temp blobs")
