from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Header, Depends
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

# ============ SINGLE ENDPOINT ============

@router.get("/access/{blob_path:path}")
async def getDocumentAccess(
    blob_path: str,
    expiry_minutes: int = Query(30, ge=5, le=240),
    jwt_user: dict = Depends(verify_jwt_token)
):
    """
    Generate SAS URL for accessing document in Azure Blob Storage
    """
    try:
        if not blob_service_client:
            logger.error("Azure Blob Storage not configured")
            raise HTTPException(
                status_code=503,
                detail="Document service unavailable - Azure not configured"
            )
        
        user_id = jwt_user.get("sub")
        if not user_id:
            logger.error("JWT token missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token")
        
        logger.debug(f"[ACCESS] User {user_id} requesting SAS URL for: {blob_path}")
        
        # Verify blob exists
        try:
            blob_client = blob_service_client.get_blob_client(
                container=AZURE_CONTAINER_NAME,
                blob=blob_path
            )
            properties = blob_client.get_blob_properties()
            logger.debug(f"[ACCESS] Blob found: {blob_path} ({properties.size} bytes)")
        except Exception as e:
            if "ContainerNotFound" in str(e):
                logger.error(f"Container not found: {e}")
                raise HTTPException(status_code=503, detail="Container not found")
            elif "BlobNotFound" in str(e) or "not found" in str(e).lower():
                logger.error(f"Blob not found: {blob_path}")
                raise HTTPException(status_code=404, detail="Document not found")
            else:
                logger.error(f"Error accessing blob: {blob_path} - {e}")
                raise HTTPException(status_code=500, detail="Failed to access document")
        
        # ✅ Generate SAS token with validation
        try:
            if not AZURE_STORAGE_ACCOUNT_KEY:
                logger.error("AZURE_STORAGE_ACCOUNT_KEY is not set")
                raise HTTPException(
                    status_code=503,
                    detail="Document service unavailable - Storage account key not configured. "
                           "Set AZURE_STORAGE_ACCOUNT_KEY environment variable."
                )
            
            if not AZURE_STORAGE_ACCOUNT_NAME:
                logger.error("AZURE_STORAGE_ACCOUNT_NAME is not set")
                raise HTTPException(
                    status_code=503,
                    detail="Document service unavailable - Storage account name not configured"
                )
            
            logger.debug(f"Generating SAS token for blob: {blob_path}")
            
            sas_token = generate_blob_sas(
                account_name=AZURE_STORAGE_ACCOUNT_NAME,
                container_name=AZURE_CONTAINER_NAME,
                blob_name=blob_path,
                account_key=AZURE_STORAGE_ACCOUNT_KEY,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes),
                rscd="inline",           
                rsct="application/pdf"
            )
            
            logger.debug(f"SAS token generated successfully")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate SAS token: {type(e).__name__}: {e}")
            if "account_key" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="Storage account key not configured - check AZURE_STORAGE_ACCOUNT_KEY"
                )
            raise HTTPException(status_code=500, detail="Failed to generate access URL")
        
        # Build full URL
        sas_url = f"{blob_client.url}?{sas_token}"
        
        logger.info(f"✓ SAS URL generated for {user_id}: {blob_path}")
        
        return {
            "success": True,
            "url": sas_url,
            "expires_in_minutes": expiry_minutes,
            "blob_path": blob_path,
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")