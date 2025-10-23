from typing import Generator, Dict, Any, Optional, List
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from ingestion.dataprep.loaders.base_loader import BaseLoader, LoaderConfig
from loguru import logger

class AzureBlobLoader(BaseLoader):
    """
    Loads files from Azure Blob Storage with batch streaming support.
    Automatically handles authentication and file streaming.
    
    Example:
        loader = AzureBlobLoader(
            connection_string="your-connection-string",
            container_name="legal-documents",
            config=LoaderConfig(batch_size=50, file_extensions=['.pdf'])
        )
        
        for batch in loader.batch_load():
            # Process batch of files
            process_batch(batch)
    """
    
    def __init__(self, container_name: str, **kwargs):
        """
        Loads files from Azure Blob Storage with batch streaming support.
        Automatically handles authentication and file streaming.

        Example:
            loader = AzureBlobLoader(
                container_name="legal-documents",
                connection_string="your-connection-string",
                config=LoaderConfig(batch_size=50, file_extensions=['.pdf'])
            )

            for batch in loader.batch_load():
                # Process batch of files
                process_batch(batch)
        """
        config = kwargs.pop('config', None)
        super().__init__(config)

        self.container_name = container_name
        self.connection_string = kwargs.pop('connection_string', None)
        self.account_url = kwargs.pop('account_url', None)

        try:
            if self.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                logger.info("Initialized AzureBlobLoader with connection string")
            else:
                credential = DefaultAzureCredential()
                self.blob_service_client = BlobServiceClient(
                    self.account_url,
                    credential=credential
                )
                logger.info("Initialized AzureBlobLoader with DefaultAzureCredential")

            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            logger.info(f"Connected to Azure Blob container: {self.container_name}")

        except Exception as e:
            logger.error(f"Failed to connect to container {self.container_name}: {e}")
            raise
    
    def lazy_load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream files from Azure Blob one at a time.
        Memory efficient - only one file in memory at a time.
        """
        try:
            blob_list = self.container_client.list_blobs()
            total_blobs = 0
            
            for blob in blob_list:
                # Filter by file extension if configured
                if self.config.file_extensions:
                    if not any(blob.name.endswith(ext) 
                             for ext in self.config.file_extensions):
                        continue
                
                try:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    content = blob_client.download_blob().readall()
                    
                    total_blobs += 1
                    
                    yield {
                        'file_name': blob.name,
                        'content': content,
                        'size_bytes': len(content),
                        'source': 'azure_blob'
                    }
                    
                except Exception as e:
                    logger.error(f"Error downloading blob {blob.name}: {e}")
                    if not self.config.skip_errors:
                        raise
                    continue
            
            logger.info(f"Loaded {total_blobs} files from Azure Blob Storage")
            
        except Exception as e:
            logger.error(f"Error listing blobs: {e}")
            if not self.config.skip_errors:
                raise
    
    def list_files(self) -> List[str]:
        """List all files in container (for inspection)"""
        try:
            blobs = self.container_client.list_blobs()
            files = [blob.name for blob in blobs]
            logger.info(f"Found {len(files)} files in container")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get container statistics"""
        try:
            blobs = self.container_client.list_blobs()
            total_size = 0
            file_count = 0
            file_types = {}
            
            for blob in blobs:
                file_count += 1
                total_size += blob.size
                
                # Count by file type
                ext = blob.name.split('.')[-1] if '.' in blob.name else 'unknown'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_types': file_types,
                'container_name': self.container_name
            }
        except Exception as e:
            logger.error(f"Error getting container stats: {e}")
            return {}