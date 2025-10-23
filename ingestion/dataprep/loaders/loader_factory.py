from typing import Optional
from ingestion.dataprep.loaders.base_loader import BaseLoader, LoaderConfig
from ingestion.dataprep.loaders.azure_loader import AzureBlobLoader
from loguru import logger

class LoaderFactory:
    """Factory for creating loaders based on configuration."""
    
    _loaders = {
        'azure': AzureBlobLoader,
    }
    
    @classmethod
    def register_loader(cls, name: str, loader_class):
        """Register a new loader class"""
        cls._loaders[name] = loader_class
        logger.info(f"Registered loader: {name}")
    
    @classmethod
    def create_loader(cls, 
                     loader_type: str,
                     config: LoaderConfig = None,
                     **kwargs) -> BaseLoader:
        """
        Create a loader instance.
        
        Args:
            loader_type: 'azure' or registered custom type
            config: LoaderConfig instance
            **kwargs: Loader-specific arguments
        
        Returns:
            BaseLoader instance
            
        Raises:
            ValueError: If loader_type is unknown
        """
        if loader_type not in cls._loaders:
            raise ValueError(
                f"Unknown loader type: {loader_type}. "
                f"Available: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[loader_type]
        logger.info(f"Creating loader: {loader_type}")
        return loader_class(config=config, **kwargs)