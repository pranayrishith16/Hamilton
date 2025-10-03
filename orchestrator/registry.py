from typing import List, Dict, Optional, Any
import importlib
from loguru import logger
import yaml
from pathlib import Path

from orchestrator.interfaces import (
    Embedder, Generator, Memory, 
    Storage, MLOpsBackend
)
from retrieval.retrievers.interface import Retriever

class Registry:

    def __init__(self,config_path:Path=None):
        self.components: Dict[str, Any] = {}
        self.config_path = config_path or "configs/pipelines/default.yaml"
        self.logger = logger
        self.config = self._load_config()


    def _load_config(self) -> Dict[str,Any]:
        try:
            with open(self.config_path,'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error('YAML file not found')
            raise

    def register(self,name:str,component:str) -> None:
        self.components[name] = component
    
    def get(self, name: str) -> Any:
        """Get a component instance, creating it if needed."""
        if name in self.components:
            print(self.components)
            return self.components[name]
    
        
        # Create component from config
        if name in self.config:
            component_config = self.config[name]
            component = self._create_component(component_config)
            self.components[name] = component

            return component
        
        raise ValueError(f"Component '{name}' not found in registry or config")
    
    def _create_component(self, config: Dict[str, Any]) -> Any:
        """Create component from configuration."""
        try:
            module_path = config["module"]
            class_name = config["class"]
            component_config = config.get("config", {})
            
            # Import module and get class
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            # Create instance with config
            return component_class(**component_config)
        
        except Exception as e:
            raise ValueError(f"Failed to create component: {e}")
        
    def list_components(self) -> Dict[str, str]:
        """List all available components."""
        return {name: str(type(comp)) for name, comp in self.components.items()}
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """Reload configuration and clear cached components."""
        if config_path:
            self.config_path = config_path
        self.config = self._load_config()
        self.components.clear()

# Global registry instance
registry = Registry()