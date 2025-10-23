from typing import List, Dict, Optional, Any
import importlib
from loguru import logger
import yaml
from pathlib import Path

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
        self.logger.info(f"Registered component: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get a component instance, creating it if needed.
        Only for components with 'module' and 'class' in config.
        """
        if name in self.components:
            return self.components[name]
    
        
        # Create component from config
        if name in self.config:
            component_config = self.config[name]

            # Check if this is a component definition (has module + class)
            if isinstance(component_config, dict) and "module" in component_config and "class" in component_config:
                try:
                    component = self._create_component(component_config)
                    self.components[name] = component
                    return component
                except Exception as e:
                    raise ValueError(f"Failed to create component '{name}': {e}")
            else:
                raise ValueError(
                    f"'{name}' in config is not a component definition. "
                    f"It should have 'module' and 'class' keys. "
                    f"Use registry.config['{name}'] to get raw config values instead."
                )

        raise ValueError(f"Component '{name}' not found in registry or config")
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get raw configuration values (not a component).
        Use this for plain config sections like 'azure_loader'.
        """
        if name not in self.config:
            raise ValueError(f"Configuration section '{name}' not found")

        config_section = self.config[name]
        
        if not isinstance(config_section, dict):
            raise ValueError(f"Configuration section '{name}' is not a dictionary")

        return config_section
    
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
    
    def list_config_sections(self) -> List[str]:
        """List all configuration sections in YAML."""
        return list(self.config.keys())
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """Reload configuration and clear cached components."""
        if config_path:
            self.config_path = config_path
        self.config = self._load_config()
        self.components.clear()
        self.logger.info("Configuration reloaded and components cleared")

# Global registry instance
registry = Registry()