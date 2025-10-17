from generation.models.adapters import interface
from generation.models.adapters import local
from generation.models.adapters import openai
from generation.models.adapters import openrouter

from generation.models.adapters.interface import (GeneratorAdapter,)
from generation.models.adapters.openrouter import (OpenRouterAdapter,)

__all__ = ['GeneratorAdapter', 'OpenRouterAdapter', 'interface', 'local',
           'openai', 'openrouter']
