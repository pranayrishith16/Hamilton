
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, AsyncGenerator
from loguru import logger
from dotenv import load_dotenv
from ingestion.dataprep.chunkers.base import Chunk
from generation.models.adapters.interface import GeneratorAdapter
from orchestrator.observability import trace_request, log_metrics
from openai import OpenAI, OpenAIError

from generation.prompts.render_template import render_messages

load_dotenv()

class OpenRouterAdapter(GeneratorAdapter):

    def __init__(self,
                 api_url:str='https://openrouter.ai/api/v1',
                 api_key:Optional[str]=None,
                 model:str=None,
                 temperature:float=0.0,
                 max_tokens:int=2500
                 ):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key or not key.strip():
            raise ValueError("OpenRouter API key not set")
        self.client = OpenAI(base_url=self.api_url, api_key=key.strip())

        

    def generate(self, query: str, context: List[Chunk], **kwargs) -> str:
        """Generate an answer given query and context."""

        with trace_request("generate", "openrouter_adapter.generate"):
            messages = self._build_messages(query, context)

            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_headers=kwargs.get("extra_headers", {}),
                    extra_body=kwargs.get("extra_body", {}),
                )
            except OpenAIError as e:
                self.logger.error(f"OpenRouter API error: {e}")
                raise

            # Extract answer
            answer = completion.choices[0].message.content.strip()

            usage = getattr(completion, "usage", None)
            total_tokens = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

        return answer
    

    def _build_messages(self, query: str, context: List[Chunk]) -> List[Dict[str, Any]]:
        return render_messages(query,context)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openrouter",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_url": self.api_url,
        }
    
    def stream_generate(self,query:str,context:List[Chunk],**kwargs) -> Iterator[Any]:
        """
        Streams OpenRouter/OpenAI chunks directly as they arrive
        """

        messages = self._build_messages(query, context)

        token_count = 0
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,  # This is the key
                extra_headers=kwargs.get("extra_headers", {}),
                extra_body=kwargs.get("extra_body", {}),
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    token_count += len(content.split())
                    yield {"choices": [{"delta": {"content": content}}]}
                    
        except OpenAIError as e:
            self.logger.error(f"OpenRouter API error: {e}")
            raise
