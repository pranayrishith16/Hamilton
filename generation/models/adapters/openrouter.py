from email import message
import os
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
            # Build messages payload
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
                log_metrics({"generation.errors": 1})
                self.logger.error(f"OpenRouter API error: {e}")
                raise ValueError(f"OpenRouter API error: {e}")

            # Extract answer
            choice = completion.choices[0]
            answer = choice.message.content.strip()

            # After receiving `completion`
            usage = getattr(completion, "usage", None)
            total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

            log_metrics({
                "generation.tokens_used": total_tokens,
                "generation.prompt_tokens": prompt_tokens,
                "generation.completion_tokens": completion_tokens,
                "generation.context_chunks": len(context),
            })

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
                # Convert OpenAI chunk to dict format
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": chunk.choices[0].delta.content
                            }
                        }]
                    }
                    
        except OpenAIError as e:
            log_metrics({"generation.errors": 1})
            self.logger.error(f"OpenRouter API error: {e}")
            raise ValueError(f"OpenRouter API error: {e}")