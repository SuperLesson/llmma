import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI, OpenAI

from .base import ModelInfo, StreamProvider


@define
class GroqProvider(StreamProvider):
    MODEL_INFO = {
        "llama-3.1-405b-reasoning": ModelInfo(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
        "llama-3.1-70b-versatile": ModelInfo(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
        "llama-3.1-8b-instant": ModelInfo(prompt_cost=0.05, completion_cost=0.08, context_limit=131072),
        "gemma2-9b-it": ModelInfo(prompt_cost=0.20, completion_cost=0.20, context_limit=131072),
        "llama-3.3-70b-versatile": ModelInfo(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
    }

    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        # Groq uses the same tokenizer as OpenAI
        return len(self.tokenizer.encode(content))

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async for chunk in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
