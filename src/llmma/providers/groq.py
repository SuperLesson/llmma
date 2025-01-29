import typing as t

import tiktoken
from attrs import define
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

    def __post_init__(self):
        super().__post_init__()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _count_tokens(self, content: list[dict]) -> int:
        # Groq uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        formatting_token_count = 4
        messages = content
        messages_text = ["".join(message.values()) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token, message in zip(tokens, messages, strict=False):
            n_tokens = len(token) + formatting_token_count
            if "name" in message:
                n_tokens += -1
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

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
