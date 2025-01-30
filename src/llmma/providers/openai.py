import json
import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI, OpenAI

from .base import ModelInfo, StreamProvider


@define
class OpenAIProvider(StreamProvider):
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-4o": ModelInfo(
            prompt_cost=2.5,
            completion_cost=10.0,
            context_limit=128_000,
            output_limit=16_384,
            limit_per_minute=30_000,
        ),
        "gpt-4o-mini": ModelInfo(
            prompt_cost=0.15,
            completion_cost=0.60,
            context_limit=128_000,
            output_limit=16_384,
            limit_per_minute=200_000,
        ),
        "o1": ModelInfo(
            prompt_cost=15.0,
            completion_cost=60.0,
            context_limit=200_000,
            output_limit=100_000,
            quirks={
                "use_max_completion_tokens": True,
            },
        ),
        "o1-mini": ModelInfo(
            prompt_cost=3.0,
            completion_cost=12.0,
            context_limit=128_000,
            output_limit=65_536,
            quirks={
                "use_max_completion_tokens": True,
            },
            limit_per_minute=200_000,
        ),
    }

    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return sum(len(enc.encode(t["content"])) + 4 for t in content)

    def prepare_input(
        self,
        **kwargs,
    ) -> dict:
        if not kwargs.get("max_completion_tokens") and self.info.quirks.get("use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        if response.choices[0].message.function_call:
            function_call = {
                "name": response.choices[0].message.function_call.name,
                "arguments": json.loads(response.choices[0].message.function_call.arguments),
            }
            completion = ""
        else:
            function_call = {}
            completion = response.choices[0].message.content

        assert response.usage
        return {
            "completion": completion,
            "function_call": function_call,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
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
        kwargs = self.prepare_input(**kwargs)
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if c := chunk.choices[0].delta.content:
                yield c

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        kwargs = self.prepare_input(**kwargs)
        async for chunk in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if c := chunk.choices[0].delta.content:
                yield c
