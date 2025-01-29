import typing as t

import tiktoken
from attrs import define
from reka.client import AsyncReka, Reka

from .base import ModelInfo, StreamProvider, msg_as_str


@define
class RekaProvider(StreamProvider):
    MODEL_INFO = {
        "reka-edge": ModelInfo(prompt_cost=0.4, completion_cost=1.0, context_limit=128000),
        "reka-flash": ModelInfo(prompt_cost=0.8, completion_cost=2.0, context_limit=128000),
        "reka-core": ModelInfo(prompt_cost=3.0, completion_cost=15.0, context_limit=128000),
    }

    def __post_init__(self):
        self.model = self.model or "reka-core"
        self.client = Reka(api_key=self.api_key)
        self.async_client = AsyncReka(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        # Reka uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return sum([len(enc.encode(msg_as_str([message]))) for message in content])

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat.create(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        return {
            "completion": t.cast(str, response.responses[0].message.content),
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "latency": self.latency,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        response = await self.async_client.chat.create(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        return {
            "completion": t.cast(str, response.responses[0].message.content),
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "latency": self.latency,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for r in self.client.chat.create_stream(model=self.model, messages=t.cast(t.Any, messages), **kwargs):
            yield t.cast(str, r.responses[0].chunk.content)

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async for chunk in self.async_client.chat.create_stream(
            model=self.model, messages=t.cast(t.Any, messages), **kwargs
        ):
            yield t.cast(str, chunk.responses[0].chunk.content)
