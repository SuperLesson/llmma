import typing as t

import tiktoken
from attrs import define, field
from reka.client import AsyncReka
from reka.client import Reka as RekaAPI

from ... import provider


@define
class Reka(provider.Stream):
    client: RekaAPI = field(init=False)
    async_client: AsyncReka = field(init=False)

    def __attrs_post_init__(self):
        self.client = RekaAPI(api_key=self.api_key)
        self.async_client = AsyncReka(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        # Reka uses the same tokenizer as OpenAI
        return len(self.tokenizer.encode(content))

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
