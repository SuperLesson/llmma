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

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        r = self.client.chat.create(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        c = r.responses[0].message.content
        assert isinstance(c, str)
        return provider.Result(
            c,
            provider.Usage(
                r.usage.input_tokens,
                r.usage.output_tokens,
            ),
            r,
        )

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        r = await self.async_client.chat.create(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        c = r.responses[0].message.content
        assert isinstance(c, str)
        return provider.Result(
            c,
            provider.Usage(
                r.usage.input_tokens,
                r.usage.output_tokens,
            ),
            r,
        )

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        for r in self.client.chat.create_stream(model=self.model, messages=t.cast(t.Any, messages), **kwargs):
            c = t.cast(str, r.responses[0].chunk.content)
            assert isinstance(c, str)
            yield provider.Result(
                c,
                provider.Usage(
                    r.usage.input_tokens,
                    r.usage.output_tokens,
                ),
                r,
            )

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        async for r in self.async_client.chat.create_stream(
            model=self.model, messages=t.cast(t.Any, messages), **kwargs
        ):
            c = t.cast(str, r.responses[0].chunk.content)
            assert isinstance(c, str)
            yield provider.Result(
                c,
                provider.Usage(
                    r.usage.input_tokens,
                    r.usage.output_tokens,
                ),
                r,
            )
