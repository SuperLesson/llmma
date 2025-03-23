import typing as t

import tiktoken
import together
from attrs import define, field
from together import AsyncTogether
from together import Together as TogetherAPI

from ... import provider


@define
class Together(provider.Stream):
    client: TogetherAPI = field(init=False)
    async_client: AsyncTogether = field(init=False)

    def __attrs_post_init__(self):
        self.client = TogetherAPI(api_key=self.api_key)
        self.async_client = AsyncTogether(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        # Together uses the same tokenizer as OpenAI
        return len(self.tokenizer.encode(content))

    @staticmethod
    def _cast(r: together.types.ChatCompletionResponse) -> provider.Result:
        cs = r.choices
        assert cs
        msg = cs[0].message
        assert msg
        text = msg.content
        assert isinstance(text, str)
        u = r.usage
        assert u
        return provider.Result(
            text,
            provider.Usage(
                u.prompt_tokens,
                u.completion_tokens,
            ),
            raw=r,
        )

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        return self._cast(
            t.cast(
                together.types.ChatCompletionResponse,
                self.client.chat.completions.create(model=self.model, messages=messages, stream=False, **kwargs),
            )
        )

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        return self._cast(
            t.cast(
                together.types.ChatCompletionResponse,
                await self.async_client.chat.completions.create(model=self.model, messages=messages, **kwargs),
            )
        )

    @staticmethod
    def _cast_chunk(r: together.types.ChatCompletionChunk) -> provider.Result:
        cs = r.choices
        assert cs
        d = cs[0].delta
        assert d
        c = d.content
        assert c
        u = r.usage
        assert u
        return provider.Result(c, provider.Usage(u.prompt_tokens, u.completion_tokens), r)

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        for chunk in self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs):
            yield self._cast_chunk(t.cast(together.types.ChatCompletionChunk, chunk))

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        async for chunk in t.cast(
            t.AsyncGenerator[together.types.ChatCompletionChunk, None],
            self.async_client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs),
        ):
            yield self._cast_chunk(t.cast(together.types.ChatCompletionChunk, chunk))
