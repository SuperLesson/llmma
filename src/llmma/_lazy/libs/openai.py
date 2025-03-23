import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI
from openai import OpenAI as OpenAIAPI

from ... import provider


@define
class OpenAI(provider.Stream):
    client: OpenAIAPI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAIAPI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    def prepare_input(
        self,
        **kwargs,
    ) -> dict:
        if not kwargs.get("max_completion_tokens") and self.info.quirks.get("use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
        return kwargs

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        kwargs = self.prepare_input(**kwargs)
        r = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        c = r.choices[0].message.content
        assert c
        u = r.usage
        assert u
        return provider.Result(c, provider.Usage(u.prompt_tokens, u.completion_tokens), r)

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        kwargs = self.prepare_input(**kwargs)
        r = await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        c = r.choices[0].message.content
        assert c
        u = r.usage
        assert u
        return provider.Result(c, provider.Usage(u.prompt_tokens, u.completion_tokens), r)

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        kwargs = self.prepare_input(**kwargs)
        for r in self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            c = r.choices[0].delta.content
            assert c
            u = r.usage
            assert u
            yield provider.Result(c, provider.Usage(u.prompt_tokens, u.completion_tokens), r)

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        kwargs = self.prepare_input(**kwargs)
        async for r in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            c = r.choices[0].delta.content
            assert c
            u = r.usage
            assert u
            yield provider.Result(c, provider.Usage(u.prompt_tokens, u.completion_tokens), r)
