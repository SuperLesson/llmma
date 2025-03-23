import typing as t

import tiktoken
from attrs import define, field
from mistralai import Mistral as MistralAPI

from ... import provider


@define
class Mistral(provider.Stream):
    client: MistralAPI = field(init=False)

    def __attrs_post_init__(self):
        self.client = MistralAPI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        # TODO: update after Mistrar support count token in their SDK
        # use gpt 3.5 turbo for estimation now
        return len(self.tokenizer.encode(content))

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        with self.client as client:
            r = client.chat.complete(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        cs = r.choices
        assert cs
        c = cs[0].message.content
        assert isinstance(c, str)
        return provider.Result(
            c,
            provider.Usage(
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            ),
            r,
        )

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        async with self.client as client:
            r = await client.chat.complete_async(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        cs = r.choices
        assert cs
        c = cs[0].message.content
        assert isinstance(c, str)
        return provider.Result(
            c,
            provider.Usage(
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            ),
            r,
        )

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        with (
            self.client as client,
            client.chat.stream(model=self.model, messages=t.cast(t.Any, messages), **kwargs) as stream,
        ):
            for r in stream:
                cs = r.data.choices
                assert cs
                c = cs[0].delta.content
                assert isinstance(c, str)
                u = r.data.usage
                assert u
                yield provider.Result(
                    c,
                    provider.Usage(
                        u.prompt_tokens,
                        u.completion_tokens,
                    ),
                    r,
                )

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        async with self.client as client:
            async for r in await client.chat.stream_async(
                model=self.model, messages=t.cast(t.Any, messages), **kwargs
            ):
                cs = r.data.choices
                assert cs
                c = cs[0].delta.content
                assert isinstance(c, str)
                u = r.data.usage
                assert u
                yield provider.Result(
                    c,
                    provider.Usage(
                        u.prompt_tokens,
                        u.completion_tokens,
                    ),
                    r,
                )
