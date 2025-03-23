import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI, OpenAI

from ... import provider


@define
class OpenRouter(provider.Stream):
    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    def prepare_input(
        self,
        site_url: str | None = None,
        app_name: str | None = None,
        **kwargs,
    ) -> dict:
        return {
            "extra_headers": {
                "HTTP-Referer": site_url or "",
                "X-Title": app_name or "",
            },
            **kwargs,
        }

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        kwargs = self.prepare_input(**kwargs)
        r = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        assert r.usage
        c = r.choices[0].message.content
        assert c
        return provider.Result(
            c,
            provider.Usage(
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            ),
            r,
        )

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        kwargs = self.prepare_input(**kwargs)
        r = await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        assert r.usage
        c = r.choices[0].message.content
        assert c
        return provider.Result(
            c,
            provider.Usage(
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            ),
            r,
        )

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        kwargs = self.prepare_input(**kwargs)
        for r in self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            c = r.choices[0].delta.content
            assert c
            u = r.usage
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
        kwargs = self.prepare_input(**kwargs)
        async for r in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            c = r.choices[0].delta.content
            assert c
            u = r.usage
            assert u
            yield provider.Result(
                c,
                provider.Usage(
                    u.prompt_tokens,
                    u.completion_tokens,
                ),
                r,
            )
