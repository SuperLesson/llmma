import typing as t

import cohere
from attrs import define, field

from ... import provider


@define
class Cohere(provider.Stream):
    client: cohere.ClientV2 = field(init=False)
    async_client: cohere.AsyncClientV2 = field(init=False)

    def __attrs_post_init__(self):
        api_key = self.api_key
        self.client = cohere.ClientV2(api_key)
        self.async_client = cohere.AsyncClientV2(api_key)

    def _count_tokens(self, content: str) -> int:
        return len(self.client.tokenize(text=content, model=self.model).tokens)

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        r = self.client.chat(
            model=self.model,
            messages=t.cast(t.Any, messages),
            **kwargs,
        )
        usage = r.usage
        assert usage
        tokens = usage.tokens
        assert tokens
        i = tokens.input_tokens
        o = tokens.output_tokens
        assert i
        assert o
        msgs = r.message.content
        assert msgs
        c = msgs[0].text
        return provider.Result(
            c,
            provider.Usage(
                int(i),
                int(o),
            ),
            r,
        )

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        async with self.async_client as client:
            r = await client.chat(
                model=self.model,
                messages=t.cast(t.Any, messages),
                **kwargs,
            )
        usage = r.usage
        assert usage
        tokens = usage.tokens
        assert tokens
        i = tokens.input_tokens
        o = tokens.output_tokens
        assert i
        assert o
        msgs = r.message.content
        assert msgs
        c = msgs[0].text
        return provider.Result(
            c,
            provider.Usage(
                int(i),
                int(o),
            ),
            r,
        )

    @staticmethod
    def _parse_stream(
        r: cohere.types.streamed_chat_response_v2.StreamedChatResponseV2,
    ) -> tuple[provider.Result | None, bool]:
        if r.type == "message-end":
            d = r.delta
            assert d
            u = d.usage
            assert u
            tok = u.tokens
            assert tok
            i = tok.input_tokens
            o = tok.output_tokens
            assert i
            assert o
            return provider.Result(
                "",
                provider.Usage(
                    int(i),
                    int(o),
                ),
                r,
            ), False
        if r.type == "content-end" or r.type == "tool-call-end" or r.type == "citation-end":
            return None, False
        if r.type == "debug" or r.type == "tool-plan-delta":
            return None, True
        if r.type == "message-start" or r.type == "citation-start":
            return None, True
        d = r.delta
        assert d
        msg = d.message
        assert msg
        c = msg.content.text  # type: ignore
        assert c
        return (
            provider.Result(
                c,
                provider.Usage(0, 0),
                r,
            ),
            True,
        )

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        for r in self.client.chat_stream(
            model=self.model,
            messages=t.cast(t.Any, messages) ** kwargs,
        ):
            r, ok = self._parse_stream(r)
            if r:
                yield r
            if not ok:
                break

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        async with self.async_client as client:
            async for r in client.chat_stream(
                model=self.model,
                messages=t.cast(t.Any, messages) ** kwargs,
            ):
                r, ok = self._parse_stream(r)
                if r:
                    yield r
                if not ok:
                    break
