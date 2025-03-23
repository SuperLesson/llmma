import typing as t

import anthropic
from attrs import define, field

from ... import provider


@define
class Anthropic(provider.Stream):
    client: anthropic.Anthropic | anthropic.AnthropicBedrock = field(init=False)
    async_client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock = field(init=False)

    def __attrs_post_init__(self):
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def _count_tokens(self, content: str) -> int:
        return self.client.messages.count_tokens(
            model=self.model,
            messages=t.cast(t.Any, provider.msg_from_raw(content)),
        ).input_tokens

    @staticmethod
    def _prepare_messages(messages: list[dict]) -> tuple[list[dict], str | None]:
        system = next((m["content"] for m in reversed(messages) if m["role"] == "system"), None)
        if not system:
            return messages, None
        messages = [m for m in messages if m["role"] != "system"]
        return messages, system

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        messages, system = self._prepare_messages(messages)
        r = self.client.messages.create(
            model=self.model,
            messages=t.cast(t.Any, messages),
            system=system or anthropic.NOT_GIVEN,
            stream=False,
            **kwargs,
        )
        c = r.content[0]
        assert c.type == "text"
        return provider.Result(c.text, provider.Usage(r.usage.input_tokens, r.usage.output_tokens), r)

    async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
        messages, system = self._prepare_messages(messages)
        r = await self.async_client.messages.create(
            model=self.model,
            messages=t.cast(t.Any, messages),
            system=system or anthropic.NOT_GIVEN,
            stream=False,
            **kwargs,
        )
        c = r.content[0]
        assert c.type == "text"
        return provider.Result(c.text, provider.Usage(r.usage.input_tokens, r.usage.output_tokens), r)

    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
        messages, system = self._prepare_messages(messages)
        with self.client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            for e in stream_manager:
                if e.type == "message_stop":
                    r = e.message
                    c = r.content[0]
                    assert c.type == "text"
                    yield provider.Result(c.text, provider.Usage(r.usage.input_tokens, r.usage.output_tokens), r)

    async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
        messages, system = self._prepare_messages(messages)
        async with self.async_client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            async for e in stream_manager:
                if e.type == "message_stop":
                    r = e.message
                    c = r.content[0]
                    assert c.type == "text"
                    yield provider.Result(c.text, provider.Usage(r.usage.input_tokens, r.usage.output_tokens), r)
