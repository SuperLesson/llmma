import typing as t

import anthropic
from attrs import define, field

from .base import ModelInfo, StreamProvider, msg_from_raw


@define
class AnthropicProvider(StreamProvider):
    MODEL_INFO = {
        "claude-3-haiku-latest": ModelInfo(
            prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-opus-latest": ModelInfo(
            prompt_cost=15.00, completion_cost=75, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-5-haiku-latest": ModelInfo(
            prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=8_192
        ),
        "claude-3-sonnet-latest": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
        ),
        "claude-3-5-sonnet-latest": ModelInfo(
            prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=8_192
        ),
    }

    client: anthropic.Anthropic | anthropic.AnthropicBedrock = field(init=False)
    async_client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock = field(init=False)

    def __attrs_post_init__(self):
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def _count_tokens(self, content: str) -> int:
        return self.client.messages.count_tokens(
            model=self.model,
            messages=t.cast(t.Any, msg_from_raw(content)),
        ).input_tokens

    @staticmethod
    def _prepare_messages(messages: list[dict]) -> tuple[list[dict], str | None]:
        system = next((m["content"] for m in reversed(messages) if m["role"] == "system"), None)
        if not system:
            return messages, None
        messages = [m for m in messages if m["role"] != "system"]
        return messages, system

    def complete(self, messages: list[dict], **kwargs) -> dict:
        messages, system = self._prepare_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            messages=t.cast(t.Any, messages),
            system=system or anthropic.NOT_GIVEN,
            stream=False,
            **kwargs,
        )
        assert response.content[0].type == "text"
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        messages, system = self._prepare_messages(messages)
        response = await self.async_client.messages.create(
            model=self.model,
            messages=t.cast(t.Any, messages),
            system=system or anthropic.NOT_GIVEN,
            stream=False,
            **kwargs,
        )
        assert response.content[0].type == "text"
        return {
            "completion": response.content[0].text,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        messages, system = self._prepare_messages(messages)
        with self.client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            yield from stream_manager.text_stream

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        messages, system = self._prepare_messages(messages)
        async with self.async_client.messages.stream(
            model=self.model, messages=t.cast(t.Any, messages), system=system or anthropic.NOT_GIVEN, **kwargs
        ) as stream_manager:
            async for text in stream_manager.text_stream:
                yield text
