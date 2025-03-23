import time
import typing as t
from contextlib import contextmanager

from attrs import define, field

from . import result
from .model import Info, Usage
from .result import Raw as Result


def msg_from_raw(cont: str | dict | list[dict], role: str = "user") -> list[dict]:
    if isinstance(cont, str):
        return [{"content": cont, "role": role}]
    if isinstance(cont, dict):
        return [cont]
    return cont


def msg_as_str(cont: list[dict]) -> str:
    return ";".join([f"{message['role'].capitalize()}: {message['content']}" for message in cont])


def as_messages(
    prompt: str | dict | list[dict],
    history: list[dict] | None = None,
    system_message: str | None = None,
) -> list[dict]:
    messages = history or []
    if system_message:
        messages.extend(msg_from_raw(system_message, "system"))
    messages.extend(msg_from_raw(prompt))
    return messages


Provider = t.Union["Async", "Stream", "Sync"]


@define
class Sync:
    """Base class for all providers.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    api_key: str = field(repr=False)
    model: str
    info: Info
    latency: float = field(default=0)
    tokenizer: t.Any = field(init=False)

    @contextmanager
    def track_latency(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency = round(time.perf_counter() - start, 2)

    def cost(self, usage: Usage) -> float:
        cost = (
            (usage.prompt_tokens * self.info.prompt_cost) + (usage.completion_tokens * self.info.completion_cost)
        ) / 1_000_000
        return round(cost, 5)

    def _count_tokens(self, content: str) -> int:
        raise

    def count_tokens(self, content: str | dict | list[dict]) -> int:
        return sum(self._count_tokens(c["content"]) for c in msg_from_raw(content))

    def _complete(self, messages: list[dict], **kwargs) -> Result:
        raise

    def complete(
        self,
        prompt: str | dict | list[dict],
        history: list[dict] | None = None,
        system_message: str | None = None,
        **kwargs: t.Any,
    ) -> result.Chat:
        messages = as_messages(prompt, history, system_message)
        with self.track_latency():
            r = self._complete(messages, **kwargs)

        kwargs["messages"] = messages
        return result.Chat(
            input=kwargs,
            latency=self.latency,
            text=r.text,
            usage=r.usage,
            raw=r.raw,
        )


@define
class Async(Sync):
    async def _acomplete(self, messages: list[dict], **kwargs) -> Result:
        raise

    async def acomplete(
        self,
        prompt: str | dict | list[dict],
        history: list[dict] | None = None,
        system_message: str | None = None,
        **kwargs: t.Any,
    ) -> result.Chat:
        messages = as_messages(prompt, history, system_message)
        with self.track_latency():
            r = await self._acomplete(messages, **kwargs)

        kwargs["messages"] = messages
        return result.Chat(
            input=kwargs,
            latency=self.latency,
            text=r.text,
            usage=r.usage,
            raw=r.raw,
        )


@define
class Stream(Async):
    def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[Result]:
        raise

    def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[Result]:
        raise

    def complete_stream(
        self,
        prompt: str | dict | list[dict],
        history: list[dict] | None = None,
        system_message: str | None = None,
        **kwargs: t.Any,
    ) -> result.Stream:
        messages = as_messages(prompt, history, system_message)

        # TODO: test
        def complete() -> t.Iterator[result.Chunk]:
            rs = self._complete_stream(messages, **kwargs)
            with self.track_latency():
                x = next(rs)
            yield result.Chunk(x.text, x.usage, x.raw, self.latency)
            while x:
                with self.track_latency():
                    x = next(rs)
                yield result.Chunk(x.text, x.usage, x.raw, self.latency)

        kwargs["messages"] = messages
        return result.Stream(
            input=kwargs,
            raw_stream=complete(),
        )

    def acomplete_stream(
        self,
        prompt: str | dict | list[dict],
        history: list[dict] | None = None,
        system_message: str | None = None,
        **kwargs: t.Any,
    ) -> result.AStream:
        messages = as_messages(prompt, history, system_message)

        # TODO: test
        async def acomplete() -> t.AsyncIterator[result.Chunk]:
            rs = self._acomplete_stream(messages, **kwargs)
            with self.track_latency():
                x = await anext(rs)
            yield result.Chunk(x.text, x.usage, x.raw, self.latency)
            while x:
                with self.track_latency():
                    x = await anext(rs)
                yield result.Chunk(x.text, x.usage, x.raw, self.latency)

        kwargs["messages"] = messages
        return result.AStream(input=kwargs, raw_stream=acomplete())
