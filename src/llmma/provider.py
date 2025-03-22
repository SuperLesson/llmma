import time
import typing as t
from contextlib import contextmanager

from attrs import define, field

from .model import ProviderInfo as Info


def msg_from_raw(cont: str | dict | list[dict], role: str = "user") -> list[dict]:
    if isinstance(cont, str):
        return [{"content": cont, "role": role}]
    if isinstance(cont, dict):
        return [cont]
    return cont


def msg_as_str(cont: list[dict]) -> str:
    return ";".join([f"{message['role'].capitalize()}: {message['content']}" for message in cont])


Provider = t.Union["Async", "Stream", "Sync"]


@define
class Sync:
    """Base class for all providers.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    api_key: str = field(repr=False)
    model: str
    info: Info
    latency: float | None = None
    tokenizer: t.Any = field(init=False)

    @contextmanager
    def track_latency(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency = round(time.perf_counter() - start, 2)

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = ((prompt_tokens * self.info.prompt_cost) + (completion_tokens * self.info.completion_cost)) / 1_000_000
        return round(cost, 5)

    def _count_tokens(self, content: str) -> int:
        raise

    def count_tokens(self, content: str | dict | list[dict]) -> int:
        return sum(self._count_tokens(c["content"]) for c in msg_from_raw(content))

    def complete(self, messages: list[dict], **kwargs) -> dict:
        raise


@define
class Async(Sync):
    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        raise


@define
class Stream(Async):
    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        raise

    def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        raise
