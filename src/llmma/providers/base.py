import json
import time
import typing as t
import warnings
from contextlib import contextmanager

from attrs import define, field


def msg_from_raw(cont: str | dict | list[dict], role: str = "user") -> list[dict]:
    if isinstance(cont, str):
        return [{"content": cont, "role": role}]
    if isinstance(cont, dict):
        return [cont]
    return cont


def msg_as_str(cont: list[dict]) -> str:
    return ";".join([f"{message['role'].capitalize()}: {message['content']}" for message in cont])


Provider = t.Union["AsyncProvider", "StreamProvider", "SyncProvider"]


@define
class ABCResult:
    provider: Provider
    model_inputs: dict
    function_call: dict = field(init=False, factory=dict)
    _meta: dict = field(init=False, factory=dict)
    text: str = field(init=False, default="")

    @property
    def completion_tokens(self) -> int:
        if not (completion_tokens := self._meta.get("completion_tokens")):
            completion_tokens = self.provider.count_tokens(self.text)
            self._meta["completion_tokens"] = completion_tokens
        return completion_tokens

    @property
    def prompt_tokens(self) -> int:
        if not (prompt_tokens := self._meta.get("prompt_tokens")):
            prompt_tokens = self.provider.count_tokens(
                self.model_inputs.get("prompt") or self.model_inputs.get("messages") or ""
            )
            self._meta["prompt_tokens"] = prompt_tokens
        return prompt_tokens

    @property
    def tokens(self) -> int:
        return self.completion_tokens + self.prompt_tokens

    @property
    def cost(self) -> float:
        if not (cost := self._meta.get("cost")):
            cost = self.provider.compute_cost(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )
            self._meta["cost"] = cost
        return cost

    @property
    def meta(self) -> dict:
        return {
            "model": self.provider.model,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "latency": self._meta.get("latency"),
            **self._meta,
        }

    def to_json(self) -> str:
        model_inputs = self.model_inputs
        # remove https related params
        model_inputs.pop("headers", None)
        model_inputs.pop("request_timeout", None)
        model_inputs.pop("aiosession", None)
        return json.dumps(
            {
                "text": self.text,
                "meta": self.meta,
                "model_inputs": model_inputs,
                "provider": str(self.provider),
                "function_call": self.function_call,
            }
        )


@define
class Result(ABCResult):
    text: str
    _meta: dict = field(factory=dict)
    function_call: dict = field(factory=dict)


@define
class StreamResult(ABCResult):
    _stream: t.Iterator
    _streamed_text: list = field(factory=list)
    _meta: dict = field(factory=dict)
    function_call: dict = field(factory=dict)

    def __attrs_post_init__(self):
        _ = all(self.stream)
        self.text = "".join(self._streamed_text)

    @property
    def stream(self):
        while t := next(self._stream, None):
            self._streamed_text.append(t)
            yield t


@define
class AsyncStreamResult(ABCResult):
    _stream: t.AsyncIterable
    _stream_exhausted: bool = False
    _streamed_text: list = field(factory=list)
    _meta: dict = field(factory=dict)
    function_call: dict = field(factory=dict)

    def __attrs_post_init__(self):
        if not self._stream_exhausted:
            msg = "Please finish streaming the result."
            raise RuntimeError(msg)
        self.text = "".join(self._streamed_text)

    @property
    async def stream(self) -> t.AsyncIterator[Result]:
        if not self._stream_exhausted:
            async for item in self._stream:
                self._streamed_text.append(item)
                yield item
            self._stream_exhausted = True
        else:
            for r in self._streamed_text:
                yield r


@define
class ModelInfo:
    prompt_cost: float
    completion_cost: float
    context_limit: int
    output_limit: int | None = None
    limit_per_minute: int | None = None
    image_input_cost: float | None = None
    hf_repo: str | None = None
    chat: bool = True
    local: bool = False
    quirks: dict[str, t.Any] = field(factory=dict)

    def __attrs_post_init__(self):
        if self.output_limit is None:
            self.output_limit = self.context_limit // 2


@define
class SyncProvider:
    """Base class for all providers.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    api_key: str = field(repr=False)
    model: t.Any = field()
    latency: float | None = None
    MODEL_INFO: t.ClassVar[dict[str, ModelInfo]] = {}
    info: t.Any = field(init=False)
    tokenizer: t.Any = field(init=False)

    @model.default
    def _model_factory(self) -> str:
        return list(self.MODEL_INFO.keys())[0]

    @info.default
    def _info_factory(self) -> ModelInfo:
        if self.model not in self.MODEL_INFO:
            warnings.warn(f"no information about cost of the model: {self.model}", UserWarning, stacklevel=2)
            return ModelInfo(
                prompt_cost=1,
                completion_cost=1,
                context_limit=4096,
            )
        return self.MODEL_INFO[self.model]

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
class AsyncProvider(SyncProvider):
    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        raise


@define
class StreamProvider(AsyncProvider):
    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        raise

    def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        raise
