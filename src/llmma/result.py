import json
import typing as t

from attrs import define, field

from .provider import Provider


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
class Chat(ABCResult):
    text: str
    _meta: dict = field(factory=dict)
    function_call: dict = field(factory=dict)


@define
class Stream(ABCResult):
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
class AStream(ABCResult):
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
    async def stream(self) -> t.AsyncIterator[Chat]:
        if not self._stream_exhausted:
            async for item in self._stream:
                self._streamed_text.append(item)
                yield item
            self._stream_exhausted = True
        else:
            for r in self._streamed_text:
                yield r
