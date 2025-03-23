import typing as t

from attrs import define, field
from pydantic import BaseModel

from . import model


@define
class Raw:
    text: str
    usage: model.Usage
    raw: BaseModel | dict


@define
class Chunk(Raw):
    latency: float


@define
class Chat(Raw):
    input: dict
    latency: float


@define
class Stream:
    input: dict
    raw_stream: t.Iterator[Chunk]
    streamed: list[Chunk] = field(factory=list)
    exhausted: bool = False

    @property
    def stream(self) -> t.Iterator[Chunk]:
        if not self.exhausted:
            while c := next(self.raw_stream, None):
                self.streamed.append(c)
                yield c
        else:
            yield from self.streamed


@define
class AStream:
    input: dict
    raw_stream: t.AsyncIterator[Chunk]
    streamed: list[Chunk] = field(factory=list)
    exhausted: bool = False

    @property
    async def stream(self) -> t.AsyncIterator[Chunk]:
        if not self._stream_exhausted:
            async for c in self.raw_stream:
                self.streamed.append(c)
                yield c
            self._stream_exhausted = True
        else:
            for r in self.streamed:
                yield r
