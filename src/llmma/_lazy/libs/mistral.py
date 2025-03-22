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

    def complete(self, messages: list[dict], **kwargs) -> dict:
        with self.client as client:
            response = client.chat.complete(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        assert response.choices
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        async with self.client as client:
            response = await client.chat.complete_async(model=self.model, messages=t.cast(t.Any, messages), **kwargs)
        assert response.choices
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        with (
            self.client as client,
            client.chat.stream(model=self.model, messages=t.cast(t.Any, messages), **kwargs) as stream,
        ):
            for chunk in stream:
                assert chunk.data.choices
                if c := chunk.data.choices[0].delta.content:
                    yield t.cast(str, c)

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async with self.client as client:
            async for chunk in await client.chat.stream_async(
                model=self.model, messages=t.cast(t.Any, messages), **kwargs
            ):
                assert chunk.data.choices
                if c := chunk.data.choices[0].delta.content:
                    yield t.cast(str, c)
