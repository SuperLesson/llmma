import typing as t

import tiktoken
from attrs import define, field
from mistralai import Mistral

from .base import ModelInfo, StreamProvider, msg_as_str


@define
class MistralProvider(StreamProvider):
    MODEL_INFO = {
        "mistral-tiny": ModelInfo(prompt_cost=0.25, completion_cost=0.25, context_limit=32_000),
        # new endpoint for mistral-tiny, mistral-tiny will be deprecated in ~June 2024
        "open-mistral-7b": ModelInfo(prompt_cost=0.25, completion_cost=0.25, context_limit=32_000),
        "mistral-small": ModelInfo(prompt_cost=0.7, completion_cost=0.7, context_limit=32_000),
        # new endpoint for mistral-small, mistral-small will be deprecated in ~June 2024
        "open-mixtral-8x7b": ModelInfo(prompt_cost=0.7, completion_cost=0.7, context_limit=32_000),
        "mistral-small-latest": ModelInfo(prompt_cost=2.0, completion_cost=6.0, context_limit=32_000),
        "mistral-medium-latest": ModelInfo(prompt_cost=2.7, completion_cost=8.1, context_limit=32_000),
        "mistral-large-latest": ModelInfo(prompt_cost=3.0, completion_cost=9.0, context_limit=32_000),
        "open-mistral-nemo": ModelInfo(prompt_cost=0.3, completion_cost=0.3, context_limit=32_000),
    }

    client: Mistral = field(init=False)

    def __attrs_post_init__(self):
        self.client = Mistral(api_key=self.api_key)

    def _count_tokens(self, content: list[dict]) -> int:
        # TODO: update after Mistrar support count token in their SDK
        # use gpt 3.5 turbo for estimation now
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        formatting_token_count = 4
        messages = content
        messages_text = [msg_as_str([message]) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token in tokens:
            n_tokens = len(token) + formatting_token_count
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

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
