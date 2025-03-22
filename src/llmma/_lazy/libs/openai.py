import json
import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI
from openai import OpenAI as OpenAIAPI

from ... import provider


@define
class OpenAI(provider.Stream):
    client: OpenAIAPI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAIAPI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    def prepare_input(
        self,
        **kwargs,
    ) -> dict:
        if not kwargs.get("max_completion_tokens") and self.info.quirks.get("use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", None)
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        if response.choices[0].message.function_call:
            function_call = {
                "name": response.choices[0].message.function_call.name,
                "arguments": json.loads(response.choices[0].message.function_call.arguments),
            }
            completion = ""
        else:
            function_call = {}
            completion = response.choices[0].message.content

        assert response.usage
        return {
            "completion": completion,
            "function_call": function_call,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )

        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        kwargs = self.prepare_input(**kwargs)
        for chunk in self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if c := chunk.choices[0].delta.content:
                yield c

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        kwargs = self.prepare_input(**kwargs)
        async for chunk in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if c := chunk.choices[0].delta.content:
                yield c
