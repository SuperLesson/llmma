import typing as t

import tiktoken
import together
from attrs import define, field
from together import AsyncTogether
from together import Together as TogetherAPI

from ... import provider


@define
class Together(provider.Stream):
    client: TogetherAPI = field(init=False)
    async_client: AsyncTogether = field(init=False)

    def __attrs_post_init__(self):
        self.client = TogetherAPI(api_key=self.api_key)
        self.async_client = AsyncTogether(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        # Together uses the same tokenizer as OpenAI
        return len(self.tokenizer.encode(content))

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = t.cast(
            together.types.ChatCompletionResponse,
            self.client.chat.completions.create(model=self.model, messages=messages, stream=False, **kwargs),
        )
        assert response.choices
        assert response.choices[0].message
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        response = t.cast(
            together.types.ChatCompletionResponse,
            await self.async_client.chat.completions.create(model=self.model, messages=messages, **kwargs),
        )
        assert response.choices
        assert response.choices[0].message
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for chunk in self.client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs):
            chunk = t.cast(together.types.ChatCompletionChunk, chunk)
            assert chunk.choices
            assert chunk.choices[0].delta
            s = chunk.choices[0].delta.content
            assert s
            yield s

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async for chunk in t.cast(
            t.AsyncGenerator[together.types.ChatCompletionChunk, None],
            self.async_client.chat.completions.create(model=self.model, messages=messages, stream=True, **kwargs),
        ):
            chunk = t.cast(together.types.ChatCompletionChunk, chunk)
            assert chunk.choices
            assert chunk.choices[0].delta
            s = chunk.choices[0].delta.content
            assert s
            yield s
