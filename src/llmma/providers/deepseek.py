import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI, OpenAI

from .base import AsyncProvider, ModelInfo


@define
class DeepSeekProvider(AsyncProvider):
    MODEL_INFO = {
        "deepseek-chat": ModelInfo(prompt_cost=0.14, completion_cost=0.28, context_limit=64_000, output_limit=8_192),
        "deepseek-reasoner": ModelInfo(
            prompt_cost=0.14, completion_cost=2.19, context_limit=64_000, output_limit=8_192
        ),
    }

    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )

    def _count_tokens(self, content: list[dict]) -> int:
        # DeepSeek uses the same tokenizer as OpenAI
        enc = tiktoken.encoding_for_model(self.model)
        formatting_token_count = 4
        messages = content
        messages_text = ["".join(message.values()) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token, message in zip(tokens, messages, strict=False):
            n_tokens = len(token) + formatting_token_count
            if "name" in message:
                n_tokens += -1
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=self.model, stream=False, messages=t.cast(t.Any, messages), **kwargs
        )
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        response = await self.async_client.chat.completions.create(
            model=self.model, stream=False, messages=t.cast(t.Any, messages), **kwargs
        )
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
