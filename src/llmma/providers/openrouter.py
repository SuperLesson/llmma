import typing as t

import tiktoken
from attrs import define, field
from openai import AsyncOpenAI, OpenAI

from .base import ModelInfo, StreamProvider, msg_as_str


@define
class OpenRouterProvider(StreamProvider):
    MODEL_INFO = {
        "nvidia/llama-3.1-nemotron-70b-instruct": ModelInfo(
            prompt_cost=0.35,
            completion_cost=0.4,
            context_limit=131072,
        ),
        "x-ai/grok-2": ModelInfo(
            prompt_cost=5.0,
            completion_cost=10.0,
            context_limit=32768,
        ),
        "nousresearch/hermes-3-llama-3.1-405b:free": ModelInfo(
            prompt_cost=0.0,
            completion_cost=0.0,
            context_limit=8192,
        ),
        "google/gemini-flash-1.5-exp": ModelInfo(
            prompt_cost=0.0,
            completion_cost=0.0,
            context_limit=1000000,
        ),
        "liquid/lfm-40b": ModelInfo(
            prompt_cost=0.0,
            completion_cost=0.0,
            context_limit=32768,
        ),
        "mistralai/ministral-8b": ModelInfo(
            prompt_cost=0.1,
            completion_cost=0.1,
            context_limit=128000,
        ),
        "qwen/qwen-2.5-72b-instruct": ModelInfo(
            prompt_cost=0.35,
            completion_cost=0.4,
            context_limit=131072,
        ),
        "x-ai/grok-2-1212": ModelInfo(
            prompt_cost=2.0,
            completion_cost=10.0,
            context_limit=131072,
        ),
        "amazon/nova-pro-v1": ModelInfo(
            prompt_cost=0.8,
            completion_cost=3.2,
            context_limit=300000,
            image_input_cost=1.2,
        ),
        "qwen/qwq-32b-preview": ModelInfo(
            prompt_cost=0.12,
            completion_cost=0.18,
            context_limit=32768,
        ),
        "mistralai/mistral-large-2411": ModelInfo(
            prompt_cost=2.0,
            completion_cost=6.0,
            context_limit=128000,
        ),
    }
    client: OpenAI = field(init=False)
    async_client: AsyncOpenAI = field(init=False)

    def __attrs_post_init__(self):
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def _count_tokens(self, content: list[dict]) -> int:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        formatting_token_count = 4
        messages = content
        messages_text = [msg_as_str([message]) for message in messages]
        tokens = [enc.encode(t, disallowed_special=()) for t in messages_text]

        n_tokens_list = []
        for token, message in zip(tokens, messages, strict=False):
            n_tokens = len(token) + formatting_token_count
            if "name" in message:
                n_tokens += -1
            n_tokens_list.append(n_tokens)
        return sum(n_tokens_list)

    def prepare_input(
        self,
        site_url: str | None = None,
        app_name: str | None = None,
        **kwargs,
    ) -> dict:
        return {
            "extra_headers": {
                "HTTP-Referer": site_url or "",
                "X-Title": app_name or "",
            },
            **kwargs,
        }

    def complete(self, messages: list[dict], **kwargs) -> dict:
        kwargs = self.prepare_input(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=False, **kwargs
        )
        assert response.usage
        return {
            "completion": response.choices[0].message.content,
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
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        kwargs = self.prepare_input(**kwargs)
        async for chunk in await self.async_client.chat.completions.create(
            model=self.model, messages=t.cast(t.Any, messages), stream=True, **kwargs
        ):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
