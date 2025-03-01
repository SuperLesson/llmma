import typing as t

import cohere
from attrs import define, field

from .base import ModelInfo, StreamProvider, msg_as_str


@define
class CohereProvider(StreamProvider):
    MODEL_INFO = {
        "command": ModelInfo(prompt_cost=15.0, completion_cost=15, context_limit=2048),
        "command-nightly": ModelInfo(prompt_cost=15.0, completion_cost=15, context_limit=4096),
    }

    client: cohere.Client = field(init=False)
    async_client: cohere.AsyncClient = field(init=False)

    def __attrs_post_init__(self):
        api_key = self.api_key
        self.client = cohere.Client(api_key)
        self.async_client = cohere.AsyncClient(api_key)

    def _count_tokens(self, content: str) -> int:
        return len(self.client.tokenize(text=content, model=self.model).tokens)

    def complete(self, messages: list[dict], **kwargs) -> dict:
        return {
            "completion": self.client.chat(
                model=self.model,
                message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                **kwargs,
            ).text
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        async with self.async_client as client:
            return {
                "completion": (
                    await client.chat(
                        model=self.model,
                        message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                        **kwargs,
                    )
                ).text
            }

    def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
        for token in self.client.chat_stream(
            model=self.model,
            message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
            **kwargs,
        ):
            yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, token).text

    async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
        async with self.async_client as client:
            async for r in client.chat_stream(
                model=self.model,
                message=messages[0]["content"] if len(messages) == 1 else msg_as_str(messages),
                **kwargs,
            ):
                yield t.cast(cohere.types.streamed_chat_response.TextGenerationStreamedChatResponse, r).text
