import os
import typing as t

import tiktoken
from aleph_alpha_client import AsyncClient, Client, CompletionRequest, Prompt
from attrs import define, field

from ... import provider


@define
class AlephAlpha(provider.Async):
    client: Client = field(init=False)
    async_client: AsyncClient = field(init=False)
    host: str = field(factory=lambda: os.getenv("ALEPHALPHA_HOST", ""))

    def __attrs_post_init__(self):
        if not self.host:
            msg = "ALEPHALPHA_HOST environment variable is required"
            raise Exception(msg)
        self.client = Client(self.api_key, self.host)
        self.async_client = AsyncClient(self.api_key, self.host)
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def _count_tokens(self, content: str) -> int:
        return len(self.tokenizer.encode(content))

    @staticmethod
    def prepare_input(
        messages: list[dict],
        **kwargs,
    ) -> CompletionRequest:
        text = str(messages[0]["content"]) if len(messages) == 1 else provider.msg_as_str(messages)
        if max_tokens := kwargs.pop("max_tokens", None):
            kwargs["maximum_tokens"] = max_tokens

        return CompletionRequest(
            prompt=Prompt.from_text(text),
            **kwargs,
        )

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.complete(request=self.prepare_input(messages, **kwargs), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "prompt_tokens": response.num_tokens_prompt_total,
            "completion_tokens": response.num_tokens_generated,
        }

    async def acomplete(self, messages: list[dict], **kwargs) -> dict:
        async with self.async_client as client:
            response = await client.complete(request=self.prepare_input(messages, **kwargs), model=self.model)
        return {
            "completion": t.cast(str, response.completions[0].completion),
            "prompt_tokens": response.num_tokens_prompt_total,
            "completion_tokens": response.num_tokens_generated,
        }
