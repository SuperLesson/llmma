import ai21
from ai21.models.chat import ChatMessage
from ai21.tokenizers import get_tokenizer
from attrs import define, field

from .base import ModelInfo, SyncProvider


@define
class AI21Provider(SyncProvider):
    # per million tokens
    MODEL_INFO = {
        "j2-grande-instruct": ModelInfo(prompt_cost=10.0, completion_cost=10.0, context_limit=8192),
        "j2-jumbo-instruct": ModelInfo(prompt_cost=15.0, completion_cost=15.0, context_limit=8192),
    }

    client: ai21.AI21Client = field(init=False)

    def __attrs_post_init__(self):
        self.client = ai21.AI21Client(self.api_key)
        self.tokenizer = get_tokenizer(self.model + "-tokenizer")

    def _count_tokens(self, content: str) -> int:
        return self.tokenizer.count_tokens(content)

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens", None):
            kwargs["maxTokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        data = self.prepare_input(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model, messages=[ChatMessage(**ms) for ms in messages], stream=False, **data
        )
        return {
            "completion": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

    # TODO: async and stream support
