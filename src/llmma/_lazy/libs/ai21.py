import ai21
from ai21.models.chat import ChatMessage
from ai21.tokenizers import get_tokenizer
from attrs import define, field

from ... import provider


@define
class AI21(provider.Sync):
    # per million tokens
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

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        data = self.prepare_input(**kwargs)
        r = self.client.chat.completions.create(
            model=self.model, messages=[ChatMessage(**ms) for ms in messages], stream=False, **data
        )
        text = r.choices[0].message.content
        assert text
        return provider.Result(
            text,
            provider.Usage(r.usage.prompt_tokens, r.usage.completion_tokens),
            r,
        )

    # TODO: async and stream support
