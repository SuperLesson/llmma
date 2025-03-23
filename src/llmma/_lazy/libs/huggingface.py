from attrs import define, field
from huggingface_hub import InferenceClient

from ... import provider


@define
class Huggingface(provider.Sync):
    client: InferenceClient = field(init=False)

    def __attrs_post_init__(self):
        self.client = InferenceClient(self.info.hf_repo, token=self.api_key)

    def _count_tokens(self, content: str) -> int:
        raise

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        r = self.client.chat_completion(messages=messages, **kwargs)
        return provider.Result(
            r.choices[0].message,
            provider.Usage(
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
            ),
            r,
        )
