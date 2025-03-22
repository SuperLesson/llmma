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

    def complete(self, messages: list[dict], **kwargs) -> dict:
        response = self.client.chat_completion(messages=messages, **kwargs)
        return {
            "completion": response.choices[0].message,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
