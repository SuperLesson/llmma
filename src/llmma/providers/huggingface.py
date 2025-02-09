from attrs import define, field
from huggingface_hub import InferenceClient

from .base import ModelInfo, SyncProvider


@define
class HuggingfaceHubProvider(SyncProvider):
    MODEL_INFO = {
        "hf_deepseek-r1": ModelInfo(
            hf_repo="deepseek-ai/DeepSeek-R1",
            prompt_cost=0,
            completion_cost=0,
            context_limit=128_000,
        ),
        "hf_deepseek-v3": ModelInfo(
            hf_repo="deepseek-ai/DeepSeek-V3",
            prompt_cost=0,
            completion_cost=0,
            context_limit=128_000,
        ),
        "hf_pythia": ModelInfo(
            hf_repo="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
        ),
        "hf_falcon40b": ModelInfo(
            hf_repo="tiiuae/falcon-40b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_falcon7b": ModelInfo(
            hf_repo="tiiuae/falcon-7b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_mptinstruct": ModelInfo(
            hf_repo="mosaicml/mpt-7b-instruct",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_mptchat": ModelInfo(
            hf_repo="mosaicml/mpt-7b-chat",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_llava": ModelInfo(
            hf_repo="liuhaotian/LLaVA-Lightning-MPT-7B-preview",
            prompt_cost=0,
            completion_cost=0,
            context_limit=2048,
            local=True,
        ),
        "hf_dolly": ModelInfo(
            hf_repo="databricks/dolly-v2-12b",
            prompt_cost=0,
            completion_cost=0,
            context_limit=-1,
            local=True,
        ),
        "hf_vicuna": ModelInfo(
            hf_repo="CarperAI/stable-vicuna-13b-delta",
            prompt_cost=0,
            completion_cost=0,
            context_limit=-1,
        ),
    }

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
