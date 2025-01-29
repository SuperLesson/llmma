import itertools as it
import math
import os
from dataclasses import dataclass

import google.generativeai as genai

from .base import ModelInfo, SyncProvider, msg_as_str


@dataclass
class GoogleGenAIProvider(SyncProvider):
    # cost is per million tokens
    MODEL_INFO = {
        # no support for "textembedding-gecko"
        "chat-bison-genai": ModelInfo(prompt_cost=0.5, completion_cost=0.5, context_limit=0),
        "text-bison-genai": ModelInfo(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
        "gemini-1.5-pro": ModelInfo(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
        "gemini-1.5-pro-latest": ModelInfo(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
        "gemini-1.5-flash": ModelInfo(prompt_cost=0.075, completion_cost=0.3, context_limit=128000),
        "gemini-1.5-flash-latest": ModelInfo(prompt_cost=0.075, completion_cost=0.3, context_limit=128000),
        "gemini-1.5-pro-exp-0801": ModelInfo(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
    }

    def __post_init__(self):
        super().__post_init__()
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")

        self.client = genai.configure(api_key=api_key)  # type: ignore

        model = self.model
        if model.startswith("text-"):
            self.client = genai.generate_text  # type: ignore[private-import]
            self.mode = "text"
        else:
            self.client = genai.GenerativeModel(model)  # type: ignore[private-import]
            self.mode = "chat"

    def _count_tokens(self, content: list[dict]) -> int:
        return self.client.count_tokens(msg_as_str(content)).total_tokens  # type: ignore[private-import]

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens"):
            kwargs["max_output_tokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], **kwargs) -> dict:
        prompts = [
            {"role": parts[0]["role"], "parts": [p["content"] for p in parts]}
            for parts in (list(ps) for _, ps in it.groupby(messages, key=lambda x: x["role"]))
        ]
        kwargs = self.prepare_input(**kwargs)
        response = self.client.generate_content(prompts, **kwargs)  # type: ignore[private-import]
        completion = response.text if self.mode == "chat" else " ".join([r.text for r in response])

        prompt_tokens = len(msg_as_str(messages))
        completion_tokens = len(completion)
        cost = ((prompt_tokens * self.info.prompt_cost) + (completion_tokens * self.info.completion_cost)) / 1_000_000

        # fast approximation. We could call count_message_tokens() but this will add latency
        prompt_tokens = math.ceil((prompt_tokens + 1) / 4)
        completion_tokens = math.ceil((completion_tokens + 1) / 4)
        total_tokens = math.ceil(prompt_tokens + completion_tokens)

        return {
            "completion": completion,
            "model": self.model,
            "tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }
