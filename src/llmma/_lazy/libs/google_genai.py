import itertools as it
import math
import os
import typing as t

import google.generativeai as genai
from attrs import define, field

from ... import provider


@define
class GoogleGenAI(provider.Sync):
    client: t.Any = field(init=False)
    mode: str = field(init=False)

    def __attrs_post_init__(self):
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")

        self.client = genai.configure(api_key=api_key)  # type: ignore

        model = self.model
        if model.startswith("text-"):
            self.client = genai.generate_text  # type: ignore[private-import]
            self.mode = "text"
        else:
            self.client = genai.GenerativeModel(model)  # type: ignore[private-import]
            self.mode = "chat"

    def _count_tokens(self, content: str) -> int:
        return self.client.count_tokens(content).total_tokens  # type: ignore[private-import]

    @staticmethod
    def prepare_input(
        **kwargs,
    ) -> dict:
        if max_tokens := kwargs.pop("max_tokens"):
            kwargs["max_output_tokens"] = max_tokens
        return kwargs

    def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
        prompts = [
            {"role": parts[0]["role"], "parts": [p["content"] for p in parts]}
            for parts in (list(ps) for _, ps in it.groupby(messages, key=lambda x: x["role"]))
        ]
        kwargs = self.prepare_input(**kwargs)
        r = self.client.generate_content(prompts, **kwargs)  # type: ignore[private-import]
        c = r.text if self.mode == "chat" else " ".join([r.text for r in r])
        return provider.Result(
            c,
            # fast approximation. We could call count_message_tokens() but this will add latency
            provider.Usage(
                math.ceil((len(provider.msg_as_str(messages)) + 1) / 4),
                math.ceil((len(c) + 1) / 4),
            ),
            r,
        )
