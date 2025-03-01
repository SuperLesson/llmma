import typing as t

from attrs import define, field
from ollama import AsyncClient, Client

from .base import ModelInfo, StreamProvider


def get_provider(host: str, context_limit: int = 4096, output_limit: int = 2048, **kwargs) -> type:
    model_info = {}
    client = Client(host=host).list()
    models = client.get("models", [])
    if not models:
        msg = "Could not retrieve any models from Ollama"
        raise ValueError(msg)

    for model in models:
        name = model["model"]
        # Ollama models are free to use locally
        model_info[name] = ModelInfo(
            prompt_cost=0.0,
            completion_cost=0.0,
            context_limit=context_limit,
            output_limit=output_limit,
        )

    @define
    class OllamaProvider(StreamProvider):
        api_key = ""
        MODEL_INFO = model_info

        client: Client = field(init=False)
        async_client: AsyncClient = field(init=False)

        def __attrs_post_init__(self):
            self.client = Client(host=host, **kwargs)
            self.async_client = AsyncClient(host=host, **kwargs)

        def _count_tokens(self, content: str) -> int:
            """Estimate token count using simple word-based heuristic"""
            # Rough estimation: split on whitespace
            # TODO: also split on punctuation
            return len(content.split())

        def complete(self, messages: list[dict], **kwargs) -> dict:
            try:
                response = self.client.chat(model=self.model, messages=messages, stream=False, **kwargs)
            except Exception as e:
                msg = f"Ollama completion failed: {str(e)}"
                raise RuntimeError(msg) from e

            return {
                "completion": response.message.content,
                "prompt_tokens": response.prompt_eval_count,
                "completion_tokens": response.eval_count,
            }

        async def acomplete(self, messages: list[dict], **kwargs) -> dict:
            try:
                response = await self.async_client.chat(model=self.model, messages=messages, stream=False, **kwargs)
            except Exception as e:
                msg = f"Ollama completion failed: {str(e)}"
                raise RuntimeError(msg) from e

            return {
                "completion": response.message.content,
                "prompt_tokens": response.prompt_eval_count,
                "completion_tokens": response.eval_count,
            }

        def complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[str]:
            for chunk in self.client.chat(model=self.model, messages=messages, stream=True, **kwargs):
                if c := chunk["message"]["content"]:
                    yield c

        async def acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[str]:
            async for chunk in await self.async_client.chat(
                model=self.model, messages=messages, stream=True, **kwargs
            ):
                if c := chunk["message"]["content"]:
                    yield c

    return OllamaProvider
