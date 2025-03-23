import typing as t

from attrs import define, field
from ollama import AsyncClient, Client

from ... import provider


def get_provider(
    host: str, context_limit: int = 4096, output_limit: int = 2048, **kwargs
) -> tuple[type, dict[str, provider.Info]]:
    model_info = {}
    client = Client(host=host).list()
    models = client.get("models", [])
    if not models:
        msg = "Could not retrieve any models from Ollama"
        raise ValueError(msg)

    for model in models:
        name = model["model"]
        # Ollama models are free to use locally
        model_info[name] = provider.Info(
            prompt_cost=0.0,
            completion_cost=0.0,
            context_limit=context_limit,
            output_limit=output_limit,
        )

    @define
    class Ollama(provider.Stream):
        api_key = ""
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

        def _complete(self, messages: list[dict], **kwargs) -> provider.Result:
            try:
                r = self.client.chat(model=self.model, messages=messages, stream=False, **kwargs)
            except Exception as e:
                msg = f"Ollama completion failed: {str(e)}"
                raise RuntimeError(msg) from e

            c = r.message.content
            assert c
            pt = r.prompt_eval_count
            assert pt
            ct = r.eval_count
            assert ct
            return provider.Result(
                c,
                provider.Usage(
                    pt,
                    ct,
                ),
                r,
            )

        async def _acomplete(self, messages: list[dict], **kwargs) -> provider.Result:
            try:
                r = await self.async_client.chat(model=self.model, messages=messages, stream=False, **kwargs)
            except Exception as e:
                msg = f"Ollama completion failed: {str(e)}"
                raise RuntimeError(msg) from e

            c = r.message.content
            assert c
            pt = r.prompt_eval_count
            assert pt
            ct = r.eval_count
            assert ct
            return provider.Result(
                c,
                provider.Usage(
                    pt,
                    ct,
                ),
                r,
            )

        def _complete_stream(self, messages: list[dict], **kwargs) -> t.Iterator[provider.Result]:
            for r in self.client.chat(model=self.model, messages=messages, stream=True, **kwargs):
                c = r.message.content
                assert c
                pt = r.prompt_eval_count
                assert pt
                ct = r.eval_count
                assert ct
                yield provider.Result(
                    c,
                    provider.Usage(
                        pt,
                        ct,
                    ),
                    r,
                )

        async def _acomplete_stream(self, messages: list[dict], **kwargs) -> t.AsyncIterator[provider.Result]:
            async for r in await self.async_client.chat(model=self.model, messages=messages, stream=True, **kwargs):
                c = r.message.content
                assert c
                pt = r.prompt_eval_count
                assert pt
                ct = r.eval_count
                assert ct
                yield provider.Result(
                    c,
                    provider.Usage(
                        pt,
                        ct,
                    ),
                    r,
                )

    return Ollama, model_info
