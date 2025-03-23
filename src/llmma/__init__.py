from ._version import __version__  # noqa: I001

import asyncio
import os
import statistics
import typing as t
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger

from prettytable import PrettyTable

from .provider import Provider
from . import provider, result
from ._lazy import use_local_provider, PROVIDERS, Spec

__all__ = [
    "__version__",
    "Provider",
    "use_local_provider",
    "provider",
    "result",
    "PROVIDERS",
]

LOGGER = getLogger(__name__)


MultiResult = list[result.Chat]
AMultiResult = t.Awaitable[MultiResult]
Result = MultiResult | AMultiResult


DEFAULT_MODEL = os.getenv("LLMMA_DEFAULT_MODEL") or "gpt-4o"


def default(**kwargs):
    try:
        return by_model(DEFAULT_MODEL, **kwargs)
    except ValueError as e:
        msg = f"Default model {DEFAULT_MODEL} not found in any provider"
        raise Exception(msg) from e


def by_model(model: str, api_key: str | None, **kwargs) -> tuple[provider.Provider, Spec]:
    p = next((p for p in PROVIDERS.values() if model in p.info), None)
    if not p:
        msg = f"{model} is not registered"
        raise ValueError(msg)
    if p.key:
        api_key = api_key or os.getenv(p.key)
        if not api_key:
            msg = f"{p.key} environment variable is required"
            raise Exception(msg)

    return p.get_model()(api_key=api_key or "", model=model, info=p.info[model], **kwargs), p


def by_provider(
    provider_name: str, model: str | None = None, api_key: str | None = None, **kwargs
) -> tuple[provider.Provider, Spec]:
    inv_map = {p.casefold().lower(): p for p in PROVIDERS}
    try:
        p = PROVIDERS[inv_map[provider_name.casefold().lower()]]
    except KeyError as e:
        msg = f"Provider {provider_name} not found among {list(PROVIDERS.keys())}"
        raise ValueError(msg) from e

    if p.key:
        api_key = api_key or os.getenv(p.key)
        if not api_key:
            msg = f"{p.key} environment variable is required"
            raise Exception(msg)

    if not model:
        model = list(p.info)[0]
        info = p.info[model]
    else:
        info = p.info.get(model)
        if not info:
            warnings.warn(f"no information about cost of the model: {model}", UserWarning, stacklevel=2)
            info = provider.Info(
                prompt_cost=1,
                completion_cost=1,
                context_limit=4096,
            )

    return p.get_model()(api_key=api_key or "", model=model, info=info, **kwargs), p


def complete(
    models: list[provider.Provider],
    prompt: str | dict | list[dict],
    history: list[dict] | None = None,
    system_message: str | None = None,
    **kwargs: t.Any,
) -> MultiResult:
    def _wrap(
        model: provider.Sync,
    ) -> result.Chat:
        return model.complete(prompt, history, system_message, **kwargs)

    with ThreadPoolExecutor() as executor:
        return list(executor.map(_wrap, models))


def acomplete(
    models: list[provider.Provider],
    prompt: str | dict | list[dict],
    history: list[dict] | None = None,
    system_message: str | None = None,
    **kwargs: t.Any,
) -> AMultiResult:
    async def _wrap(
        model: provider.Async,
    ) -> result.Chat:
        return await model.acomplete(prompt, history, system_message, **kwargs)

    async def gather():
        return await asyncio.gather(*[_wrap(p) for p in models if isinstance(p, provider.Async)])

    return gather()


def benchmark(
    models: list[provider.Provider],
    problems: list[tuple[str, str]] | None = None,
    delay: float = 0,
    evaluator: provider.Provider | None = None,
    show_outputs: bool = False,
    **kwargs: t.Any,
) -> tuple[PrettyTable, PrettyTable | None]:
    from . import _bench as bench

    problems = problems or bench.PROBLEMS

    model_results = {}

    # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
    with ThreadPoolExecutor() as executor:
        fmap = {
            executor.submit(bench.process_prompts_sequentially, model, problems, evaluator, delay, **kwargs): model
            for model in models
        }

        for future in as_completed(fmap):
            try:
                (
                    outputs,
                    equeue,
                    threads,
                ) = future.result()
            except Exception as e:
                warnings.warn(f"Error processing results: {str(e)}", stacklevel=2)
                # Don't add failed models to results
                continue

            latency = [o["latency"] for o in outputs]
            total_latency = sum(latency)
            tokens = sum([o["tokens"] for o in outputs])
            model = fmap[future]
            model_results[model] = {
                "outputs": outputs,
                "total_latency": total_latency,
                "total_cost": sum([o["cost"] for o in outputs]),
                "total_tokens": tokens,
                "evaluation": [None] * len(outputs),
                "aggregated_speed": tokens / total_latency,
                "median_latency": statistics.median(latency),
            }

            if evaluator:
                for t in threads:
                    t.join()

                # Process all evaluation results
                while not equeue.empty():
                    i, evaluation = equeue.get()
                    if evaluation:
                        model_results[model]["evaluation"][i] = sum(evaluation)

    def eval(x):
        data = model_results[x]
        return data["aggregated_speed"] * (sum(data["evaluation"]) if evaluator else 1)

    sorted_models = sorted(
        model_results,
        key=eval,
        reverse=True,
    )

    pytable = defaultdict(list)
    for model in sorted_models:
        data = model_results[model]
        total_score = 0
        scores: list[int] = data["evaluation"]
        for i, out in enumerate(data["outputs"]):
            latency = out["latency"]
            tokens = out["tokens"]
            pytable["model"].append(str(model))
            pytable["text"].append(out["text"])
            pytable["tokens"].append(tokens)
            pytable["cost"].append(f"{out['cost']:.5f}")
            pytable["latency"].append(f"{latency:.2f}")
            pytable["speed"].append(f"{(tokens / latency):.2f}")
            if evaluator:
                score = scores[i]
                total_score += score
                score = str(score)
            else:
                score = "N/A"
            pytable["score"].append(score)

        pytable["model"].append(str(model))
        pytable["text"].append("")
        pytable["tokens"].append(str(data["total_tokens"]))
        pytable["cost"].append(f"{data['total_cost']:.5f}")
        pytable["latency"].append(f"{data['median_latency']:.2f}")
        pytable["speed"].append(f"{data['aggregated_speed']:.2f}")
        if evaluator and len(scores):
            acc = 100 * total_score / len(scores)
            score = f"{acc:.2f}%"
        else:
            score = "N/A"
        pytable["score"].append(score)

    order = ("model", "text", "tokens", "cost", "latency", "speed", "score")
    headers = {
        "model": "Model",
        "text": "Output",
        "tokens": "Tokens",
        "cost": "Cost ($)",
        "latency": "Latency (s)",
        "speed": "Speed (tokens/sec)",
        "score": "Evaluation",
    }
    questions_table: PrettyTable | None = None
    if evaluator:
        questions_table = PrettyTable(["Category", "Index", "Question"])
        questions_table.align["Question"] = "l"

        for i, problem in enumerate(problems):
            scores = [m["evaluation"][i] for m in model_results.values()]

            if all(scores):
                questions_table.add_row(["Easiest", i, _ellipsize(problem[0])])
            elif not any(scores):
                questions_table.add_row(["Hardest", i, _ellipsize(problem[0])])
    else:
        headers.pop("score")
        pytable.pop("score")

    if not show_outputs:
        headers.pop("text")
        pytable.pop("text")
    table = PrettyTable([headers[o] for o in order])
    ordered = [pytable[o] for o in order]
    table.add_rows(list(zip(*ordered, strict=False)))

    return table, questions_table


@staticmethod
def _ellipsize(text: str, max_len: int = 100) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."
