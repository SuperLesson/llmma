import typing as t

from attrs import define

from .. import provider
from .loader import LazyLoader

ollama = LazyLoader("llmma.ollama", globals(), "llmma._lazy.libs.ollama")


@define
class Spec:
    get_model: t.Callable[[], provider.Provider]
    info: dict[str, provider.Info]
    key: str | None = None

    @classmethod
    def lazy(cls, mod: str, p: str, info: dict[str, provider.Info], key: str | None = None) -> "Spec":
        m = LazyLoader(f"llmma.{mod}", globals(), f"llmma._lazy.libs.{mod}")

        def get_model() -> type[provider.Provider]:
            return t.cast(type[provider.Provider], getattr(m, p))

        return cls(get_model, info, key)  # type: ignore


PROVIDERS = {
    "AI21": Spec.lazy(
        "ai21",
        "AI21",
        {
            "j2-grande-instruct": provider.Info(prompt_cost=10.0, completion_cost=10.0, context_limit=8192),
            "j2-jumbo-instruct": provider.Info(prompt_cost=15.0, completion_cost=15.0, context_limit=8192),
        },
        "AI21_API_KEY",
    ),
    "AlephAlpha": Spec.lazy(
        "aleph_alpha",
        "AlephAlpha",
        {
            "luminous-base": provider.Info(prompt_cost=6.6, completion_cost=7.6, context_limit=2048),
            "luminous-extended": provider.Info(prompt_cost=9.9, completion_cost=10.9, context_limit=2048),
            "luminous-supreme": provider.Info(prompt_cost=38.5, completion_cost=42.5, context_limit=2048),
            "luminous-supreme-control": provider.Info(prompt_cost=48.5, completion_cost=53.6, context_limit=2048),
        },
        "ALEPHALPHA_API_KEY",
    ),
    "Anthropic": Spec.lazy(
        "anthropic",
        "Anthropic",
        {
            "claude-3-haiku-latest": provider.Info(
                prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=4_096
            ),
            "claude-3-opus-latest": provider.Info(
                prompt_cost=15.00, completion_cost=75, context_limit=200_000, output_limit=4_096
            ),
            "claude-3-5-haiku-latest": provider.Info(
                prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=8_192
            ),
            "claude-3-sonnet-latest": provider.Info(
                prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
            ),
            "claude-3-5-sonnet-latest": provider.Info(
                prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=8_192
            ),
        },
        "ANTHROPIC_API_KEY",
    ),
    "BedrockAnthropic": Spec.lazy(
        "bedrock_anthropic",
        "BedrockAnthropic",
        {
            "anthropic.claude-v2": provider.Info(prompt_cost=11.02, completion_cost=32.68, context_limit=100_000),
            "anthropic.claude-3-haiku-20240307-v1:0": provider.Info(
                prompt_cost=0.25, completion_cost=1.25, context_limit=200_000, output_limit=4_096
            ),
            "anthropic.claude-3-sonnet-20240229-v1:0": provider.Info(
                prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
            ),
            "anthropic.claude-3-5-sonnet-20240620-v1:0": provider.Info(
                prompt_cost=3.00, completion_cost=15, context_limit=200_000, output_limit=4_096
            ),
        },
    ),
    "Cohere": Spec.lazy(
        "cohere",
        "Cohere",
        {
            "command": provider.Info(prompt_cost=15.0, completion_cost=15, context_limit=2048),
            "command-nightly": provider.Info(prompt_cost=15.0, completion_cost=15, context_limit=4096),
        },
        "COHERE_API_KEY",
    ),
    "DeepSeek": Spec.lazy(
        "deepseek",
        "DeepSeek",
        {
            "deepseek-chat": provider.Info(
                prompt_cost=0.14, completion_cost=0.28, context_limit=64_000, output_limit=8_192
            ),
            "deepseek-reasoner": provider.Info(
                prompt_cost=0.14, completion_cost=2.19, context_limit=64_000, output_limit=8_192
            ),
        },
        "DEEPSEEK_API_KEY",
    ),
    "Google": Spec.lazy(
        "google",
        "Google",
        {
            # no support for "textembedding-gecko"
            "chat-bison": provider.Info(prompt_cost=0.5, completion_cost=0.5, context_limit=0),
            "text-bison": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "text-bison-32k": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "code-bison": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "code-bison-32k": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "codechat-bison": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "codechat-bison-32k": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "gemini-pro": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "gemini-1.5-pro-preview-0514": provider.Info(
                prompt_cost=0.35,
                completion_cost=0.53,
                context_limit=0,
                quirks={
                    "uses_characters": False,
                },
            ),
            "gemini-1.5-flash-preview-0514": provider.Info(
                prompt_cost=0.35,
                completion_cost=0.53,
                context_limit=0,
                quirks={
                    "uses_characters": False,
                },
            ),
        },
    ),
    "GoogleGenAI": Spec.lazy(
        "google_genai",
        "GoogleGenAI",
        {
            # no support for "textembedding-gecko"
            "chat-bison-genai": provider.Info(prompt_cost=0.5, completion_cost=0.5, context_limit=0),
            "text-bison-genai": provider.Info(prompt_cost=1.0, completion_cost=1.0, context_limit=0),
            "gemini-1.5-pro": provider.Info(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
            "gemini-1.5-pro-latest": provider.Info(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
            "gemini-1.5-flash": provider.Info(prompt_cost=0.075, completion_cost=0.3, context_limit=128000),
            "gemini-1.5-flash-latest": provider.Info(prompt_cost=0.075, completion_cost=0.3, context_limit=128000),
            "gemini-1.5-pro-exp-0801": provider.Info(prompt_cost=3.5, completion_cost=10.5, context_limit=128000),
        },
        "GOOGLE_API_KEY",
    ),
    "Groq": Spec.lazy(
        "groq",
        "Groq",
        {
            "llama-3.1-405b-reasoning": provider.Info(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
            "llama-3.1-70b-versatile": provider.Info(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
            "llama-3.1-8b-instant": provider.Info(prompt_cost=0.05, completion_cost=0.08, context_limit=131072),
            "gemma2-9b-it": provider.Info(prompt_cost=0.20, completion_cost=0.20, context_limit=131072),
            "llama-3.3-70b-versatile": provider.Info(prompt_cost=0.59, completion_cost=0.79, context_limit=131072),
        },
        "GROQ_API_KEY",
    ),
    "Huggingface": Spec.lazy(
        "huggingface",
        "Huggingface",
        {
            "hf_deepseek-r1": provider.Info(
                hf_repo="deepseek-ai/DeepSeek-R1",
                prompt_cost=0,
                completion_cost=0,
                context_limit=128_000,
            ),
            "hf_deepseek-v3": provider.Info(
                hf_repo="deepseek-ai/DeepSeek-V3",
                prompt_cost=0,
                completion_cost=0,
                context_limit=128_000,
            ),
            "hf_pythia": provider.Info(
                hf_repo="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
            ),
            "hf_falcon40b": provider.Info(
                hf_repo="tiiuae/falcon-40b-instruct",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
                local=True,
            ),
            "hf_falcon7b": provider.Info(
                hf_repo="tiiuae/falcon-7b-instruct",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
                local=True,
            ),
            "hf_mptinstruct": provider.Info(
                hf_repo="mosaicml/mpt-7b-instruct",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
                local=True,
            ),
            "hf_mptchat": provider.Info(
                hf_repo="mosaicml/mpt-7b-chat",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
                local=True,
            ),
            "hf_llava": provider.Info(
                hf_repo="liuhaotian/LLaVA-Lightning-MPT-7B-preview",
                prompt_cost=0,
                completion_cost=0,
                context_limit=2048,
                local=True,
            ),
            "hf_dolly": provider.Info(
                hf_repo="databricks/dolly-v2-12b",
                prompt_cost=0,
                completion_cost=0,
                context_limit=-1,
                local=True,
            ),
            "hf_vicuna": provider.Info(
                hf_repo="CarperAI/stable-vicuna-13b-delta",
                prompt_cost=0,
                completion_cost=0,
                context_limit=-1,
            ),
        },
        "HUGGINFACEHUB_API_KEY",
    ),
    "Mistral": Spec.lazy(
        "mistral",
        "Mistral",
        {
            "mistral-tiny": provider.Info(prompt_cost=0.25, completion_cost=0.25, context_limit=32_000),
            # new endpoint for mistral-tiny, mistral-tiny will be deprecated in ~June 2024
            "open-mistral-7b": provider.Info(prompt_cost=0.25, completion_cost=0.25, context_limit=32_000),
            "mistral-small": provider.Info(prompt_cost=0.7, completion_cost=0.7, context_limit=32_000),
            # new endpoint for mistral-small, mistral-small will be deprecated in ~June 2024
            "open-mixtral-8x7b": provider.Info(prompt_cost=0.7, completion_cost=0.7, context_limit=32_000),
            "mistral-small-latest": provider.Info(prompt_cost=2.0, completion_cost=6.0, context_limit=32_000),
            "mistral-medium-latest": provider.Info(prompt_cost=2.7, completion_cost=8.1, context_limit=32_000),
            "mistral-large-latest": provider.Info(prompt_cost=3.0, completion_cost=9.0, context_limit=32_000),
            "open-mistral-nemo": provider.Info(prompt_cost=0.3, completion_cost=0.3, context_limit=32_000),
        },
        "MISTRAL_API_KEY",
    ),
    "OpenAI": Spec.lazy(
        "openai",
        "OpenAI",
        {
            "gpt-4o": provider.Info(
                prompt_cost=2.5,
                completion_cost=10.0,
                context_limit=128_000,
                output_limit=16_384,
                limit_per_minute=30_000,
            ),
            "gpt-4o-mini": provider.Info(
                prompt_cost=0.15,
                completion_cost=0.60,
                context_limit=128_000,
                output_limit=16_384,
                limit_per_minute=200_000,
            ),
            "o1": provider.Info(
                prompt_cost=15.0,
                completion_cost=60.0,
                context_limit=200_000,
                output_limit=100_000,
                quirks={
                    "use_max_completion_tokens": True,
                },
            ),
            "o1-mini": provider.Info(
                prompt_cost=3.0,
                completion_cost=12.0,
                context_limit=128_000,
                output_limit=65_536,
                quirks={
                    "use_max_completion_tokens": True,
                },
                limit_per_minute=200_000,
            ),
        },
        "OPENAI_API_KEY",
    ),
    "OpenRouter": Spec.lazy(
        "openrouter",
        "OpenRouter",
        {
            "nvidia/llama-3.1-nemotron-70b-instruct": provider.Info(
                prompt_cost=0.35,
                completion_cost=0.4,
                context_limit=131072,
            ),
            "x-ai/grok-2": provider.Info(
                prompt_cost=5.0,
                completion_cost=10.0,
                context_limit=32768,
            ),
            "nousresearch/hermes-3-llama-3.1-405b:free": provider.Info(
                prompt_cost=0.0,
                completion_cost=0.0,
                context_limit=8192,
            ),
            "google/gemini-flash-1.5-exp": provider.Info(
                prompt_cost=0.0,
                completion_cost=0.0,
                context_limit=1000000,
            ),
            "liquid/lfm-40b": provider.Info(
                prompt_cost=0.0,
                completion_cost=0.0,
                context_limit=32768,
            ),
            "mistralai/ministral-8b": provider.Info(
                prompt_cost=0.1,
                completion_cost=0.1,
                context_limit=128000,
            ),
            "qwen/qwen-2.5-72b-instruct": provider.Info(
                prompt_cost=0.35,
                completion_cost=0.4,
                context_limit=131072,
            ),
            "x-ai/grok-2-1212": provider.Info(
                prompt_cost=2.0,
                completion_cost=10.0,
                context_limit=131072,
            ),
            "amazon/nova-pro-v1": provider.Info(
                prompt_cost=0.8,
                completion_cost=3.2,
                context_limit=300000,
                image_input_cost=1.2,
            ),
            "qwen/qwq-32b-preview": provider.Info(
                prompt_cost=0.12,
                completion_cost=0.18,
                context_limit=32768,
            ),
            "mistralai/mistral-large-2411": provider.Info(
                prompt_cost=2.0,
                completion_cost=6.0,
                context_limit=128000,
            ),
        },
        "OPENROUTER_API_KEY",
    ),
    "Reka": Spec.lazy(
        "reka",
        "Reka",
        {
            "reka-core": provider.Info(prompt_cost=3.0, completion_cost=15.0, context_limit=128000),
            "reka-edge": provider.Info(prompt_cost=0.4, completion_cost=1.0, context_limit=128000),
            "reka-flash": provider.Info(prompt_cost=0.8, completion_cost=2.0, context_limit=128000),
        },
        "REKA_API_KEY",
    ),
    "Together": Spec.lazy(
        "together",
        "Together",
        {
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": provider.Info(
                prompt_cost=5.0,
                completion_cost=5.0,
                context_limit=4096,
            ),
        },
        "TOGETHER_API_KEY",
    ),
}


def use_local_provider(
    host: str = "http://localhost:11434", context_limit: int = 4096, output_limit: int = 2048, **kwargs
):
    global PROVIDERS
    Ollama, info = ollama.get_provider(host=host, context_limit=context_limit, output_limit=output_limit, **kwargs)
    PROVIDERS["Ollama"] = Spec(lambda: Ollama, info)


__all__ = [
    "PROVIDERS",
    "use_local_provider",
]
