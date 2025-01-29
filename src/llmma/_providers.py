from dataclasses import dataclass

from .providers import (
    AI21Provider,
    AlephAlphaProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
    CohereProvider,
    DeepSeekProvider,
    GoogleGenAIProvider,
    GoogleProvider,
    GroqProvider,
    HuggingfaceHubProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    RekaProvider,
    TogetherProvider,
)
from .providers.base import Provider
from .providers.ollama import get_provider


@dataclass
class ProviderSpec:
    kind: type[Provider]
    api_key_name: str | None = None
    api_key: str | None = None

    @property
    def supported_models(self):
        return self.kind.MODEL_INFO.keys()


PROVIDERS = {
    "OpenAI": ProviderSpec(OpenAIProvider, "OPENAI_API_KEY"),
    "Anthropic": ProviderSpec(AnthropicProvider, "ANTHROPIC_API_KEY"),
    "BedrockAnthropic": ProviderSpec(BedrockAnthropicProvider),
    "AI21": ProviderSpec(AI21Provider, "AI21_API_KEY"),
    "Cohere": ProviderSpec(CohereProvider, "COHERE_API_KEY"),
    "AlephAlpha": ProviderSpec(AlephAlphaProvider, "ALEPHALPHA_API_KEY"),
    "HuggingfaceHub": ProviderSpec(HuggingfaceHubProvider, "HUGGINFACEHUB_API_KEY"),
    "GoogleGenAI": ProviderSpec(GoogleGenAIProvider, "GOOGLE_API_KEY"),
    "Mistral": ProviderSpec(MistralProvider, "MISTRAL_API_KEY"),
    "Google": ProviderSpec(GoogleProvider),
    "DeepSeek": ProviderSpec(DeepSeekProvider, "DEEPSEEK_API_KEY"),
    "Groq": ProviderSpec(GroqProvider, "GROQ_API_KEY"),
    "Reka": ProviderSpec(RekaProvider, "REKA_API_KEY"),
    "Together": ProviderSpec(TogetherProvider, "TOGETHER_API_KEY"),
    "OpenRouter": ProviderSpec(OpenRouterProvider, "OPENROUTER_API_KEY"),
}


def use_local_provider(
    host: str = "http://localhost:11434", context_limit: int = 4096, output_limit: int = 2048, **kwargs
):
    global PROVIDERS
    OP = get_provider(host=host, context_limit=context_limit, output_limit=output_limit, **kwargs)
    PROVIDERS["Ollama"] = ProviderSpec(OP)
