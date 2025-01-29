# LLMMA

<!--[![PyPI version](https://badge.fury.io/py/pyllms.svg)](https://badge.fury.io/py/pyllms)-->
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/)

LLMMA is a minimal Python library to connect to various Language Models (LLMs) with a built-in model performance benchmark.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Multi-model Usage](#multi-model-usage)
  - [Async Support](#async-support)
  - [Streaming Support](#streaming-support)
  - [Chat History and System Message](#chat-history-and-system-message)
  - [Other Methods](#other-methods)
- [Configuration](#configuration)
- [Model Benchmarks](#model-benchmarks)
- [Supported Models](#supported-models)
- [Advanced Usage](#advanced-usage)
  - [Using OpenAI API on Azure](#using-openai-api-on-azure)
  - [Using Google Vertex LLM models](#using-google-vertex-llm-models)
  - [Using Local Ollama LLM models](#using-local-ollama-llm-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- Connect to top LLMs in a few lines of code
- Response meta includes tokens processed, cost, and latency standardized across models
- Multi-model support: Get completions from different models simultaneously
- LLM benchmark: Evaluate models on quality, speed, and cost
- Async and streaming support for compatible models

## Usage

### Basic Usage

```py
from llmma import LLMMA

model = LLMMA.default()  # defaults to 'gpt-4o'
result = model.complete(
    "What is the capital of the country where Mozart was born?",
    temperature=0.1,
    max_tokens=200
)[0]

print(result.text)
print(result.meta)
```

### Multimodel Usage

```py
models = LLMMA().add_model('gpt-3.5-turbo').add_model('claude-instant-v1')
result = models.complete('What is the capital of the country where Mozart was born?')[0]

print(result.text)
print(result.meta)
```

### Async Support

```py
result = await model.acomplete("What is the capital of the country where Mozart was born?")
```

### Streaming Support

```py
model = LLMMA().add_model('claude-v1')
# can only work with a single provider, returns a single stream
result = model.complete_stream("Write an essay on the Civil War")
for chunk in result.stream:
   if chunk is not None:
      print(chunk, end='')
```

### Chat History and System Message

```py
history = previous_messages.extend(
    {"role": "user", "content": user_input},
    {"role": "assistant", "content": result.text},
)

model.complete(history)

# you can also have LLMMA build the messages for you
model.complete(prompt, system_message=system, history=history)
```

### Other Methods

```py
count = model.count_tokens('The quick brown fox jumped over the lazy dog')
```

## Configuration

LLMMA will attempt to read API keys and the default model from environment variables. You can set them like this:

```sh
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
export COHERE_API_KEY="your_api_key_here"
export ALEPHALPHA_API_KEY="your_api_key_here"
export HUGGINFACEHUB_API_KEY="your_api_key_here"
export GOOGLE_API_KEY="your_api_key_here"
export MISTRAL_API_KEY="your_api_key_here"
export REKA_API_KEY="your_api_key_here"
export TOGETHER_API_KEY="your_api_key_here"
export GROQ_API_KEY="your_api_key_here"
export DEEPSEEK_API_KEY="your_api_key_here"

export LLMMA_DEFAULT_MODEL="gpt-4o"
```

Alternatively, you can pass in your API key as keyword args:

```py
model = LLMMA.add_model('gpt-4', api_key='your_api_key_here')
```

## Model Benchmarks

LLMMA includes an automated benchmark system. The quality of models is evaluated using a powerful model (e.g., GPT-4) on a range of predefined questions, or you can supply your own.

```py
bench = ['claude-3-haiku-20240307', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620', 'gpt-4o', 'mistral-large-latest', 'open-mistral-nemo', 'gpt-4', 'gpt-3.5-turbo', 'deepseek-coder', 'deepseek-chat', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile']
models = LLMMA()
for m in bench:
    models.add_model(m)

gpt4 = LLMMA.default()

bench.benchmark(evaluator=gpt4)
```

## Supported Models

To get a full list of supported models:

```py
from llmma import PROVIDERS

providers = PROVIDERS.keys()
for p in providers:
    print(p, PROVIDERS[p].supported_models())
```

<!-- ## Advanced Usage -->
<!---->
<!-- ### Using OpenAI API on Azure -->
<!---->
<!-- ```python -->
<!-- import llms -->
<!-- AZURE_API_BASE = "{insert here}" -->
<!-- AZURE_API_KEY = "{insert here}" -->
<!---->
<!-- model = llms.init('gpt-4') -->
<!---->
<!-- azure_args = { -->
<!--     "engine": "gpt-4",  # Azure deployment_id -->
<!--     "api_base": AZURE_API_BASE, -->
<!--     "api_type": "azure", -->
<!--     "api_version": "2023-05-15", -->
<!--     "api_key": AZURE_API_KEY, -->
<!-- } -->
<!---->
<!-- azure_result = model.complete("What is 5+5?", **azure_args) -->
<!-- ``` -->
<!---->
<!-- ### Using Google Vertex LLM models -->
<!---->
<!-- 1. Set up a GCP account and create a project -->
<!-- 2. Enable Vertex AI APIs in your GCP project -->
<!-- 3. Install gcloud CLI tool -->
<!-- 4. Set up Application Default Credentials -->
<!---->
<!-- Then: -->
<!---->
<!-- ```python -->
<!-- model = llms.init('chat-bison') -->
<!-- result = model.complete("Hello!") -->
<!-- ``` -->
<!---->
<!-- ### Using Local Ollama LLM models -->
<!---->
<!-- 1. Ensure Ollama is running and you've pulled the desired model -->
<!-- 2. Get the name of the LLM you want to use -->
<!-- 3. Initialize LLMMA: -->
<!---->
<!-- ```python -->
<!-- model = llms.init("tinyllama:latest") -->
<!-- result = model.complete("Hello!") -->
<!-- ``` -->
