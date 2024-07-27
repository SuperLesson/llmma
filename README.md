# PyLLMs

[![](https://dcbadge.vercel.app/api/server/aDNg6E9szy?compact=true&style=flat)](https://discord.gg/aDNg6E9szy) [![Twitter](https://img.shields.io/twitter/follow/KagiHQ?style=social)](https://twitter.com/KagiHQ) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/)

PyLLMs is a minimal Python library to connect to LLMs (OpenAI, Anthropic, Google, AI21, Cohere, Aleph Alpha, HuggingfaceHub) with a built-in model performance benchmark.

It is ideal for fast prototyping and evaluating different models thanks to:
- Connect to top LLMs in a few lines of code
- Response meta includes tokens processed, cost and latency standardized across the models
- Multi-model support: Get completions from different models at the same time
- LLM benchmark: Evaluate models on quality, speed and cost

Feel free to reuse and expand. Pull requests are welcome.

# Installation

Install the package using pip:

```
pip install pyllms
```

# Usage

```
import llms

model = llms.init('gpt-4')
result = model.complete("what is 5+5")

print(result.text)

```

Library will attempt to read the API keys and the default model from environment variables, which you can set like this (for the provider you are using):

```
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
export COHERE_API_KEY="your_api_key_here"
export ALEPHALPHA_API_KEY="your_api_key_here"
export HUGGINFACEHUB_API_KEY="your_api_key_here"
export GOOGLE_API_KEY="your_api_key_here"
export MISTRAL_API_KEY="your_api_key_here"

export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
```

Alternatively, you can pass initialization values to the init() method:

```
model=llms.init(openai_api_key='your_api_key_here', model='gpt-4')
```

For using Google LLMs through Vertex AI API, see "Using Google AI models" below.

You can also pass optional parameters to the complete method. 'temperature' and 'max_tokens' are standardized across all APIs and get converted to the corresponding API params.

Any other parameters accepted by the original model are supported in their verbatim form.

```
result = model.complete(
    "what is the capital of country where mozzart was born",
    temperature=0.1,
    max_tokens=200
)
```

Note: By default, temperature for all models is set to 0, and max_tokens to 300.

The result meta will contain helpful information like tokens used, cost (which is automatically calculated using current pricing), and response latency:
```
>>> print(result.meta)
{
  'model': 'gpt-3.5-turbo',
  'tokens': 34,
  'tokens_prompt': 20,
  'tokens_completion': 14,
  'cost': '0.00007',
  'latency': 1.4
}
```

## Multi-model usage

You can also initialize multiple models at once! This is very useful for testing and comparing output of different models in parallel.

```
>>> models=llms.init(model=['gpt-3.5-turbo','claude-instant-v1'])
>>> result=models.complete('what is the capital of country where mozzart was born')
>>> print(result.text)
[
 'The capital of the country where Mozart was born is Vienna, Austria.',
 'Wolfgang Amadeus Mozart was born in Salzburg, Austria.\n\nSo the capital of the country where Mozart was born is Vienna, Austria.'
]

>>> print(result.meta)
[
 {'model': 'gpt-3.5-turbo', 'tokens': 34, 'tokens_prompt': 20, 'tokens_completion': 14, 'cost': 6.8e-05, 'latency': 0.7097790241241455},
 {'model': 'claude-instant-v1', 'tokens': 54, 'tokens_prompt': 20, 'tokens_completion': 34, 'cost': 5.79e-05, 'latency': 0.7291600704193115}
]
```

## Async support
Async completion is supported for compatible models. It is not supported in multi-models mode yet.
```
result = await model.acomplete("what is the capital of country where mozzart was born")
```

## Streaming support

PyLLMs supports streaming from compatible models. 'complete_stream' method will return generator object and all you have to do is iterate through it:

```
model= llms.init('claude-v1')
result = model.complete_stream("write an essay on civil war")
for chunk in result.stream:
   if chunk is not None:
      print(chunk, end='')
```

Current limitations:
- When streaming, 'meta' is not available
- Multi-models are not supported for streaming

Tip: if you are testing this in python3 CLI, run it with -u parameter to disable buffering:

```
python3 -u
```

## Chat history and system message

You can pass the history of conversation in a list using the following format:

```
history=[]
history.append({"role": "user", "content": user_input})
history.append({"role": "assistant", "content": result.text})

model.complete(prompt=prompt, history=history)

```

In addition, OpenAI chat models support system message:

```
model.complete(prompt=prompt, system_message=system, history=history)

```

## Other methods

You can count tokens using the model's tokenizer:

```
count=model.count_tokens('the quick brown fox jumped over the lazy dog')
```

## Using OpenAI API on Azure (and elsewhere, where compatible)

PyLLMs supports optional params to specify base path for the OpenAI input/output format, for example OpenAI models running on Azure.

```
import llms
AZURE_API_BASE = "{insert here}"
AZURE_API_KEY = "{insert here}"

model = llms.init('gpt-4')

azure_args = {
    "engine": "gpt-4",  # Azure deployment_id
    "api_base": AZURE_API_BASE,
    "api_type": "azure",
    "api_version": "2023-05-15",
    "api_key": AZURE_API_KEY,
}

openai_result = model.complete("what is 5+5")
azure_result = model.complete("what is 5+5", **azure_args)

model.benchmark(**azure_args)
```

# Model Benchmarks

Models are appearing like mushrooms after rain and everyone is interested in three things:

1) Quality
2) Speed
3) Cost

PyLLMs icludes an automated benchmark system. The quality of models is evaluated using a powerful model (for example gpt-4) on a range of predefined questions, or you can supply your own.

```
model=llms.init(model=['claude-3-haiku-20240307','gpt-4o-mini','claude-3-5-sonnet-20240620','gpt-4o','mistral-large-latest','open-mistral-nemo','gpt-4','gpt-3.5-turbo','deepseek-coder','deepseek-chat','llama-3.1-8b-instant','llama-3.1-70b-versatile'])

gpt4=llms.init('gpt-4o')

models.benchmark(evaluator=gpt4)
```

| Model                                   | Tokens         | Total Cost ($)   | Median Latency (s) | Speed (tokens/sec) | Accuracy |
|-----------------------------------------|----------------|------------|-------------|--------------------|------------|
|         OpenAIProvider('gpt-3.5-turbo')         | 1569 | 0.01552 | 0.51 | 45.03 | 22.00% |
|       MistralProvider('open-mistral-nemo')      | 4135 | 0.00323 | 0.65 | 82.65 | 22.00% |
|             OpenAIProvider('gpt-4')             | 2477 | 0.33408 | 1.32 | 16.68 | 26.00% |
| DeepSeekProvider('deepseek-coder') | 8079 | 0.00327 | 4.13 | 16.72 | 28.00% |
| GroqProvider('llama-3.1-8b-instant') |  6628  | 0.00085  |     2.26    |       82.02        |   28.00%   |
|   AnthropicProvider('claude-3-haiku-20240307')  | 5642 | 0.00881 | 1.33 | 55.46 | 28.00% |
| DeepSeekProvider('deepseek-chat')  | 7310 | 0.00304 | 4.81 | 17.20 | 32.00% |
|          OpenAIProvider('gpt-4o-mini')          | 6029 | 0.00451 | 1.64 | 36.92 | 34.00% |
|     GroqProvider('llama-3.1-70b-versatile')     | 5190 | 0.00781 | 0.71 | 81.62 | 40.00% |
|     MistralProvider('mistral-large-latest')     | 5097 | 0.06787 | 3.08 | 18.03 | 44.00% |
| AnthropicProvider('claude-3-5-sonnet-20240620') | 6595 | 0.12018 | 2.54 | 48.90 | 46.00% |
|             OpenAIProvider('gpt-4o')            | 7482 | 0.14310 | 1.60 | 48.00 | 52.00% |

Overall Model Quality (Composite Score) (0.5 accuracy; 0.3 speed; 0.2 cost) as of July 26 2024


![Overall Model Quality (Composite Score by Rank) with Updated Weights](https://github.com/user-attachments/assets/93e42566-a632-44ba-b5eb-5ae20fb31c13)


To evaluate models on your own prompts, simply pass a list of questions and answers as tuple. The evaluator will automatically evaluate the responses:

```
models.benchmark(prompts=[("what is the capital of finland", "helsinki")], evaluator=gpt4)
```

# Supported Models

To get a list of supported models, call list(). Models will be shown in the order of least expensive to most expensive.

```
>>> model=llms.init()

>>> model.list()

>>> model.list("gpt') # lists only models with 'gpt' in name/provider name
```

Here is a pretty table of supported models (in alphabetical order).
```

| Provider            | Name                   | Prompt Cost | Completion Cost | Token Limit |
|---------------------|------------------------|-------------|-----------------|-------------|
| AI21Provider        | j2-grande-instruct     |        10.0 |            10.0 |        8192 |
| AI21Provider        | j2-jumbo-instruct      |        15.0 |            15.0 |        8192 |
| AlephAlphaProvider  | luminous-base          |         6.6 |             7.6 |        2048 |
| AlephAlphaProvider  | luminous-extended      |         9.9 |            10.9 |        2048 |
| AlephAlphaProvider  | luminous-supreme       |        38.5 |            42.5 |        2048 |
| AlephAlphaProvider  | luminous-supreme-control |      48.5 |            53.6 |        2048 |
| AnthropicProvider   | claude-instant-v1      |        1.63 |            5.51 |        9000 |
| AnthropicProvider   | claude-v1              |       11.02 |           32.68 |        9000 |
| CohereProvider      | command                |        25.0 |            25.0 |        8192 |
| CohereProvider      | command-nightly        |        25.0 |            25.0 |        8192 |
| GoogleProvider      | chat-bison             |         0.5 |             0.5 |        2048 |
| HuggingfaceHub      | hf_pythia              |         0.0 |             0.0 |        2048 |
| OpenAIProvider      | gpt-3.5-turbo          |         2.0 |             2.0 |        4000 |
| OpenAIProvider      | gpt-4                  |        30.0 |            60.0 |        8000 |

```

Useful links:\
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)\
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)\
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)\
[Cohere documentation](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#api)\
[Aleph Alpha documentation](https://aleph-alpha-client.readthedocs.io/en/latest/aleph_alpha_client.html#aleph_alpha_client.CompletionRequest)\
[Google Generateive AI
documentation](https://developers.generativeai.google/guide)
[Google Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts)

## Using Google Vertex LLM models

0. (Set up a GCP account and create a project)
1. Enable Vertex AI APIs in your GCP project - https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com
1. Install gcloud CLI tool - https://cloud.google.com/sdk/docs/install
2. Set up Application Default Credentials - https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
3. Now you should be able to init Google LLM

```
model = llms.init('chat-bison')
result = model.complete("hello!")
```

## Using Local Ollama LLM models

PyLLMs supports locally installed [Ollama](https://ollama.com/) models.

To use your Ollama models:

0. Ensure Ollama is running (as well as reachable at `localhost:11434`) and you've pulled the model you would like to use.

1. Get the name of the LLM you would like to use.

Run:

```bash
ollama list
```

to get a list of installed models.

```
NAME            	ID   SIZE  	MODIFIED
tinyllama:latest	...  637 MB	...
```

2. Initialize PyLLMs as you would any other model:

```python
model = llms.init("tinyllama:latest")
result = model.complete("hello!")
```

where `tinyllama:latest` is the model name of an installed model.

# License

This project is licensed under the MIT License.
