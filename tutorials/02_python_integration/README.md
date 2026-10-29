# Sprint 2: Python Integration and Chat Applications

## Objectives

- Connect Python applications to Ollama through its REST API.
- Build a reusable Ollama client.
- Support chat requests, system prompts, timeouts, and health checks.
- Handle connection errors and invalid responses.
- Build an interactive command-line chatbot.
- Test application logic with mocked HTTP responses.

## Prerequisites

- Python 3.12 or compatible
- Ollama installed and running
- A local model such as `llama3.2`
- Dependencies installed from `requirements.txt`

## Ollama Client

Implementation: `src/ollama_client.py`

The `OllamaClient` supports configurable URL, model, timeout, chat inference, optional system prompts, temperature settings, health checks, and injectable HTTP sessions for testing.

Configuration variables:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2`
- `OLLAMA_TIMEOUT=120`

## Usage

```python
from src.ollama_client import OllamaClient

client = OllamaClient()
if client.health_check():
    print(client.chat("Explain Python functions in two sentences."))
else:
    print("Ollama is unavailable.")
```

## Command-Line Chatbot

Implementation: `projects/cli_chatbot/main.py`

Run from the repository root:

```bash
python -m projects.cli_chatbot.main
```

Type `exit` or `quit` to stop.

## Automated Tests

Run `python -m pytest tests/unit -v`.

Verified during Sprint 2 development: **15 tests passed**. Coverage includes API availability, model installation, chat inference, system prompts, error handling, health checks, and CLI behavior.

## Model Quality Observation

During live testing, `llama3.2` explained Python functions correctly but failed to identify Ollama. Successful API connectivity does not guarantee factual accuracy. Model evaluation and benchmarking are planned for Sprint 3.

## Limitations

- Conversation history is retained between turns within a single CLI session.
- HTTP requests are synchronous.
- The client returns plain-text assistant content.
- Unit tests mock HTTP interactions; live inference is tested separately.
