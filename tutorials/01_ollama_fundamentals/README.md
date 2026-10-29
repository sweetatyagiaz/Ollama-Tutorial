# Sprint 1: Ollama Fundamentals

## Objectives

- Understand local LLM inference and Ollama.
- Verify the Ollama installation and service.
- Download and run a language model.
- Inspect available and running models.
- Call the Ollama REST API.
- Record the development environment and baseline latency.

## Tasks

1. Verify the Ollama installation.
2. Verify the local API endpoint.
3. Download a suitable model.
4. Run an interactive inference session.
5. Test the chat REST API.
6. Record model and hardware information.

## Acceptance Criteria

- Ollama is installed and operational.
- The selected model responds to a prompt.
- The REST API returns a valid response.
- The environment and test results are documented.

## Next Sprint

Sprint 2: Python Integration and Chat Applications.

## Verified Environment

- Operating system: Linux
- Python: 3.12.3
- Ollama: 0.40.12
- Model: llama3.2:latest
- Model download size reported by Ollama: 2.0 GB
- Logical CPU count: 32
- Total RAM: approximately 30 GiB
- Available RAM at measurement time: approximately 12 GiB
- Model runner: llama.cpp
- Processor reported by Ollama: 100% CPU
- Context reported by Ollama: 4096 tokens

Environment measurements are snapshots and may change over time.
See `benchmarks/results/environment_report.json` for the recorded details.

## Test Results

Command:

    python -m pytest tests/unit/test_01_ollama_connection.py -v

Verified result: 3 passed, 0 failed.

The tests check:
- Ollama API availability
- Model installation
- Chat inference

## Known Observations

- The initial model runs on CPU in this environment.
- The home filesystem was approximately 91% full at the time of measurement.
- A successful API response does not guarantee a factually correct answer.
