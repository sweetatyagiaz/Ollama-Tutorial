# Ollama Installation Guide

## 1. Verified Environment

- Operating system: Linux
- Python: 3.12.3
- Ollama: 0.40.12
- Model: llama3.2
- API URL: http://localhost:11434

## 2. Verify Ollama

    ollama --version
    ollama list

## 3. Download and Run the Model

    ollama pull llama3.2
    ollama run llama3.2

Use /bye to exit the interactive session.

## 4. Verify the Local API

    curl -sS http://localhost:11434/api/tags

## 5. Test Chat Inference

Run the following command:

    curl -sS http://localhost:11434/api/chat \
      -H "Content-Type: application/json" \
      -d @- <<'JSON'
    {
      "model": "llama3.2",
      "messages": [
        {
          "role": "user",
          "content": "Explain how a transformer generates text."
        }
      ],
      "stream": false
    }
    JSON

## 6. Install Dependencies

    python -m pip install -r requirements.txt

## 7. Run Sprint 1 Tests

    python -m pytest tests/unit/test_01_ollama_connection.py -v

Initial result: 3 passed, 0 failed. Later runs should be recorded with their actual results.

## 8. Troubleshooting

- Connection refused: check whether the Ollama service is running.
- Model not found: run ollama pull llama3.2.
- Slow inference: inspect available system resources.
- Failed tests: inspect the full pytest traceback.

## Official Resources

- https://docs.ollama.com/
- https://docs.ollama.com/api
- https://ollama.com/library
