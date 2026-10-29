import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def test_ollama_api_is_available():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=10,
    )

    assert response.status_code == 200


def test_llama_model_is_installed():
    response = requests.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=10,
    )
    response.raise_for_status()

    models = response.json().get("models", [])
    model_names = [model["name"] for model in models]

    assert any(name.startswith("llama3.2") for name in model_names)


def test_llama_chat_returns_a_response():
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": "llama3.2",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: Ollama is working."
                }
            ],
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    result = response.json()

    assert result["message"]["role"] == "assistant"
    assert result["message"]["content"].strip()
    assert result["done"] is True
