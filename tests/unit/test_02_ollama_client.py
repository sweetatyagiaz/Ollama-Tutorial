"""Unit tests for the reusable Ollama client."""

from unittest.mock import Mock

import pytest
import requests

from src.ollama_client import OllamaClient


@pytest.fixture
def mock_session():
    return Mock(spec=requests.Session)


@pytest.fixture
def client(mock_session):
    return OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2",
        timeout=10,
        session=mock_session,
    )


def test_chat_returns_assistant_content(client, mock_session):
    response = Mock()
    response.json.return_value = {
        "message": {"role": "assistant", "content": "Hello from Ollama"}
    }
    mock_session.post.return_value = response

    result = client.chat("Hello")

    assert result == "Hello from Ollama"
    mock_session.post.assert_called_once_with(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {"temperature": 0.7},
        },
        timeout=10,
    )
    response.raise_for_status.assert_called_once()


def test_chat_includes_system_message(client, mock_session):
    response = Mock()
    response.json.return_value = {
        "message": {"role": "assistant", "content": "I am helpful"}
    }
    mock_session.post.return_value = response

    result = client.chat("Introduce yourself", system="You are a tutor.")

    assert result == "I am helpful"
    payload = mock_session.post.call_args.kwargs["json"]
    assert payload["messages"][0] == {
        "role": "system",
        "content": "You are a tutor.",
    }


def test_chat_rejects_empty_message(client, mock_session):
    with pytest.raises(ValueError, match="message must not be empty"):
        client.chat("   ")

    mock_session.post.assert_not_called()


def test_chat_raises_clear_error_on_connection_failure(client, mock_session):
    mock_session.post.side_effect = requests.ConnectionError("Connection refused")

    with pytest.raises(RuntimeError, match="Could not communicate with Ollama"):
        client.chat("Hello")


def test_health_check_returns_true_when_api_is_available(
    client, mock_session
):
    response = Mock()
    mock_session.get.return_value = response

    assert client.health_check() is True
    mock_session.get.assert_called_once_with(
        "http://localhost:11434/api/tags",
        timeout=10,
    )
    response.raise_for_status.assert_called_once()


def test_health_check_returns_false_on_connection_failure(
    client, mock_session
):
    mock_session.get.side_effect = requests.ConnectionError("Connection refused")

    assert client.health_check() is False


def test_chat_rejects_invalid_response(client, mock_session):
    response = Mock()
    response.json.return_value = {}
    mock_session.post.return_value = response

    with pytest.raises(RuntimeError, match="invalid chat response"):
        client.chat("Hello")
