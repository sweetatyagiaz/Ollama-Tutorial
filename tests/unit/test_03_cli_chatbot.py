"""Unit tests for the interactive CLI chatbot."""

from unittest.mock import Mock, patch

from projects.cli_chatbot.main import main


@patch("projects.cli_chatbot.main.OllamaClient")
def test_chatbot_answers_and_exits(mock_client_class, monkeypatch, capsys):
    client = Mock()
    client.health_check.return_value = True
    client.model = "llama3.2"
    client.chat.return_value = "Ollama runs language models locally."
    mock_client_class.return_value = client

    inputs = iter(["What is Ollama?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Ollama CLI Chatbot" in output
    assert "Ollama runs language models locally." in output
    assert "Goodbye!" in output
    client.chat.assert_called_once()



@patch("projects.cli_chatbot.main.OllamaClient")
def test_chatbot_preserves_history_between_turns(mock_client_class, monkeypatch):
    client = Mock()
    client.health_check.return_value = True
    client.model = "llama3.2"
    client.chat.side_effect = ["Your name is Alex.", "You said your name is Alex."]
    mock_client_class.return_value = client

    inputs = iter(["My name is Alex.", "What is my name?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    assert client.chat.call_count == 2
    first_call = client.chat.call_args_list[0]
    second_call = client.chat.call_args_list[1]

    assert first_call.kwargs["history"] == []
    assert second_call.args == ("What is my name?",)
    assert second_call.kwargs["history"] == [
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Your name is Alex."},
    ]


@patch("projects.cli_chatbot.main.OllamaClient")
def test_chatbot_handles_unavailable_ollama(mock_client_class, capsys):
    client = Mock()
    client.health_check.return_value = False
    mock_client_class.return_value = client

    main()

    output = capsys.readouterr().out
    assert "Ollama is unavailable" in output
    client.chat.assert_not_called()


@patch("projects.cli_chatbot.main.OllamaClient")
def test_chatbot_skips_empty_input(mock_client_class, monkeypatch, capsys):
    client = Mock()
    client.health_check.return_value = True
    client.model = "llama3.2"
    client.chat.return_value = "Hello!"
    mock_client_class.return_value = client

    inputs = iter(["", "   ", "Hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    client.chat.assert_called_once()
    args, kwargs = client.chat.call_args
    assert args == ("Hello",)
    assert kwargs["system"] == (
        "You are a helpful AI assistant. "
        "Answer directly and acknowledge uncertainty when needed."
    )
    assert kwargs["history"] == []
