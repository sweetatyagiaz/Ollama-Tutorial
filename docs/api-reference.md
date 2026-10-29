# Ollama REST API Reference

## Base URL

http://localhost:11434

## GET /api/tags

Lists locally available models.

Example:

    curl -sS http://localhost:11434/api/tags

A successful response contains a models array.

## POST /api/chat

Generates an assistant response from a conversation.

Example:

    curl -sS http://localhost:11434/api/chat -H "Content-Type: application/json" -d @chat-request.json

Example request body for chat-request.json:

    {
      "model": "llama3.2",
      "messages": [
        {
          "role": "user",
          "content": "Explain Ollama."
        }
      ],
      "stream": false
    }

Important fields:

- model: local model name or tag.
- messages: conversation messages with roles and content.
- stream: false returns a single JSON response.

The response includes the assistant message and completion metadata.
HTTP success does not guarantee factual correctness.

## Security

Do not expose the local Ollama API publicly without appropriate
authentication, network restrictions, and access controls.

## Official Reference

https://docs.ollama.com/api
