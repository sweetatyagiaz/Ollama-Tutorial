"""Reusable HTTP client for the Ollama REST API."""

import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


class OllamaClient:
    """A configurable client for Ollama chat inference."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT", "120"))
        )
        self.session = session or requests.Session()

    def chat(
        self,
        message: str,
        system: str | None = None,
        temperature: float = 0.7,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Send a message with optional prior conversation messages."""
        if not message or not message.strip():
            raise ValueError("message must not be empty")

        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        if history:
            for item in history:
                if (
                    not isinstance(item, dict)
                    or item.get("role") not in {"user", "assistant"}
                    or not isinstance(item.get("content"), str)
                ):
                    raise ValueError(
                        "history entries must contain a user/assistant role and string content"
                    )
                messages.append(
                    {"role": item["role"], "content": item["content"]}
                )

        messages.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not communicate with Ollama at {self.base_url}: {exc}"
            ) from exc

        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "Ollama returned an invalid chat response."
            ) from exc

        if not isinstance(content, str):
            raise RuntimeError("Ollama returned a non-text chat response.")

        return content

    def health_check(self) -> bool:
        """Return True when the Ollama API is reachable."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
