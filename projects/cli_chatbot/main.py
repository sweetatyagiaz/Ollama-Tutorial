"""Interactive command-line chatbot powered by Ollama."""

from src.ollama_client import OllamaClient


def main() -> None:
    client = OllamaClient()

    if not client.health_check():
        print("Error: Ollama is unavailable.")
        print("Start Ollama and try again.")
        return

    print("=" * 55)
    print("Ollama CLI Chatbot")
    print(f"Model: {client.model}")
    print("Conversation history is enabled for this session.")
    print("Type exit or quit to stop.")
    print("=" * 55)

    history: list[dict[str, str]] = []
    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer directly and acknowledge uncertainty when needed."
    )

    while True:
        try:
            message = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not message:
            continue

        try:
            response = client.chat(
                message,
                system=system_prompt,
                history=history.copy(),
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            print(f"\nAssistant: {response}")
        except RuntimeError as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
