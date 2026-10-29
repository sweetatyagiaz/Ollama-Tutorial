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
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 55)

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
                system=(
                    "You are a helpful AI assistant. "
                    "Answer directly and acknowledge uncertainty when needed."
                ),
            )
            print(f"\nAssistant: {response}")
        except RuntimeError as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
