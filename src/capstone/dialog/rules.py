from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    argument: str = ""


def parse_command(text: str) -> Intent:
    """Map a small, explicit set of phrases to reusable action intents."""
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"help", "commands"}:
        return Intent("help")
    if normalized in {"status", "check status"}:
        return Intent("status")
    if normalized in {"time", "what time is it"}:
        return Intent("time")
    if normalized in {"hello", "hi", "greet"}:
        return Intent("greet")
    if normalized.startswith("say "):
        return Intent("say", text.strip()[4:].strip())
    if normalized in {"quit", "exit", "goodbye"}:
        return Intent("exit")
    return Intent("unknown", text.strip())
