from datetime import datetime

from capstone.dialog.rules import Intent


class CommandExecutor:
    """Execute only the built-in actions; never pass user input to a shell."""

    def __init__(self, user_name: str):
        self.user_name = user_name

    def execute(self, intent: Intent) -> str:
        if intent.name == "help":
            return "Commands: help, status, time, hello, say <text>, exit."
        if intent.name == "status":
            return f"All systems ready. Authenticated as {self.user_name}."
        if intent.name == "time":
            return f"The local time is {datetime.now().astimezone().strftime('%H:%M:%S %Z')}."
        if intent.name == "greet":
            return f"Hello, {self.user_name}!"
        if intent.name == "say":
            return intent.argument or "You did not provide any text."
        if intent.name == "exit":
            return "Goodbye."
        return "I did not recognize that command. Type 'help' to see available commands."
