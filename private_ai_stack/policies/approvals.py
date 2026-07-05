class ConsoleApprover:
    async def approve(self, reason: str) -> bool:
        response = input(f"{reason} Approve? [y/N] ")
        return response.strip().lower() == "y"


class DenyByDefaultApprover:
    async def approve(self, reason: str) -> bool:
        return False
