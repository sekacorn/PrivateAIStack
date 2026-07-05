from pydantic import BaseModel


class ToolAvailability(BaseModel):
    name: str
    available: bool
    path: str | None = None
    install_hint: str
