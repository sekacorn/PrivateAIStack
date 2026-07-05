from pydantic import BaseModel


class PolicyOutcome(BaseModel):
    policy: str
    action: str
    allowed: bool
    reason: str
