from fastapi import APIRouter

from private_ai_stack.policies.defaults import policy_catalog

router = APIRouter(tags=["policies"])


@router.get("/policies")
async def list_policies() -> dict[str, object]:
    return {"policies": policy_catalog()}
