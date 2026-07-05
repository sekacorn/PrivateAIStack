from fastapi import APIRouter, Depends, Request

from private_ai_stack.api.dependencies import review_service
from private_ai_stack.api.schemas import ReviewRequest, ReviewResponse
from private_ai_stack.services.review_service import ReviewService

router = APIRouter(tags=["reviews"])


@router.post("/reviews", response_model=ReviewResponse)
async def create_review(payload: ReviewRequest, request: Request, service: ReviewService = Depends(review_service)) -> ReviewResponse:
    return await service.create_review(payload, request.state.request_id, getattr(request.state, "trace_id", None))


@router.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review(review_id: str, service: ReviewService = Depends(review_service)) -> ReviewResponse:
    return service.get_review(review_id)


@router.get("/reviews/{review_id}/findings")
async def get_review_findings(review_id: str, service: ReviewService = Depends(review_service)) -> dict[str, object]:
    return {"review_id": review_id, "findings": [finding.model_dump() for finding in service.get_findings(review_id)]}


@router.get("/reviews/{review_id}/report")
async def get_review_report(
    review_id: str, format: str = "markdown", service: ReviewService = Depends(review_service)
) -> dict[str, object]:
    return service.get_report(review_id, format)
