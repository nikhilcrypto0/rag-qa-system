from fastapi import APIRouter, BackgroundTasks

from app.api.schemas import FeedbackRequest
from app.monitoring.mlflow_logger import log_feedback

router = APIRouter()


@router.post("/feedback", summary="Rate an answer (thumbs up/down)")
async def feedback(request: FeedbackRequest, background_tasks: BackgroundTasks) -> dict:
    """Record a rating against an answer's response_id. No question or answer text is stored."""
    background_tasks.add_task(log_feedback, request.response_id, request.rating)
    return {"status": "ok"}
