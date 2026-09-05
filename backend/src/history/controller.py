from fastapi import APIRouter, HTTPException, Depends

from src.history.models import LoanPredictionHistory
from src.auth.service import verify_token
from src.database.core import (
    users_collection,
    histories_collection,
)

router = APIRouter(
    prefix="/history",
    tags=["Prediction History"],
)


@router.get(
    "/histories",
    response_model=list[LoanPredictionHistory],
)
async def get_histories(
    payload: dict = Depends(verify_token),
):
    user = await users_collection.find_one({"email": payload["sub"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    histories = await histories_collection.find({"user_id": str(user["_id"])}).to_list(
        length=None
    )
    if not histories:
        return []
    for history in histories:
        history["user_id"] = str(history["user_id"])
    return histories


@router.get("/dashboard")
async def dashboard(
    payload: dict = Depends(verify_token),
):
    user = await users_collection.find_one({"email": payload["sub"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    histories = await histories_collection.find({"user_id": str(user["_id"])}).to_list(
        length=None
    )
    if not histories:
        return {"total": 0, "approved": 0, "declined": 0, "histories": []}

    total = len(histories)
    approved = 0
    declined = 0
    for history in histories:
        if history["prediction"] == 1:
            approved += 1
        else:
            declined += 1
    for history in histories:
        history["id"] = str(history["_id"])
        history["user_id"] = str(history["user_id"])
        del history["_id"]
    latest_histories = histories[:10]
    return {
        "total": total,
        "approved": approved,
        "declined": declined,
        "histories": latest_histories,
    }
