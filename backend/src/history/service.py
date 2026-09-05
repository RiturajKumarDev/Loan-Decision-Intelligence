from src.history.models import LoanPredictionHistory
from src.database.core import histories_collection


async def create_prediction_history(
    predicted_data: LoanPredictionHistory,
):
    history_data = predicted_data.model_dump()
    result = await histories_collection.insert_one(history_data)
    return {
        "success": True,
        "history_id": str(result.inserted_id),
    }
