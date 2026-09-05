from fastapi import APIRouter, HTTPException, Depends
import pandas as pd

from src.auth.service import verify_token
from src.database.core import users_collection
from src.ml_model.model import LoanPredictionRequest
from src.ml_model.service import load_models
from src.history.service import create_prediction_history
from src.history.models import LoanPredictionHistory

router = APIRouter(
    prefix="/ml",
    tags=["AI/ML"],
)


@router.post("/predict")
async def predict(
    request_model: LoanPredictionRequest,
    payload: dict = Depends(verify_token),
):
    user = await users_collection.find_one({"email": payload["sub"]})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        model, columns = load_models()
        data = request_model.model_dump()
        if not data:
            raise HTTPException(status_code=400, detail="No input data provided")
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(
            columns=columns,
            fill_value=0,
        )
        pred = model.predict(input_df)[0]
        pred_proba = (
            model.predict_proba(input_df)[0].tolist()
            if hasattr(model, "predict_proba")
            else []
        )
        history = LoanPredictionHistory(
            user_id=str(user["_id"]),
            age=request_model.age,
            annual_income=request_model.annual_income,
            loan_amount=request_model.loan_amount,
            credit_score=request_model.credit_score,
            employment_years=request_model.employment_years,
            education_level=request_model.education_level,
            housing_status=request_model.housing_status,
            probability=pred_proba,
            prediction=int(pred),
        )
        await create_prediction_history(history)
        return {"success": True, "prediction": int(pred), "probability": pred_proba}

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file error: {str(e)}")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
