from pydantic import BaseModel, Field


class LoanPredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0)
    education_level: str
    housing_status: str
