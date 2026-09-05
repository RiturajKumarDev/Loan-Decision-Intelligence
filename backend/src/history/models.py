from enum import Enum

from pydantic import BaseModel, Field


class HousingStatus(str, Enum):
    OWN = "Own"
    RENT = "Rent"
    MORTGAGE = "Mortgage"
    OTHER = "Other"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "High School"
    BACHELOR_DEGREE = "Bachelor's Degree"
    MASTER_DEGREE = "Master's Degree"
    PHD_DOCTORATE = "PhD / Doctorate"
    ASSOCIATE_DEGREE = "Associate Degree"


class LoanPredictionHistory(BaseModel):
    user_id: str

    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0)

    education_level: EducationLevel
    housing_status: HousingStatus

    probability: list
    prediction: int
