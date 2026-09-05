from fastapi import APIRouter

from src.auth import controller as auth_controller
from src.ml_model import controller as ml_controller
from src.history import controller as history_controller

router = APIRouter()

router.include_router(auth_controller.router, tags=["Authentication"])
router.include_router(ml_controller.router, tags=["AI/ML"])
router.include_router(history_controller.router, tags=["Prediction History"])
