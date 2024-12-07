from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import src.service.models as service
from src.dto.models import PredictRequest

router = APIRouter(
    prefix="/models",
    tags=["AI Models"],
)


@router.post("/food-classification/predict")
def predict_food_cls(body: PredictRequest):
    result = service.predict_food_cls(body.url)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)


@router.post("/gym_equipment/predict")
def predict_gym_equipment(body: PredictRequest):
    result = service.predict_gym_equipment(body.url)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)


@router.post("/yolo11/predict")
def predict_yolo11(body: PredictRequest):
    result = service.predict_yolo11(body.url)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)
