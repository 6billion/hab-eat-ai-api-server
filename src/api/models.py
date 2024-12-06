from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import src.service.models as service
from src.dto.models import PredictFoodClassificationPredictRequest

router = APIRouter(
    prefix="/models",
    tags=["AI Models"],
)


@router.post("/food-classification/predict")
def predict_food_cls(body: PredictFoodClassificationPredictRequest):
    result = service.predict_food_cls(body.url)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)
