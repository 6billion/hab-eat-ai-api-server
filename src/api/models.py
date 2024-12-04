from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.service.models import predict_food_cls
from src.dto.models import PredictFoodClassificationPredictRequest

router = APIRouter(
    prefix="/models",
    tags=["AI Models"],
)


@router.post("/food-classification/predict")
def predict(body: PredictFoodClassificationPredictRequest):
    result = predict_food_cls(body.url)
    return JSONResponse(status_code=status.HTTP_200_OK, content=result)
