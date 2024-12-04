from enum import Enum
from pydantic import BaseModel


class ModelId(str, Enum):
    food_cls = "food-classification"


class PredictFoodClassificationPredictRequest(BaseModel):
    """
    음식 분류 모델 예측
    - url (str): 음식 사진 이미지 url
    """
    url: str
