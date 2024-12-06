from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO


food_cls_model = YOLO('src/ai/weights/food-cls-best.pt')
gym_equipment_cls_model = YOLO('src/ai/weights/gym-equipment-cls-best.pt')
yolo11_model = YOLO('src/ai/weights/yolo11m.pt')


def load_image(img_url: str):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


def predict_food_cls(url: str):
    img = load_image(url)
    result = food_cls_model(img)[0]

    probs = result.probs
    top1_class_index = probs.top1
    top1_class_name = result.names[top1_class_index]
    top1_score = float(probs.top1conf)

    top5_class_index = probs.top5
    top5_class_name = list(
        map(lambda idx: result.names[idx], top5_class_index))
    top5_score = list(map(lambda conf: float(conf), probs.top5conf))

    return {
        "top1ClassName": top1_class_name,
        "top1Score": top1_score,
        "top5ClassName": top5_class_name,
        "top5Score": top5_score
    }


def predict_gym_equipment_cls(url: str):
    img = load_image(url)
    result = gym_equipment_cls_model(img)[0]

    class_index = result.boxes.cls.item()
    class_name = result.names[class_index]
    score = float(result.boxes.conf)

    return {"top1ClassName": class_name, "top1Score": score}


def predict_yolo11(url: str):
    img = load_image(url)
    result = yolo11_model(img)[0]

    class_index = result.boxes.cls.item()
    class_name = result.names[class_index]
    score = float(result.boxes.conf)

    return {"top1ClassName": class_name, "top1Score": score}
