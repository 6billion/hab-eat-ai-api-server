# 🍽️ HAB-EAT AI API Server

운동과 식사 관련 이미지를 인식하는 AI 모델들을 제공하는 FastAPI 기반 서버입니다.

## 🚀 주요 기능

### AI 모델
1. **음식 분류 (Food Classification)** 
   - **모델**: YOLOv11-cls (Classification)
   - **데이터셋**: 셀렉트스타 오픈데이터셋 - 음식 사진 데이터셋
   - **클래스 수**: 90개 음식 카테고리 (총 100개 중 선별)
   - **이미지 수**: 클래스당 1,000개 (총 100,000개)
   - **훈련 이미지 크기**: 640x640
   - **성능**: Top-1 Accuracy ~88%, Top-5 Accuracy ~98%
   - 한국 음식, 양식, 아시아 음식 등 포함
   - 신뢰도 점수와 함께 예측 결과 제공

2. **운동기구 탐지 (Gym Equipment Detection)**
   - **모델**: YOLOv11m (Object Detection)
   - **데이터셋**: Roboflow Universe - workout dataset
   - **이미지 수**: 6,656개
   - **클래스 수**: 13개 운동 장비
   - **훈련 이미지 크기**: 416x416
   - **성능**: mAP50 ~68%, mAP50-95 ~51%
   - 객체 탐지 기능으로 이미지 내 운동기구 위치까지 파악
   - Chest Press, Lat Pull Down, Leg Press 등 주요 운동기구 지원

3. **범용 객체 탐지 (YOLO11)**
   - **모델**: YOLOv11m (Object Detection)
   - **데이터셋**: COCO 데이터셋 기반
   - **클래스 수**: 80개 클래스 객체 탐지
   - **훈련 이미지 크기**: 640x640
   - 사람, 동물, 차량, 일상용품 등 다양한 객체 인식

## 🏗️ 프로젝트 구조

```
hab-eat-ai-api-server/
├── src/
│   ├── ai/
│   │   ├── session/          # AI 모델 세션 관리
│   │   │   ├── food.py       # 음식 분류 모델
│   │   │   ├── gym_equipment.py  # 운동기구 탐지 모델
│   │   │   └── yolo11.py     # YOLO11 객체 탐지 모델
│   │   └── torchscript/      # 학습된 TorchScript 모델 파일
│   ├── api/                  # FastAPI 라우터
│   │   ├── health_check.py   # 헬스체크 API
│   │   └── models.py         # AI 모델 예측 API
│   ├── dto/                  # 데이터 전송 객체
│   ├── service/              # 비즈니스 로직
│   └── app.py                # FastAPI 애플리케이션 진입점
├── Dockerfile                # Docker 이미지 빌드 설정
├── docker-compose.yml        # Docker Compose 설정
└── requirements.txt          # Python 의존성
```

## 🛠️ 기술 스택

- **Framework**: FastAPI
- **AI/ML**: PyTorch, TorchScript, OpenCV
- **Image Processing**: Pillow, torchvision
- **Containerization**: Docker, Docker Compose
- **Python Version**: 3.10

## 📋 API 엔드포인트

### 헬스체크
- `GET /health-check/` - 서버 상태 확인

### AI 모델 예측
- `POST /models/food-classification/predict` - 음식 분류
- `POST /models/gym_equipment/predict` - 운동기구 탐지  
- `POST /models/yolo11/predict` - 범용 객체 탐지

### 요청 형식
```json
{
  "url": "https://example.com/image.jpg"
}
```

### 응답 형식
```json
{
  "top1ClassName": "치킨",
  "top1Score": 0.95
}
```

## 🚀 설치 및 실행

### 1. Docker를 이용한 실행 (권장)

```bash
# Docker Compose로 실행
docker-compose up -d

# 또는 Docker 이미지 직접 실행
docker run -p 8000:8000 abjin/hab-eat-ai-api-server:latest
```

### 2. 로컬 개발 환경

```bash
# 저장소 클론
git clone <repository-url>
cd hab-eat-ai-api-server

# 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 서버 실행
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 사용 예시

### Python을 이용한 API 호출

```python
import requests

# 음식 분류 예시
response = requests.post(
    "http://localhost:8000/models/food-classification/predict",
    json={"url": "https://example.com/food-image.jpg"}
)
result = response.json()
print(f"음식: {result['top1ClassName']}, 신뢰도: {result['top1Score']:.2f}")

# 운동기구 탐지 예시
response = requests.post(
    "http://localhost:8000/models/gym_equipment/predict",
    json={"url": "https://example.com/gym-image.jpg"}
)
result = response.json()
print(f"운동기구: {result['top1ClassName']}, 신뢰도: {result['top1Score']:.2f}")
```

### curl을 이용한 API 호출

```bash
# 음식 분류
curl -X POST "http://localhost:8000/models/food-classification/predict" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/food-image.jpg"}'

# 운동기구 탐지
curl -X POST "http://localhost:8000/models/gym_equipment/predict" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/gym-image.jpg"}'
```

## 🧠 모델 훈련 정보

### 음식 분류 모델
- **기반 모델**: YOLOv11-cls
- **데이터셋**: [셀렉트스타 오픈데이터셋 - 음식 사진 데이터셋](https://open.selectstar.ai/ko/computervisionlab)
- **데이터셋 특징**:
  - 고해상도: 1024x1024 픽셀 이상의 고품질 이미지
  - 다국가 음식: 한국, 서양, 아시아 등 다양한 나라의 음식 포함
  - 계층 구조 레이블링: 음식 재료별 세부 분류 정보
  - 크라우드소싱: 캐시미션 플랫폼을 통한 데이터 수집
  - 라이선스: CC BY-SA

**훈련 코드 예시:**
```python
from ultralytics import YOLO

model = YOLO("yolo11m-cls.pt")
results = model.train(
    data="./food-dataset", 
    epochs=10, 
    imgsz=640,
    batch=32
)
```

### 운동기구 탐지 모델  
- **기반 모델**: YOLOv11m
- **데이터셋**: [Roboflow Universe - workout dataset](https://universe.roboflow.com/yekwon/workout-yem51)
- **데이터셋 특징**:
  - 헬스장 환경의 실제 운동기구 이미지
  - 다양한 각도와 조명 조건
  - 바운딩 박스 어노테이션
  - 라이선스: CC BY 4.0

**훈련 코드 예시:**
```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")
results = model.train(
    data="workout-dataset/data.yaml",
    epochs=100,
    imgsz=416,
    batch=16
)
```

## 📊 성능 결과

### 음식 분류 성능
- **Top-1 Accuracy**: 88.3%
- **Top-5 Accuracy**: 97.8%
- **훈련 손실**: 0.219
- **검증 손실**: 0.509

### 운동기구 탐지 성능
- **mAP50**: 68.0%
- **mAP50-95**: 51.3%
- **Precision**: 64.9%
- **Recall**: 60.8%

## 🔬 모델 추론 예시

### TorchScript 모델 직접 사용
```python
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import requests
from io import BytesIO

# 음식 분류 모델 로드
food_model = torch.jit.load("src/ai/torchscript/food.torchscript")
transformation = Compose([Resize((640, 640)), ToTensor()])

# 이미지 전처리 및 추론
def predict_food(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = transformation(img).unsqueeze(0)
    
    with torch.no_grad():
        output = food_model(tensor)
        probabilities = output[0]
        predicted_class_id = probabilities.argmax()
        confidence = probabilities[predicted_class_id].item()
    
    return predicted_class_id, confidence

# 운동기구 탐지 모델 사용
gym_model = torch.jit.load("src/ai/torchscript/gym_equipment.torchscript")

def predict_gym_equipment(image_url):
    # 이미지 로드 및 전처리
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    tensor = Compose([Resize((416, 416)), ToTensor()])(img).unsqueeze(0)
    
    with torch.no_grad():
        output = gym_model(tensor)
        # 후처리를 통해 바운딩 박스 및 클래스 추출
        # (실제 구현은 src/service/models.py 참고)
    
    return detection_results
```

### 원본 YOLO 모델 사용 (훈련 환경에서)
```python
from ultralytics import YOLO

# 음식 분류
food_model = YOLO("food/runs/classify/train6/weights/best.pt")
results = food_model("food_image.jpg")

for result in results:
    probs = result.probs
    class_name = result.names[probs.top1]
    confidence = float(probs.top1conf)
    print(f"음식: {class_name} (신뢰도: {confidence:.2f})")

# 운동기구 탐지
gym_model = YOLO("gym/runs/detect/train/weights/best.pt")
results = gym_model("gym_image.jpg")

for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = result.names[class_id]
        print(f"운동기구: {class_name} (신뢰도: {confidence:.2f})")
```

## 🎯 지원 클래스

### 음식 분류 (90개 클래스)
바베큐, 백숙, 바게트, 반미, 육회, 비빔밥, 빙수, 불고기, 분짜, 버거, 부리또, 케이크, 빵, 카프레제, 치킨, 칠리크랩, 초콜릿, 츄러스, 커피, 아이스 커피, 쿠키, 크레페, 크로와상, 크로크 무슈, 카레, 다쿠아즈, 딤섬, 도넛, 샌드위치_에그, 에그 타르트, 골뱅이, 피시 앤 칩스, 퐁듀, 프렌치 프라이, 프렌치 토스트, 갈비, 젤라토, 김밥, 그라탱, 핫도그, 훠궈, 자장면, 잡채, 카야샌드, 케밥, 김치찌개, 전, 라자냐, 랍스터, 마카롱, 마들렌, 마파두부, 밀푀유, 머핀, 난, 나쵸, 나시고랭, 오믈렛, 삼각김밥, 팟타이, 빠에야, 팬케이크, 파스타, 파이, 피자, 팝콘, 폭찹, 파운드케이크, 푸딩, 또띠야, 라멘, 라따뚜이, 쌀국수, 리조또, 샐러드, 샌드위치, 사시미, 슈바인학센, 미역국, 시리얼, 소바, 쏨땀, 스프, 스테이크, 스시, 타코, 타코야끼, 티라미수, 또띠야, 떡볶이, 우동, 와플, 월남쌈

### 운동기구 탐지 (13개 클래스)
- Chest Press machine
- Lat Pull Down  
- Seated Cable Rows
- arm curl machine
- chest fly machine
- chinning dipping
- lateral raises machine
- leg extension
- leg press
- reg curl machine
- seated dip machine
- shoulder press machine
- smith machine

### YOLO11 객체 탐지 (80개 클래스)
COCO 데이터셋의 모든 클래스 (사람, 동물, 차량, 일상용품 등)

## ⚙️ 설정

- **서버 포트**: 8000
- **신뢰도 임계값**: 0.5
- **IoU 임계값**: 0.5 (객체 탐지용)
- **입력 이미지 크기**: 
  - 음식 분류: 640x640
  - 운동기구 탐지: 416x416
  - YOLO11: 640x640
