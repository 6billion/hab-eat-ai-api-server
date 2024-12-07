import onnxruntime
from torchvision.transforms import Compose, Resize, ToTensor

transformation = Compose(
    [
        Resize((416, 416)),
        ToTensor(),
    ]
)

session = onnxruntime.InferenceSession(
    "src/ai/onnx/gym_equipment.onnx")


labels = ['Chest Press machine',
          'Lat Pull Down',
          'Seated Cable Rows',
          'arm curl machine',
          'chest fly machine',
          'chinning dipping',
          'lateral raises machine',
          'leg extension',
          'leg press',
          'reg curl machine',
          'seated dip machine',
          'shoulder press machine',
          'smith machine']
