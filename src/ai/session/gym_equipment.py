from torchvision.transforms import Compose, Resize, ToTensor
import torch

session = torch.jit.load(
    "src/ai/torchscript/gym_equipment.torchscript")

input_shape = (416, 416)

transformation = Compose([Resize(input_shape), ToTensor()])

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
