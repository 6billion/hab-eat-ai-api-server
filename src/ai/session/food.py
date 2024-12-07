from torchvision.transforms import Compose, Resize, ToTensor
import torch

session = torch.jit.load(
    "src/ai/torchscript/food.torchscript")

input_shape = (640, 640)

transformation = Compose([Resize(input_shape), ToTensor()])

labels = ['BBQ', 'baek_sook', 'baguette', 'banh_mi', 'beef_tartare', 'bibimbap', 'bingsu', 'bulgogi', 'bunza', 'burger', 'burrito', 'cake', 'cannoli', 'caprese', 'chicken', 'chili_crab', 'chocolate', 'churros', 'coffee_hot', 'coffee_ice', 'cookie', 'crepe', 'croissant', 'croque_monsieur', 'curry', 'dacquoise', 'dim_sum', 'donut', 'egg_benedict', 'egg_tart', 'escargot', 'fish_and_chips', 'fondue', 'french_fries', 'french_toast', 'galbi', 'gelato', 'gimbap', 'gratin', 'hot_dog', 'hot_pot', 'jajangmyeon', 'japchae', 'kaya_toast', 'kebap',
          'kimchi_stew', 'korean_pancake', 'lasana', 'lobster', 'macaroon', 'madeleine', 'mapa_tofu', 'milfeuille', 'muffin', 'naan', 'nachos', 'nasi_goreng', 'omelet', 'onigiri', 'pad_thai', 'paea', 'pan_cake', 'pasta', 'pie', 'pizza', 'popcorn', 'pork_chop', 'pound_cake', 'pudding', 'quesadilla', 'ramen', 'ratatouille', 'rice_noodle', 'risotto', 'salad', 'sandwich', 'sashimi', 'schweinshaxe', 'seaweed_soup', 'serial', 'soba', 'somtam', 'soup', 'steak', 'sushi', 'taco', 'takoyaki', 'tiramisu', 'tortilla', 'tteokbokki', 'udon', 'waffle', 'wolnam_ssam']
