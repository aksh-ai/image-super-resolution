import torch
from PIL import Image
from models import Generator
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

UPSCALE_FACTOR = 2
MODEL = 'trained_models/Generator_2_64_100.pth'
IMAGE_PATH = input('Enter image path: ')
OUPUT_PATH = 'results/test/'

model = Generator(UPSCALE_FACTOR).eval().to(device)
model.load_state_dict(torch.load(MODEL))

def generate(img_path, output_path='results/test/'):
  img = Image.open(img_path)
  save_path = output_path+ 'sr_' + img_path.split('/')[-1]
  img = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
  img = model(img)
  img = transforms.ToPILImage()(img[0].detach().data.cpu())
  img.save(save_path)
  return img, save_path

_, _ = generate(IMAGE_PATH, OUPUT_PATH)