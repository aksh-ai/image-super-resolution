import torch
from PIL import Image
from models import Generator
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

UPSCALE_FACTOR = 4
MODEL = 'trained_models/generator_4_100.pth'
IMAGE_PATH = str(input('Enter image path: '))
OUTPATH = 'results/test/'

model = Generator(UPSCALE_FACTOR).eval().to(device)
model.load_state_dict(torch.load(MODEL))

img = Image.open(IMAGE_PATH)
img = (transforms.toTensor()(img)).unsqueeze(0).to(device)

sr_image = model(img)

sr_image = transforms.ToPILImage()(sr_image[0].data.cpu())

sr_image.save(OUTPATH + 'sr_' + IMAGE_PATH.split('/')[-1])