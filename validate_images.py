import os
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from models import Generator
from utils import ValidationDataset, display_transform

if __name__ == '__main__':
	torch.cuda.empty_cache()

	CROP_SIZE = 64
	UPSCALE_FACTOR = 2
	EPOCHS = 100
	OUTPUT_PATH = 'results'
	MODEL_PATH = 'models/Generator_2_64_100.pth'

	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	val_data = ValidationDataset('./data/DIV2k_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)

	val_loader = DataLoader(val_data, num_workers=4, batch_size=1, pin_memory=True, shuffle=True)

	generator = Generator(UPSCALE_FACTOR).to(device)
	generator.load_state_dict(torch.load(MODEL_PATH))

	out_path = os.path.join(OUTPUT_PATH, 'evaluation')

	indx = 1

	with torch.no_grad():
		for b, (val_lr, val_hr_restore, val_hr) in enumerate(val_loader):
			val_images = []

			lr = val_lr.to(device)
			hr = val_hr.to(device)

			sr = generator(lr)
		
			val_images.extend([display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)), display_transform()(sr.data.cpu().squeeze(0))])
					
			val_images = torch.stack(val_images)

			image = utils.make_grid(val_images, nrow=3, padding=5)
			utils.save_image(image, os.path.join(out_path, f'validate_index_{indx}.png'), padding=5)
			indx += 1