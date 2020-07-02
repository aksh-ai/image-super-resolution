import torch
from PIL import Image
from os import listdir
from os.path import join
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

def isImage(filename: str) -> bool:
	return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def validateCropSize(crop_size: int, upscale_factor: int) -> int:
	return crop_size - (crop_size % upscale_factor)

def transform(crop_size: int = None, upscale_factor: int = None, lr: bool = False, hr: bool = False) -> torch.Tensor:
	if hr == True and crop_size is not None:
		return Compose([RandomCrop(crop_size),
						ToTensor()])

	elif lr == True and upscale_factor is not None and crop_size is not None:
		return Compose([ToPILImage(),
						Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
						ToTensor()])
	
	else:
		print('Specify valid arguments for transformation')

def display_transform() -> torch.Tensor:
	return Compose([ToPILImage(mode='RGB'),
					Resize(400),
					CenterCrop(400),
					ToTensor()])

class TrainDataset(Dataset):
	def __init__(self, path: str = None, crop_size: int = None, upscale_factor: int = None) -> None:
		super(TrainDataset, self).__init__()

		self.image_names = [join(path, image) for image in listdir(path) if isImage(image)]
		self.crop_size = validateCropSize(crop_size, upscale_factor)
		self.hr_transform = transform(crop_size=self.crop_size, hr=True, lr=False, upscale_factor=None)
		self.lr_transform = transform(crop_size=self.crop_size, upscale_factor=upscale_factor, lr=True, hr=False)
	
	def __len__(self) -> int:
		return len(self.image_names)

	def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
		hr_image = self.hr_transform(Image.open(self.image_names[idx]))
		lr_image = self.lr_transform(hr_image)
		return lr_image, hr_image

class ValidationDataset(Dataset):
	def __init__(self, path: str = None, crop_size: int = None, upscale_factor: int = None) -> None:
		super(ValidationDataset, self).__init__()

		self.image_names = [join(path, img) for img in listdir(path) if isImage(img)]
		self.upscale_factor = upscale_factor
		self.crop_size = validateCropSize(crop_size, upscale_factor)
	
	def __len__(self) -> int:
		return len(self.image_names)
	
	def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		hr_image = Image.open(self.image_names[idx])

		lr_transform = Resize(self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
		hr_transform = Resize(self.crop_size, interpolation=Image.BICUBIC)

		hr_image = CenterCrop(self.crop_size)(hr_image)

		lr_image = lr_transform(hr_image)
		hr_interpolated = hr_transform(lr_image)

		return ToTensor()(lr_image), ToTensor()(hr_interpolated), ToTensor()(hr_image)

class TestDataset(Dataset):
	def __init__(self, path: str = None, upscale_factor: int = None) -> None:
		super(TestDataset, self).__init__()

		self.upscale_factor = upscale_factor

		self.lr_path = path + '/SRF_' + str(upscale_factor) + '/data/'
		self.hr_path = path + '/SRF_' + str(upscale_factor) + '/target/'

		self.lr_images = [join(self.lr_path, img) for img in listdir(self.lr_path) if isImage(img)]
		self.hr_images = [join(self.hr_path, img) for img in listdir(self.hr_path) if isImage(img)]
	
	def __len__(self) -> int:
		return len(self.lr_images)
	
	def __getitem__(self, idx: int) -> (str, torch.Tensor, torch.Tensor, torch.Tensor):
		image_name = self.lr_images[idx].split('/')[-1]
		
		lr_image = Image.open(self.lr_images[idx])
		width, height = lr_image.size

		hr_image = Image.open(self.hr_images[idx])
		transform = Resize((self.upscale_factor * height, self.upscale_factor * width), interpolation=Image.BICUBIC)
		hr_interpolated = transform(lr_image)

		return image_name, ToTensor()(lr_image), ToTensor()(hr_interpolated), ToTensor()(hr_image)

if __name__ == '__main__':
	pass