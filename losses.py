import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16

class TotalVariationLoss(nn.Module):
	def __init__(self, loss_weight: int = 1) -> None:
		super(TotalVariationLoss, self).__init__()
		self.loss_weight = loss_weight
	
	@staticmethod
	def tensor_size(t: torch.Tensor) -> torch.Tensor:
		return t.size()[1] * t.size()[2] * t.size()[3]

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size()[0]
		h = x.size()[2]
		w = x.size()[3]
		
		count_h = self.tensor_size(x[:, :, 1:, :])
		count_w = self.tensor_size(x[:, :, :, 1:])
		
		h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
		w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
		
		tv_loss = (self.loss_weight * 2 * ((h_tv / count_h) + (w_tv / count_w))) / batch_size

		return tv_loss

class GeneratorLoss(nn.Module):
	def __init__(self) -> None:
		super(GeneratorLoss, self).__init__()

		vgg = vgg16(pretrained=True)

		features = nn.Sequential(*(list(vgg.features.children())[:36])).eval()

		for param in features.parameters():
			param.requires_grad = False
		
		self.net = features
		self.mse_loss = nn.MSELoss()
		self.tv_loss = TotalVariationLoss()
	
	def forward(self, out_labels: torch.Tensor, out_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
		adversarial_loss = torch.mean(1 - out_labels)

		perception_loss = self.mse_loss(self.net(out_images), self.net(target_images))

		image_loss = self.mse_loss(out_images, target_images)

		tv_loss = self.tv_loss(out_images)

		total_loss = image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss

		return total_loss

if __name__ == '__main__':
	pass