import torch
from math import log
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
	def __init__(self, channels: int = None) -> None:
		super(ResidualBlock, self).__init__()

		self.block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
								nn.BatchNorm2d(channels),
								nn.PReLU(),
								nn.Conv2d(channels, channels, kernel_size=3, padding=1),
								nn.BatchNorm2d(channels))
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.block(x)
		return x + residual

class UpSample(nn.Module):
	def __init__(self, in_feat: int = None, upscale: int = None) -> None:
		super(UpSample, self).__init__()

		self.block = nn.Sequential(nn.Conv2d(in_feat, in_feat * upscale ** 2, kernel_size=3, padding=1),
								nn.PixelShuffle(upscale),
								nn.PReLU())
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)

class Generator(nn.Module):
	def __init__(self, scale_factor: int = None) -> None:
		super(Generator, self).__init__()
		
		num_upsamples = int(log(scale_factor, 2))
		
		self.conv_block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
									nn.PReLU())
		
		self.conv_block2 = ResidualBlock(64)
		self.conv_block3 = ResidualBlock(64)
		self.conv_block4 = ResidualBlock(64)
		self.conv_block5 = ResidualBlock(64)
		self.conv_block6 = ResidualBlock(64)
		
		self.conv_block7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
									nn.BatchNorm2d(64))

		upsample_blocks = [UpSample(64, 2) for _ in range(num_upsamples)]
		upsample_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))

		self.conv_block8 = nn.Sequential(*upsample_blocks)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		block1 = self.conv_block1(x)
		block2 = self.conv_block2(block1)
		block3 = self.conv_block3(block2)
		block4 = self.conv_block4(block3)
		block5 = self.conv_block5(block4)
		block6 = self.conv_block6(block5)
		block7 = self.conv_block7(block6)
		block8 = self.conv_block8(block1 + block7)

		return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
	def __init__(self) -> None:
		super(Discriminator, self).__init__()

		self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
							nn.LeakyReLU(0.2),

							nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
							nn.BatchNorm2d(64),
							nn.LeakyReLU(0.2),

							nn.Conv2d(64, 128, kernel_size=3, padding=1),
							nn.BatchNorm2d(128),
							nn.LeakyReLU(0.2),

							nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
							nn.BatchNorm2d(128),
							nn.LeakyReLU(0.2))

		self.block2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
							nn.BatchNorm2d(256),
							nn.LeakyReLU(0.2),

							nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
							nn.BatchNorm2d(256),
							nn.LeakyReLU(0.2),

							nn.Conv2d(256, 512, kernel_size=3, padding=1),
							nn.BatchNorm2d(512),
							nn.LeakyReLU(0.2),

							nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
							nn.BatchNorm2d(512),
							nn.LeakyReLU(0.2))

		self.block3 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
							nn.Conv2d(512, 1024, kernel_size=1),
							nn.LeakyReLU(0.2),
							nn.Conv2d(1024, 1, kernel_size=1),
							nn.Sigmoid())
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		return x.view(batch_size)

if __name__ == '__main__':
	pass