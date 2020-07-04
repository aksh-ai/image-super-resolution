import os
import time
import torch
import pyssim
import pandas as pd
from math import log10
import matplotlib.pyplot as plt
from losses import GeneratorLoss
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from utils import TrainDataset, ValidationDataset, display_transform

if __name__ == '__main__':
	torch.cuda.empty_cache()

	CROP_SIZE = 64
	UPSCALE_FACTOR = 2
	EPOCHS = 100
	BATCH_SIZE = 32
	OUTPUT_PATH = 'results'

	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	train_data = TrainDataset('./data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
	val_data = ValidationDataset('./data/DIV2k_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)

	train_loader = DataLoader(train_data, num_workers=4, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
	val_loader = DataLoader(val_data, num_workers=4, batch_size=1, pin_memory=True, shuffle=True)

	generator = Generator(UPSCALE_FACTOR).to(device)
	discriminator = Discriminator().to(device)

	criterion = GeneratorLoss().to(device)

	g_optim = torch.optim.Adam(generator.parameters(), lr=1e-5)
	d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

	results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

	training_start_time = time.time()

	print(f'Training Config - Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Crop Size: {CROP_SIZE} | Upscale Factor: {UPSCALE_FACTOR}\n')
	print(f"Generator - Number of parameters: {sum(param.numel() for param in generator.parameters())}")
	print(f"Discriminator - Number of parameters: {sum(param.numel() for param in discriminator.parameters())}\n\nStarting Training...\n")

	# val_images = []

	for epoch in range(1, EPOCHS+1):
		print(f"Epoch [{epoch:03}/{EPOCHS}]")
		
		e_start_time = time.time()
		epoch_results = {'batch_size': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

		generator.train()
		discriminator.train()

		for b, (data, target) in enumerate(train_loader):
			batch_size = data.size(0)
			epoch_results['batch_size'] += batch_size

			real_img = Variable(target).to(device)
			
			z = Variable(data).to(device)
			fake_img = generator(z)

			discriminator.zero_grad()

			real_out = discriminator(real_img).mean()
			fake_out = discriminator(fake_img.detach()).mean()

			d_loss = 1 - real_out + fake_out
			d_loss.backward()
			d_optim.step()

			discriminator.eval()

			generator.zero_grad()

			fake_img = generator(z)
			fake_out = discriminator(fake_img.detach()).mean()

			g_loss = criterion(fake_out, fake_img, real_img)
			g_loss.backward()

			g_optim.step()

			epoch_results['g_loss'] += g_loss.item() * batch_size
			epoch_results['d_loss'] += d_loss.item() * batch_size
			epoch_results['d_score'] += real_out.item() * batch_size
			epoch_results['g_score'] += fake_out.item() * batch_size
		
		print(f"Discriminator - Loss: {epoch_results['d_loss'] / epoch_results['batch_size']:.4f} | Score: {epoch_results['d_score'] / epoch_results['batch_size']:.4f}")
		print(f"Generator - Loss: {epoch_results['g_loss'] / epoch_results['batch_size']:.4f} | Score: {epoch_results['g_score'] / epoch_results['batch_size']:.4f}")

		generator.eval()

		with torch.no_grad():
			val_results = {'batch_size': 0, 'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0}
			val_images = []

			for b, (val_lr, val_hr_restore, val_hr) in enumerate(val_loader):
				batch_size = val_lr.size(0)
				val_results['batch_size'] += batch_size

				lr = val_lr.to(device)
				hr = val_hr.to(device)

				sr = generator(lr)

				batch_mse = ((sr - hr) ** 2).data.mean()
				val_results['mse'] += batch_mse * batch_size
				
				batch_ssim = pyssim.ssim(sr, hr).item()
				val_results['ssims'] += batch_ssim * batch_size

				try:
					val_results['psnr'] = 10 * log10((hr.max()**2) / (val_results['mse'] / val_results['batch_size']))
				
				except ValueError:
					val_results['psnr'] = 10 * (1 / (val_results['mse'] / val_results['batch_size']))

				val_results['ssim'] = val_results['ssims'] / val_results['batch_size']
			
				val_images.extend([display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)), display_transform()(sr.data.cpu().squeeze(0))])
			
			print(f"Validation - Structural Similarity: {val_results['ssim']:.4f} | Peak Signal to Noise Ratio: {val_results['psnr']:.4f} db")
			
			val_images = torch.stack(val_images)
			val_images = torch.chunk(val_images, val_images.size(0) // 15)

			print(f'Duration: {(time.time() - e_start_time)/60:.4} minutes | Saving results!\n')

			out_path = os.path.join(OUTPUT_PATH, 'evaluation')
		
		results['d_loss'].append(epoch_results['d_loss'] / epoch_results['batch_size'])
		results['g_loss'].append(epoch_results['g_loss'] / epoch_results['batch_size'])
		results['d_score'].append(epoch_results['d_score'] / epoch_results['batch_size'])
		results['g_score'].append(epoch_results['g_score'] / epoch_results['batch_size'])
		results['psnr'].append(val_results['psnr'])
		results['ssim'].append(val_results['ssim'])

		torch.save(generator.state_dict(), f'models/Generator_{UPSCALE_FACTOR}_{CROP_SIZE}_{epoch}.pth')
		torch.save(discriminator.state_dict(), f'models/Discriminator_{UPSCALE_FACTOR}_{CROP_SIZE}_{epoch}.pth')

		if epoch==1 or epoch==EPOCHS or epoch%10==0 and epoch != 0:
			out_path = OUTPUT_PATH
			
			data_frame = pd.DataFrame(data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'], 'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']}, index=range(1, epoch + 1))
			
			data_frame.to_csv(os.path.join(out_path, 'SRGAN_' + str(UPSCALE_FACTOR) + '_train_results.csv'), index_label='Epoch')

			indx = 1
			for image in val_images:
				image = utils.make_grid(image, nrow=3, padding=5)
				utils.save_image(image, os.path.join(out_path, f'epoch_{epoch}_index_{indx}.png'), padding=5)
				indx += 1
	
	print(f'Completed Training...\nTotal Training Duration: {(time.time() - training_start_time)/60:.4} minutes')

	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2)

	ax1[0].plot(results['d_loss'], label='Discriminator Loss')
	ax1[0].plot(results['g_loss'], label='Generator Loss')
	ax1[0].set(xlabel='Epochs', ylabel='Loss')
	ax1[0].set_title('Losses')
	ax1[0].legend(loc='best')
	ax1[0].grid()

	ax1[1].plot(results['d_score'], label='Discriminator Score')
	ax1[1].plot(results['g_score'], label='Generator Score')
	ax1[1].set(xlabel='Epochs', ylabel='Score')
	ax1[1].set_title('Scores')
	ax1[1].legend(loc='best')
	ax1[1].grid()

	ax2[0].plot(results['ssim'], label='Structural Similarity')
	ax2[0].set(xlabel='Epochs', ylabel='SSIM Score')
	ax2[0].set_title('Structural Similarity Score')
	ax2[0].legend(loc='best')
	ax2[0].grid()

	ax2[1].plot(results['psnr'], label='Peak Signal to Noise Ratio')
	ax2[1].set(xlabel='Epochs', ylabel='PSNR level')
	ax2[1].set_title('Peak Signal to Noise Ratio levels')
	ax2[1].legend(loc='best')
	ax2[1].grid()

	fig.suptitle('Training Metrics', fontsize=12)
	fig.tight_layout()
	
	plt.savefig('results/training_metrics.png')
	plt.show()

	torch.cuda.empty_cache()

	exit()