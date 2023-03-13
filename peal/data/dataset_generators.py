import code # code.interact(local=dict(globals(), **locals()))
import random
import pygame
import torch
import torchvision
import json
import os
import shutil
import random
import numpy as np
import pkgutil

from PIL import Image
from torchvision.transforms import ToTensor
from IPython.core.debugger import set_trace # set_trace()

from peal.data.dataloaders import get_dataloader
from peal.utils import get_project_resource_dir

class MNISTConfounderDatasetGenerator:
	def __init__(self, dataset_name, mnist_dir = 'datasets/mnist', digits = ['0', '8']):
		'''

		'''
		self.dataset_name = dataset_name
		self.dataset_dir = os.path.join('datasets', self.dataset_name)
		self.mnist_dir = mnist_dir
		self.digits = digits

	def generate_dataset(self):
		'''
		
		'''
		shutil.rmtree(self.dataset_dir, ignore_errors = True)
		os.makedirs(self.dataset_dir)
		os.makedirs(os.path.join(self.dataset_dir, 'imgs'))
		os.makedirs(os.path.join(self.dataset_dir, 'masks'))

		hint_np = np.zeros([32,32], dtype = np.uint8)
		hint_np[12:20,12:20] = 255 * np.ones([8,8], dtype = np.uint8)
		hint = Image.fromarray(hint_np)

		confounder_np = np.stack([np.ones([32,32], dtype=np.uint8), np.zeros([32,32], dtype=np.uint8), np.zeros([32,32], dtype=np.uint8)], axis = -1)

		attributes = ['ImgName', 'Feature', 'Confounder']
		lines_out = [','.join(attributes)]
		for digit in self.digits:
			for it, img_name in enumerate(os.listdir(os.path.join(self.mnist_dir, digit))):
				if it % 100 == 0:
					print(it)

				img = Image.open(os.path.join(self.mnist_dir, digit, img_name)).resize([32,32])
				img_np = np.array(img)
				img_np = np.expand_dims(img_np, -1)
				img_np = np.tile(img_np, [1,1,3])
				background_intensity = np.random.randint(0, 255)
				img_np = np.maximum(img_np, background_intensity * confounder_np)
				has_confounder = bool(background_intensity >= 128)

				line = [img_name, str(int(digit == self.digits[1])), str(int(has_confounder))]
				lines_out.append(','.join(line))
				Image.fromarray(img_np).save(os.path.join(self.dataset_dir, 'imgs', img_name))
				hint.save(os.path.join(self.dataset_dir, 'masks', img_name))

		open(os.path.join(self.dataset_dir, 'data.csv'), 'w').write('\n'.join(lines_out))


class ConfounderDatasetGenerator:
	'''

	'''
	def __init__(self, base_dataset_dir, dataset_name = None, label_dir = None, delimiter = ',', confounder_type = 'intensity', num_samples = 40000):
		'''

		'''
		self.base_dataset_dir = base_dataset_dir
		self.confounder_type = confounder_type
		if dataset_name is None:
			self.dataset_name = os.path.split(base_dataset_dir)[-1] + '_' + self.confounder_type

		else:
			self.dataset_name = dataset_name

		if label_dir is None:
			self.label_dir = os.path.join(base_dataset_dir, 'data.csv')

		else:
			self.label_dir = label_dir

		self.delimiter = delimiter
		self.dataset_dir = os.path.join('datasets', self.dataset_name)
		self.num_samples = num_samples

	def generate_dataset(self):
		'''
		
		'''
		shutil.rmtree(self.dataset_dir, ignore_errors = True)
		os.makedirs(self.dataset_dir)
		os.makedirs(os.path.join(self.dataset_dir, 'imgs'))
		if self.confounder_type == 'copyrighttag':
			os.makedirs(os.path.join(self.dataset_dir, 'masks'))

		raw_data = open(self.label_dir, 'r').read().split('\n')
		attributes = raw_data[1].split(self.delimiter)
		attributes.remove('')
		attributes.append('Confounder')
		attributes.append('ConfounderStrength')
		data = []
		instance_names = []
		for line in raw_data[2:-1]:
			instance_attributes = line.split(self.delimiter)
			while '' in instance_attributes:
				instance_attributes.remove('')
			instance_attributes_int = list(map(lambda x: bool(max(0, int(x))), instance_attributes[1:]))
			instance_names.append(instance_attributes[0])
			data.append(instance_attributes_int)
		
		num_has_confounder = 0
		lines_out = ['ImgPath,' + ','.join(attributes)]

		if self.confounder_type == 'copyrighttag':
			resource_dir = get_project_resource_dir()
			copyright_tag = np.array(Image.open(os.path.join(resource_dir, 'imgs', 'copyright_tag.png')).resize([50,50]))
			copyright_tag = np.concatenate([np.ones([50, 120, 3], dtype = np.uint8), copyright_tag, np.ones([50, 8, 3], dtype = np.uint8)], axis = 1)
			copyright_tag = 255 * np.concatenate([np.ones([160, 178, 3], dtype = np.uint8), copyright_tag, np.ones([8, 178, 3], dtype = np.uint8)], axis = 0)

			copyright_tag_bg = np.ones([50,50,3], dtype = np.uint8)
			copyright_tag_bg = np.concatenate([np.zeros([50, 120, 3], dtype = np.uint8), copyright_tag_bg, np.zeros([50, 8, 3], dtype = np.uint8)], axis = 1)
			copyright_tag_bg = 255 * np.concatenate([np.zeros([160, 178, 3], dtype = np.uint8), copyright_tag_bg, np.zeros([8, 178, 3], dtype = np.uint8)], axis = 0)

		attribute_vs_no_attribute_idxs = np.zeros([2], dtype=np.int32)
		for sample_idx in range(self.num_samples):
			if sample_idx % 100 == 0:
				print(sample_idx)
				open(os.path.join(self.dataset_dir, 'data.csv'), 'w').write('\n'.join(lines_out))

			has_attribute = int(sample_idx  % 4 == 0 or sample_idx  % 4 == 1)
			has_confounder = bool(sample_idx % 2 == 0)

			while not data[attribute_vs_no_attribute_idxs[has_attribute]][attributes.index('Blond_Hair')] == has_attribute:
				attribute_vs_no_attribute_idxs[has_attribute] += 1

			name = instance_names[attribute_vs_no_attribute_idxs[has_attribute]]
			img = Image.open(os.path.join(self.base_dataset_dir, name))
			sample = data[attribute_vs_no_attribute_idxs[has_attribute]]

			if sample_idx < 0.9 * self.num_samples:
				confounder_intensity = random.uniform(0, 1)

			else:
				confounder_intensity = 1.0

			if self.confounder_type == 'intensity':
				intensity_change = 64 * confounder_intensity * (2 * int(has_confounder) - 1)
				img = np.array(img)
				img = (img + intensity_change + 64) * (255 / (255 + 2 * 64))
				img_out = Image.fromarray(np.array(img, dtype = np.uint8))
			
			elif self.confounder_type == 'color':
				color_change = 64 * confounder_intensity * (2 * int(has_confounder) - 1)
				img = np.array(img)
				img = np.stack([(img[:,:,0] + color_change + 64) * (255 / (255 + 2 * 64)), img[:,:,1], (img[:,:,2] - color_change + 64) * (255 / (255 + 2 * 64))], axis = -1)
				img_out = Image.fromarray(np.array(img, dtype = np.uint8))
			
			elif self.confounder_type == 'copyrighttag':
				img_copyrighttag = np.maximum(np.array(img), copyright_tag_bg)
				img_copyrighttag = np.minimum(np.array(img_copyrighttag), copyright_tag)
				alpha = 0.5 + 0.5 * confounder_intensity * (2 * int(has_confounder) - 1)
				img = alpha * img_copyrighttag + (1 - alpha) * np.array(img)
				img_out = Image.fromarray(np.array(img, dtype = np.uint8))

			img_out.save(os.path.join(self.dataset_dir, 'imgs', name))
			if self.confounder_type == 'copyrighttag':
				mask = Image.fromarray(np.array(np.abs(np.array(copyright_tag_bg, dtype = np.float32) / 255 - 1) * 255, dtype = np.uint8))
				mask.save(os.path.join(self.dataset_dir, 'masks', name))

			sample.append(has_confounder)
			sample.append(confounder_intensity)
			lines_out.append(name + ',' + ','.join(list(map(lambda x: str(float(x)), sample))))
			attribute_vs_no_attribute_idxs[has_attribute] += 1

		open(os.path.join(self.dataset_dir, 'data.csv'), 'w').write('\n'.join(lines_out))