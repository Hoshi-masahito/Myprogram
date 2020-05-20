import torch
import torch.nn as nn
import torchvision
import json
import numpy as np
import sys
import os
import cv2
import csv
import glob
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image

vgg16 = models.vgg16(pretrained = True)
vgg16.eval()

#Hyper Parameters
num_epochs = 10
num_classes = 5
batch_size = 32
learning_rate = 0.001

train_dir= 'madoca_magica_images/train'
valodation_dir = 'madoca_magica_images/validation'
file_name = 'vgg16_madomagi_fine'

normalize = transforms.Normalize(
	mean = [0.485, 0.456, 0.406],
	std = [0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	normalize
])

class_index = json.load(open('imagenet_class_index.json', 'r'))
labels = {int(key):value for (key, value) in class_index.items()}

files = glob.glob('./data/*')

for f in files:
	img = Image.open(f)
	img_tensor = preprocess(img)
	img_tensor.unsqueeze_(0)

	out = vgg16(Variable(img_tensor))

	out = nn.functional.softmax(out, dim = 1)
	out = out.data.numpy()

	maxid = np.argmax(out)
	maxprob = np.max(out)
	label = labels[maxid]
	print(label, maxprob)
