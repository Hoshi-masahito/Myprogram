import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import json
import numpy as np
import sys
import os
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image

vgg16 = models.vgg16(pretrained = True)
vgg16.eval()

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

if len(sys.argv) != 2:
	print("usage: python vgg.py [image file]")
	sys.exit(1)
filename = sys.argv[1]

class_index = json.load(open('imagenet_class_index.json', 'r'))
labels = {int(key):value for (key, value) in class_index.items()}

def predict(image_file):
	img = Image.open(filename)
	img_tensor = preprocess(img)
	img_tensor.unsqueeze_(0)

	out = vgg16(Variable(img_tensor))

	out = nn.functional.softmax(out, dim = 1)
	out = out.data.numpy()

	maxid = np.argmax(out)
	maxprob = np.max(out)
	label = labels[maxid]
	return img, label, maxprob

img, label, prob = predict(filename)
print(label, prob)
img.show()
