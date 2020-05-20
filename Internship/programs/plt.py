import torch #Tensorや様々な数学関数が含まれるパッケージ。NumPyの構造を模している
import torch.nn as nn #ニューラルネットワークを構築するための様々なデータ構造やレイヤが定義されている。
import torchvision #コンピュータビジョンにおける有名なデータセット、モデルアーキテクチャ、画像変換処理から構成される。
import torchvision.transforms as transforms #画像に関する前処理が実装されている。
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

#Hyper Parameters
num_epochs = 10 #エポック数:何周学習するか
num_classes = 10 #クラス数:0~9の10クラス
batch_size = 50 #バッチサイズ:いくつまとめて処理するか
learning_rate = 0.001 #学習率:1回の学習でどれだけ学習するか、どれだけパラメータを更新するか

#データの下処理
transform = transforms.Compose(
	[transforms.ToTensor(), 
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#dataset
train_dataset = CIFAR10('~/practice/cifar10', train=True, download=True, transform=transform)

test_dataset = CIFAR10('~/practice/cifar10', train=False, download=True, transform=transform)

#Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

#画像を表示する関数
def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()

#ランダムな訓練画像を取得
dataiter = iter(train_loader)
images, labels = dataiter.next()


#画像を表示
imshow(torchvision.utils.make_grid(images))

#ラベルを表示
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
