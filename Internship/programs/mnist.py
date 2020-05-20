import torch #Tensorや様々な数学関数が含まれるパッケージ。NumPyの構造を模している
import torch.nn as nn #ニューラルネットワークを構築するための様々なデータ構造やレイヤが定義されている。
import torchvision #コンピュータビジョンにおける有名なデータセット、モデルアーキテクチャ、画像変換処理から構成される。
import torchvision.transforms as transforms #画像に関する前処理が実装されている。
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyper Parameters
num_epochs = 10 #エポック数:何周学習するか
num_classes = 10 #クラス数:0~9の10クラス
batch_size = 50 #バッチサイズ:いくつまとめて処理するか
learning_rate = 0.001 #学習率:1回の学習でどれだけ学習するか、どれだけパラメータを更新するか

#dataset
train_dataset = MNIST('~/practice/data/mnist', train=True, download=True, transform=transforms.ToTensor())

test_dataset = MNIST('~/practice/data/mnist', train=False, download=True, transform=transforms.ToTensor())

#Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

class ConvNet(nn.Module):
	def __init__ (self, num_classes=10):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),#28*28*16
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),#28*28*16
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)) #14*14*16
		self.layer3 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),#14*14*32
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),#16*16*32
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))#7*7*32
		self.layer5 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#8*8*64
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),#8*8*64
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))#4*4*64
		self.fc1 = nn.Sequential(
			nn.Linear(4*4*64, 2048),
			nn.ReLU(),
			nn.Dropout2d(p=0.5))
		self.fc2 = nn.Sequential(
			nn.Linear(2048, num_classes),
			nn.Dropout2d(p=0.5))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

model = ConvNet(num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss() #交差エントロピー誤差
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Adamによる最適化

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = images.to(device)
		labels = labels.to(device)

		# Forward pass
		outputs = model(images) #出力の計算
		loss = criterion(outputs, labels) #損失関数の計算

		# Backward and optimize
		optimizer.zero_grad() #勾配の初期化
		loss.backward() #勾配の計算(誤差逆伝搬)
		optimizer.step() #重みの更新

		if (i+1) % 100 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
			   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad(): #With:ファイル処理　
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
