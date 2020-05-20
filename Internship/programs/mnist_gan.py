import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt

from IPython.display import Image

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')

def initialize_weights(model):
  for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(62, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128,7,7)
        x = self.deconv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 7*7*128)
        x = self.fc(x)
        return x

#ハイパーパラメータ
batch_size = 128
lr = 0.0002
z_dim = 62
num_epochs = 25
sample_num = 16
log_dir = './logs'

#ネットワークの初期化
G = Generator()
D = Discriminator()
if cuda:
    G.cuda()
    D.cuda()

#Optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

#損失関数
criterion = nn.BCELoss()

#データセットのロード
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST('~/practice/data/mnist', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(D, G, criterion, D_optimizer, G_optimizer, data_loader):
    D.train()
    G.train()

    y_real = Variable(torch.ones(batch_size, 1))
    y_fake = Variable(torch.zeros(batch_size, 1))
    if cuda:
        y_real = y_real.cuda()
        y_fake = y_fake.cuda()
 
    D_running_loss = 0
    G_running_loss = 0
    for batch_idx, (real_images, _) in enumerate(data_loader):
        if real_images.size()[0] != batch_size:
            break

        z = torch.rand((batch_size, z_dim))
        if cuda:
            real_images, z = real_images.cuda(), z.cuda()
        real_images, z = Variable(real_images), Variable(z)

        #Discriminatorの更新
        D_optimizer.zero_grad()

        D_real = D(real_images)
        D_real_loss = criterion(D_real, y_real)

        fake_images = G(z)
        D_fake = D(fake_images.detach())
        D_fake_loss = criterion(D_fake, y_fake)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        D_running_loss += D_loss.item()

        #Generatorの更新
        z = torch.rand((batch_size, z_dim))
        if cuda:
            z = z.cuda()
        z = Variable(z)

        G_optimizer.zero_grad()

        fake_images = G(z)
        D_fake = D(fake_images)
        G_loss = criterion(D_fake, y_real)
        G_loss.backward()
        G_optimizer.step()
        G_running_loss += G_loss.item()

    D_running_loss /= len(data_loader)
    G_running_loss /= len(data_loader)

    return D_running_loss, G_running_loss

def generate(epoch, G, log_dir='logs'):
    G.eval()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sample_z = torch.rand((64, z_dim))
    if cuda:
        sample_z = sample_z.cuda()
    with torch.no_grad():
        sample_z = Variable(sample_z)

    samples = G(sample_z).data.cpu()
    save_image(samples, os.path.join(log_dir, 'epoch_%03d.png'%(epoch)))

history = {}
history['D_loss'] = []
history['G_loss'] = []
for epoch in range(num_epochs):
    D_loss, G_loss = train(D, G, criterion, D_optimizer, G_optimizer, data_loader)

    print('epoch %d, D_loss:%4f, G_loss:%4f'%(epoch+1, D_loss, G_loss))
    history['D_loss'].append(D_loss)
    history['G_loss'].append(G_loss)

    if epoch == 0 or epoch == 9 or epoch == 24:
        generate(epoch+1, G, log_dir)
        torch.save(G.state_dict(), os.path.join(log_dir, 'G_%03d.pth'%(epoch+1)))
        torch.save(D.state_dict(), os.path.join(log_dir, 'D_%03d.pth'%(epoch+1)))

with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
    pickle.dump(history, f)


with open(os.path.join(log_dir, 'history.pkl'), 'rb') as f:
    history = pickle.load(f)

D_loss, G_loss = history['D_loss'], history['G_loss']
plt.plot(D_loss, label='D_loss')
plt.plot(G_loss, label='G_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid()

Image('logs/epoch_001.png')
Image('logs/epoch_010.png')
Image('logs/epoch_025.png')
