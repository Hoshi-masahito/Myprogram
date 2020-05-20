import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import os
import random
import time

outf = './result_lsgan'
if not os.path.exists(outf):
    os.makedirs(outf)

#設定
nz = 100 #画像を生成するための特徴マップの次元数
nch_g = 64 #Generatorの最終層の入力チャネル数
nch_d = 64 #Discriminatorの先頭層の出力チャネル数
workers = 2 #データロードに使用するコア数
batch_size = 50 #バッチサイズ
n_epoch = 10 #エポック数
lr = 0.0002 #学習率
beta1 = 0.5 #最適化関数に使用するパラメータ

display_interval = 100 #学習経過を表示するスパン

#乱数のシードを固定
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#MNISTのトレーニングデータセットを読みこむ
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])

dataset = torchvision.datasets.CIFAR10(root='~/practice/data/cifar', train=True, download=True, transform=transform)

#データローダーの作成
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#学習に使用するデバイスを得る。可能ならGPUを使用する
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=64, nch=1):
        super(Generator, self).__init__()
        
        #ネットワーク構造の定義
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g*8, kernel_size = 2, stride = 1, padding = 0),
                nn.BatchNorm2d(nch_g*8),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g*8, nch_g*4, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(nch_g*4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g*4, nch_g*2, kernel_size = 4, stride = 2, padding = 1),
                nn.BatchNorm2d(nch_g*2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g*2, nch_g, kernel_size = 2, stride = 2, padding = 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, kernel_size = 4, stride = 2, padding = 1), 
                nn.Tanh()
            ),
    ])

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

class Discriminator(nn.Module):
    def __init__(self, nch = 1, nch_d = 64):
        super(Discriminator, self).__init__()
        
        #ニューラルネットワークの定義
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nch, nch_d*2, kernel_size=3, stride=3, padding=0),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.Conv2d(nch_d*2, nch_d*4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nch_d*4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                nn.Conv2d(nch_d*4, nch_d*8, kernel_size=3, stride=3, padding=1),
                nn.BatchNorm2d(nch_d*8),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Conv2d(nch_d*8, 1, kernel_size=3, stride=1, padding=0)
        ])

    def forward(self, x): #xは本物画像or偽物画像
        for layer in self.layers:
            x = layer(x)
        return x.squeeze() #不要な次元を削除

#重みを初期化する関数を定義
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1: #畳込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1: #転置畳み込み層の場合
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1: #バッチ正規化の場合
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Generatorの作成
netG = Generator(nz=nz+10, nch_g = nch_g).to(device) #10はn_class=10を指す。出しわけに必要なラベル情報
netG.apply(weights_init)

#Discriminatorの作成
netD = Discriminator(nch=1+10).to(device) #10はn_class=10を指す。分類に必要なラベル情報
netD.apply(weights_init)

def onehot_encode(label, device, n_class=10):
    eye = torch.eye(n_class, device=device)
    return eye[label].view(-1, n_class, 1, 1) #連結するために(Batchsize, n_class, 1,1)のTensorにして戻す

#画像とラベルを連結する
def concat_image_label(image, label, device, n_class=10):
    B,C,H,W = image.shape #画像のTensorの大きさを取得
    
    oh_label = onehot_encode(label, device) #ラベルをOne-hotベクトル化
    oh_label = oh_label.expand(B, n_class, H, W) #画像のサイズに合わせるようラベルを拡張
    return torch.cat((image, oh_label), dim=1) #画像とラベルをチャンネル方向(dim=1)で連結する

#ノイズとラベルを連結する
def concat_noise_label(noise, label, device):
    oh_label = onehot_encode(label, device)
    return torch.cat((noise, oh_label), dim=1)

#画像確認用のノイズとラベルを設定
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device) #ノイズの生成
fixed_label = [i for i in range(10)] * (batch_size // 10) #0~9の値で繰り返す(5回)
fixed_label = torch.tensor(fixed_label, dtype=torch.long, device=device) #torch.longはint64を指す
fixed_noise_label = concat_noise_label(fixed_noise, fixed_label, device) #確認用のノイズとラベルを連結

criterion = nn.MSELoss() #二乗誤差損失

#最適化関数
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)

#エポックごとのロスを可視化
def plot_loss(G_loss_mean, D_loss_mean, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss - EPOCH "+ str(epoch))
    plt.plot(G_loss_mean, label="G")
    plt.plot(D_loss_mean, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#全体でのlossを可視化
def plot_loss_average(G_loss_mean, D_loss_mean):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss - EPOCH ")
    plt.plot(G_loss_mean, label="G")
    plt.plot(D_loss_mean, label="D")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

G_loss_mean = [] #学習全体でのLossを格納するリスト
D_loss_mean = []
epoch_time = [] #時間の計測結果を格納するリスト

for epoch in range(n_epoch):
    start = time.time() #時間の計測を開始
    G_losses = []
    D_losses = []
    
    for itr, data in enumerate(dataloader):
        #本物画像のロード  
        real_image = data[0].to(device)
        real_label = data[1].to(device)
        real_image_label = concat_image_label(real_image, real_label, device) #画像とラベルを連結
        
        #贋作画像生成用のノイズとラベルを準備
        sample_size = real_image.size(0) #0は1次元目(バッチ数)を指す
        noise = torch.randn(sample_size, nz, 1, 1, device=device)
        fake_label = torch.randint(10, (sample_size,), dtype=torch.long, device=device)
        fake_noise_label = concat_noise_label(noise, fake_label, device) #ノイズとラベルを連結

        #識別の目標値を設定
        real_target = torch.full((sample_size,), 1, device=device) #本物は1
        fake_target = torch.full((sample_size,), 0, device=device) #偽物は0
        
        #Discriminatorの更新
        netD.zero_grad() #勾配の初期化
        
        output = netD(real_image_label) #順伝搬させて出力(分類結果)を計算
        errD_real = criterion(output, real_target) #本物画像に対する損失値
        D_x = output.mean().item()
        
        fake_image = netG(fake_noise_label) #生成器Gで贋作画像を生成
        fake_image_label = concat_image_label(fake_image, fake_label, device) #贋作画像とラベルを連結
        output = netD(fake_image_label.detach()) #判別器Dで贋作画像とラベルの組み合わせに対する識別信号を出力
        errD_fake = criterion(output, fake_target) #偽物画像に対する損失値
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake #Dの損失の合計
        errD.backward() #誤差逆伝搬
        optimizerD.step() #Dのパラメータを更新
        
        #Generatorの更新
        netG.zero_grad()
        
        output = netD(fake_image_label) #更新した判別器で改めて判別結果を出力
        errG = criterion(output, real_target) #贋作画像を本物と誤認させたいので、目標値はreal_target
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        #lossの保存
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        #学習経過の表示
        if itr % display_interval == 0:
            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'.format(epoch + 1, n_epoch, itr + 1, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        if epoch == 0 and itr == 0:
            vutils.save_image(fake_image.detach(), '{}/real_samples.png'.format(outf), normalize=True, nrow=10)
            
    #確認用画像の生成(1エポックごと)
    fake_image = netG(fixed_noise_label)
    vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1), normalize=True, nrow=10)
    
    #lossの平均を格納
    G_loss_mean.append(sum(G_losses) / len(G_losses))
    D_loss_mean.append(sum(D_losses) / len(D_losses))
    
    #lossのプロット
    plot_loss(G_losses, D_losses, epoch)

#学習全体の状況をグラフ化
plot_loss_average(G_loss_mean, D_loss_mean)
