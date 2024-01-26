import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 例: RGB画像用
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # デコーダ
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 画像の出力は0-1の範囲
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def evaluate_l2_norm(model, dataloader):
    l2_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            outputs = model(data)
            loss = torch.sqrt(torch.sum((outputs - data) ** 2))
            l2_loss += loss.item()
    return l2_loss / len(dataloader)



from torchvision import transforms
from PIL import Image
import torch

def load_and_transform_image(image_path, size=(200, 200)):
    # 画像を読み込む
    image = Image.open(image_path).convert('RGB')  # グレースケール画像の場合は 'L' を使用

    # 画像の前処理（リサイズ、テンソルへの変換、正規化など）
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    # 変換を適用
    image = transform(image)

    # バッチ次元を追加（PyTorchモデルはバッチ入力を想定）
    image = image.unsqueeze(0)

    return image


def evaluate_image(image_path, model):
    # 画像の読み込みと前処理
    image = load_and_transform_image(image_path)

    # モデルを評価モードに設定
    model.eval()

    # 推論（無勾配モード）
    with torch.no_grad():
        reconstructed = model(image)

    # L2ノルムの計算
    l2_norm = torch.sqrt(torch.sum((reconstructed - image) ** 2))

    return l2_norm.item()


model = Autoencoder()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# dataset = ImageDataset(folder_path='output/fluo')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# num_epochs = 10
# for epoch in range(num_epochs):
#     for data in dataloader:
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, data)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# torch.save(model.state_dict(), 'model.pth')

model.load_state_dict(torch.load('model.pth'))  # 学習済みモデルの重みを読み込み

# 評価実行
image_path = 'output_positive/fluo/1.png'
l2_norm_loss = evaluate_image(image_path, model)
print(f"L2 Norm Loss for the image: {l2_norm_loss:.4f}")
