import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from data_loader import load_database
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()

def box_plot_function(data:list[np.ndarray] | list[float|int], 
                      labels:list[str], 
                      xlabel:str, 
                      ylabel:str, 
                      save_name:str) -> None:
    fig = plt.figure(figsize=[10,7])
    plt.boxplot(data,sym="")
    for i, d in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(d)) 
        plt.plot(x, d, 'o', alpha=0.5)  
    plt.xticks([i+1 for i in range(len(data))], [f"{i}" for i in labels])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    fig.savefig(f"{save_name}.png",dpi = 500)

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
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
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

def load_and_transform_image(image_path, size=(200, 200)):
    image = Image.open(image_path).convert('RGB') 
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def evaluate_image(image_path, model):
    image = load_and_transform_image(image_path)

    model.eval()

    with torch.no_grad():
        reconstructed = model(image)
    l2_norm = torch.sqrt(torch.sum((reconstructed - image) ** 2))
    return l2_norm.item()

def autoencoder(train_mode: bool = True):
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = ImageDataset(folder_path='ctrl/fluo')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'model.pth')

    model.load_state_dict(torch.load('model.pth'))  # 学習済みモデルの重みを読み込み
    data = []
    for i in range(1,len(os.listdir('data/fluo'))):
        image_path = f'data/fluo/{i}.png'
        l2_norm_loss = evaluate_image(image_path, model)
        if l2_norm_loss <24:
            print(f"Image {i} is an outlier")
        print(f"L2 Norm Loss for the image: {l2_norm_loss:.4f}")
        data.append(l2_norm_loss)
    
    ctrl = []
    for i in range(1,len(os.listdir('ctrl/fluo'))):
        image_path = f'ctrl/fluo/{i}.png'
        l2_norm_loss = evaluate_image(image_path, model)
        if l2_norm_loss <24:
            print(f"Image {i} is an outlier")
        print(f"L2 Norm Loss for the image: {l2_norm_loss:.4f}")
        data.append(l2_norm_loss)
        ctrl.append(l2_norm_loss)
    box_plot_function([np.array(ctrl),np.array(data)], ["Ctrl.","data"], "Image", "L2 Norm Loss", "l2_norm_loss")

    



if __name__ == '__main__':
    #load control
    load_database("ctrl", "sk320gen0min.db")
    #load data
    load_database("data", "sk320tri90min.db")
    autoencoder(train_mode=True)