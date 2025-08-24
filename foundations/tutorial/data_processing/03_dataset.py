import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import random

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root = r"D:\PlantVillage", transform = transform)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_set, batch_size =32, shuffle=False)

def imshow(img, label):
    img = img.permute(1, 2, 0)  
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')
    plt.show()

classes = dataset.classes
data_iter = iter(train_loader)
images, labels = next(data_iter)

for i in range(10):
    imshow(images[i], classes[labels[i]])
