import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),               
    transforms.RandomHorizontalFlip(p = 0.5),      
    transforms.RandomRotation(20),               
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
    transforms.ToTensor(),                      
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))       
])

data_dir = r"D:\PlantVillage"
dataset = datasets.ImageFolder(root = data_dir, transform = transform)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_ds, batch_size = 32, shuffle = False)

print(f"Total images: {len(dataset)}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)   
print("Batch labels shape:", labels.shape)

classes = dataset.classes
fig, axs = plt.subplots(2, 4, figsize=(10, 6))
for i, ax in enumerate(axs.flat):
    img = images[i].permute(1, 2, 0) 
    img = (img * 0.5) + 0.5          
    ax.imshow(img.numpy())
    ax.set_title(classes[labels[i]], fontsize = 8)
    ax.axis("off")

plt.show()
