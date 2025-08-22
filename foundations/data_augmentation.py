import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

img_path = r"foundations/raw/leaf1.jpg"
img = Image.open(img_path)

transform = transforms.Compose([
    transforms.RandomRotation(30),         
    transforms.RandomHorizontalFlip(p = 0.5), 
    transforms.ColorJitter(brightness = 0.3), 
    transforms.RandomResizedCrop(224, scale = (0.8, 1.0)) 
])

fig, axs = plt.subplots(1, 6, figsize = (15, 5))
axs[0].imshow(img)
axs[0].axis("off")
axs[0].set_title("Original")
for i in range(1, 6):
    aug_img = transform(img)
    axs[i].imshow(aug_img)
    axs[i].axis("off")
    axs[i].set_title(f"Aug {i}")

plt.show()
