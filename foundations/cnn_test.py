import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Config
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 3
DATA_ROOT = r"D:\dataset_split"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# DataLoader
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  
])

train_dataset = datasets.ImageFolder(root = f"{DATA_ROOT}/train", transform = transform)
val_dataset   = datasets.ImageFolder(root = f"{DATA_ROOT}/val", transform = transform)
test_dataset  = datasets.ImageFolder(root = f"{DATA_ROOT}/test", transform = transform)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader   = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_loader  = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# Simple CNN 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # img(channel, height, width)
            # Input: 32 img(3, 224, 224)

            # Conv2d(in_channels, out_channels, kernel_size, padding)
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),  
            # Output: 32 img(32, 224, 224)
            # 32 kernels with size 3 x 3 x 3(each slides 1 pixel over img => 1 feature map 224 x 224) => 32 feature maps 224 x 224
            # The way to compute the value of each pixel in the feature map is by taking the dot product of the kernel and the input image patch it covers.

            nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            # img(32, 32, 112, 112)
            
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), 
            # img(32, 64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(2,2)   
            # img(32, 64, 56, 56)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #img
            nn.Linear(64*56*56, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes).to(device)
print(model)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

# Train & Validate
for epoch in range(EPOCHS):
    # Train 
    model.train()
    running_loss, running_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = running_correct / len(train_dataset)

    # Validate
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# Test 
model.eval()
test_correct = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        test_correct += (outputs.argmax(1) == labels).sum().item()

test_acc = test_correct / len(test_dataset)
print(f"[Test Accuracy] {test_acc:.4f}")
