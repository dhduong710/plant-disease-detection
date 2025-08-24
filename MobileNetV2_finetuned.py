import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Config

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS_PHASE_A = 10   
EPOCHS_PHASE_B = 5   
LR_PHASE_A = 1e-3
LR_PHASE_B = 1e-4
DATA_ROOT = r"D:\dataset_split"
N_LAST_LAYERS = 20   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Data

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root = f"{DATA_ROOT}/train", transform = train_transform)
val_dataset   = datasets.ImageFolder(root = f"{DATA_ROOT}/val", transform = val_transform)
test_dataset  = datasets.ImageFolder(root = f"{DATA_ROOT}/test", transform = val_transform)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader   = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_loader  = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)


# Model: MobileNetV2

mobilenet = models.mobilenet_v2(pretrained=True)

# Replace classifier
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, num_classes)
model = mobilenet.to(device)

criterion = nn.CrossEntropyLoss()


# Training function

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs, phase_name = "Phase"):
    for epoch in range(epochs):
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

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"{phase_name} Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return model


# Phase A: Train classifier

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier[1].parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.classifier[1].parameters(), lr = LR_PHASE_A)
model = train_model(model, criterion, optimizer, train_loader, val_loader,
                    epochs = EPOCHS_PHASE_A, phase_name = "Phase A")


# Phase B: Fine-tuning last N layers

# Freeze all first
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze last N layers + classifier
for name, param in list(model.named_parameters())[-N_LAST_LAYERS:]:
    param.requires_grad = True

# Optimizer (only train unfrozen layers)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = LR_PHASE_B)

# Train Phase B
model = train_model(model, criterion, optimizer, train_loader, val_loader,
                    epochs = EPOCHS_PHASE_B, phase_name = "Phase B")


# Test

model.eval()
test_loss, test_correct = 0.0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * imgs.size(0)
        test_correct += (outputs.argmax(1) == labels).sum().item()

test_loss /= len(test_loader.dataset)
test_acc = test_correct / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


# Save

torch.save(model.state_dict(), "D:/saved_models/mobilenetv2_finetuned.pt")
