import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================
#  CONFIG
# =========================
TRAIN_DIR = r"ADD YOUR TRAINING DATA PATH HERE"
VAL_DIR = r"ADD YOUR VALIDATION DATA PATH HERE"
MODEL_PATH = "MODEL PATH HERE.pth"
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================
#  DATA TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"✅ Loaded {len(train_dataset.classes)} classes.")
print(f"Classes: {train_dataset.classes}")

# =========================
#  MODEL + LOSS + OPTIMIZER
# =========================
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
#  TRAINING + VALIDATION LOOP
# =========================
for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    train_loss, train_corrects = 0.0, 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_train_loss = train_loss / len(train_dataset)
    epoch_train_acc = train_corrects.double() / len(train_dataset)

    # --- VALIDATION ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)

    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} "
          f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

# =========================
#  SAVE MODEL + CLASSES
# =========================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": train_dataset.classes
}, MODEL_PATH)

print(f"🎯 Training complete! Model saved at {MODEL_PATH}")
