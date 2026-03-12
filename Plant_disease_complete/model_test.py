from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(r"A:\Desktop\Plant_disease\plant_disease_model.pth", map_location=DEVICE)

# Rebuild model architecture
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(model.last_channel, len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

class_names = checkpoint["class_names"]
print("✅ Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_path = r"ADD YOUR IMAGE PATH HERE"

# Convert to RGB to remove alpha channel
img = Image.open(img_path).convert("RGB")

img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    _, pred = torch.max(outputs, 1)
    print(f"Predicted class: {class_names[pred.item()]}")
