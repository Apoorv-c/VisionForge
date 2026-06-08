# VisionForge
# 🧠 Image Classification using MobileNetV2 (PyTorch)

This project trains a deep learning image classification model using MobileNetV2 with PyTorch.
It supports custom datasets, performs data augmentation, and includes a training + validation pipeline with progress tracking.

The trained model is saved along with the class labels for future inference.


# 🚀 Features

Uses Transfer Learning (MobileNetV2 pretrained on ImageNet)

Supports custom datasets

Automatic GPU usage if CUDA is available

Data augmentation for better generalization

Training + validation accuracy tracking

Progress bars with tqdm

Saves model weights + class labels

# ⚙️ Requirements

Install the required libraries:

pip install torch torchvision tqdm

Optional (recommended for faster loading):

pip install accelerate


# 🧩 Configuration

Before running the script, update these paths in the code:

TRAIN_DIR = "path_to_training_data"
VAL_DIR = "path_to_validation_data"
MODEL_PATH = "model.pth"

Other parameters:

BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001


# 🖼 Data Augmentation

The following augmentations are applied to training images:

Resize to 224x224

Random horizontal flip

Random rotation

Color jitter (brightness, contrast, saturation)

This improves model generalization.


#🏋️ Training the Model

Run the training script:

python train_model.py

During training you will see progress like:

Epoch 1/20 [Train]: 100%|██████████|
Train Loss: 0.5432 | Train Acc: 0.8123
Val Loss: 0.4312 | Val Acc: 0.8541


# 💾 Model Saving

After training completes, the model is saved as:

model.pth

It contains:

Model weights

Class names

Example saved format:

{
 "model_state_dict": ...,
 "class_names": [...]
}


# 🖥 Device Support

The script automatically selects:

GPU (CUDA) if available

CPU otherwise

Using device: cuda

or

Using device: cpu


# 📊 Model Architecture

The project uses:

MobileNetV2

Advantages:

Lightweight

Fast training

Good performance on small datasets

The final classifier layer is modified to match the number of dataset classes.

# 🧠 Transfer Learning

The model loads pretrained weights:

ImageNet pretrained weights

Then only the final layer is adapted for the custom dataset.

This significantly improves performance on smaller datasets.

