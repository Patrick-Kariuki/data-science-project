import json
import torch
import torch.nn as nn
import numpy as np
file_path = "data&labels.json"  
with open(file_path, "r") as f:
    data = json.load(f)


# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size):
  # Load image using OpenCV or Pillow (replace with your preferred library)
  img = cv2.imread(image_path)  # Example using OpenCV
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

  # Resize the image to the target size
  img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_AREA)

  # Convert to PyTorch tensor and normalize (adjust as needed)
  img = img.astype(np.float32) / 255.0
  img = torch.from_numpy(img).permute(2, 0, 1)  # Convert to CHW format

  # Normalize (adjust mean and std as needed)
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  img = (img - torch.tensor(mean)) / torch.tensor(std)

  return img.unsqueeze(0)  # Add batch dimension

# Function to build the FCN model (replace with a more complex FCN if needed)
class FCN(nn.Module):
  def __init__(self, input_shape, num_classes):
    super(FCN, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(16, num_classes, kernel_size=1)
    )

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# Example usage
data_dict = data  # Replace with your data
target_size = (224, 224)  # Adjust image size as needed
num_classes = len(set(data_dict.values()))  # Assuming unique class labels

# Prepare training data (lists or PyTorch tensors)
X_train = []
y_train = []
for image_path, label in data_dict.items():
  # One-hot encode the label (adjust for your label format)
  one_hot_label = torch.nn.functional.one_hot(torch.tensor([int(data_dict[image_path])]), num_classes)
  X_train.append(load_and_preprocess_image(image_path, target_size))
  y_train.append(one_hot_label)

# Convert to PyTorch tensors
X_train = torch.cat(X_train)
y_train = torch.cat(y_train)

# Define device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and optimizer
model = FCN(target_size + (3,), num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Loss function (adjust for multi-class classification)
criterion = nn.CrossEntropyLoss()