import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import gdown

# Define the URL of the model
MODEL_URL = 'https://drive.google.com/uc?id=1BJf2SBr9383z-WXXLkc1R2_tmXSSjJhA'
MODEL_PATH = 'cat_dog_classifier.pth'

# Download the model if it does not exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Define the CNN model (same architecture as used during training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model object
model = CNN()

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# Set the model to evaluation mode (important for inference)
model.eval()

# Define the image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the size expected by the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
])

# Function to predict the class of a new image
def predict_image(image_path):
    try:
        # Open the image, apply the transformations
        img = Image.open(image_path)
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Make the prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get the index of the highest probability

        # Convert predicted index to class (0 for dog, 1 for cat)
        return "It's a dog!" if predicted.item() == 1 else "It's a cat!"
    except Exception as e:
        return f"Error predicting image: {e}"

# Example usage
if __name__ == "__main__":
    prediction = predict_image("m.jpeg")
    print(prediction)
