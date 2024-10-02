from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn

app = Flask(__name__)

# Define your CNN architecture
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

# Load the model weights
model = CNN()
model.load_state_dict(torch.load('cat_dog_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    img = Image.open(file.stream)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    prediction = 'dog' if predicted.item() == 1 else 'cat'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

