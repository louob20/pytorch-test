import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define model:
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) # 16x28x28
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # 32x14x14

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 =  nn.Linear(32*7*7, 128) # Flattened input
        self.fc2 = nn.Linear(128, 10)      # Output layer - 10 classes

    def forward(self, x):
        # Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 32*7*7)  # Flatten feature maps
        x = F.relu(self.fc1(x)) # Fully connected layer with ReLU
        x = self.fc2(x)         # Output layer - no activation since CrossEntropyLoss applies softmax
        return x

model = SimpleCNN()


# Load and preprocess data:

# Convert images to tensor & normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define loss function and optimiser:
criterion = nn.CrossEntropyLoss() # Loss for classification
optimiser = optim.Adam(model.parameters(), lr=0.001)


# Training:
def make_cnn():
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    print("Training complete")


    # Evaluation:

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # saving the model
    torch.save(model.state_dict(), "cnn.pth")
    print("Saved PyTorch Model State to cnn.pth")


# make_cnn()
model = SimpleCNN().to("cpu")
model.load_state_dict(torch.load("cnn.pth", weights_only=True))
model.eval()

import random
image, label = test_data[random.randint(0, len(test_data)-1)]  # Change index to test different images


# Preprocess and reshape for model input
image = image.unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image)
    predicted_label = torch.argmax(output).item()

# Display the image
import matplotlib.pyplot as plt
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True: {label}, Predicted: {predicted_label}")
plt.show()