import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ----- Hyperparameters -----
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# ----- Device configuration -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Data loading and preprocessing -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----- Neural Network model -----
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 28x28 image to 784
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# ----- Training -----
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


# ----- Evaluation -----
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


# ----- Main -----
if __name__ == "__main__":
    model = NeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {device}\n")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Test Loss':<14} {'Test Acc'}")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(
            f"{epoch:<8} {train_loss:<14.4f} {train_acc:<14.2f} {test_loss:<14.4f} {test_acc:.2f}"
        )

    # Save the trained model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("\nModel saved to mnist_model.pth")
