import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import os
import csv
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Sigmoid, CrossEntropyLoss
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from datetime import datetime

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = MNIST('./MNIST/data', train=True,
                   transform=transform, download=True)
test_data = MNIST('./MNIST/data', train=False,
                  transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"Train data size: {train_data_size}")
print(f"Test data size: {test_data_size}")


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32,
                            kernel_size=3, stride=1, padding=1)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=32, out_channels=64,
                            kernel_size=3, stride=1, padding=1)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, stride=1, padding=1)
        self.maxpool3 = MaxPool2d(2)
        self.fc1 = Linear(in_features=128*3*3, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=64)
        self.fc3 = Linear(in_features=64, out_features=10)

        self.ReLU = ReLU()

    def forward(self, x):
        x = self.ReLU(self.maxpool1(self.conv1(x)))
        x = self.ReLU(self.maxpool2(self.conv2(x)))
        x = self.ReLU(self.maxpool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = MnistModel()

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer):
    # 创建结果文件夹
    results_path = "./MNIST/results/"
    os.makedirs(results_path, exist_ok=True)
    
    # 创建CSV文件记录训练结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_csv_path = os.path.join(results_path, f'training_results_{timestamp}.csv')
    
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Step', 'Loss'])
        
        for epoch in range(10):
            for index, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    print(f"Epoch {epoch}, Step {index}, Loss: {loss.item()}")
                    writer.writerow([epoch, index, loss.item()])
    
    save_path = "./MNIST/model/"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + "model.pth")
    print("Training complete")
    print(f"Training results saved to {train_csv_path}")

    if os.path.exists("./MNIST/model/model.pth"):
        model.load_state_dict(torch.load("./MNIST/model/model.pth"))


def test(model, test_loader):
    correct = 0
    total = 0
    results_path = "./MNIST/results/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_csv_path = os.path.join(results_path, f'test_results_{timestamp}.csv')
    
    with torch.no_grad(), open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Total_Images', 'Correct_Predictions', 'Accuracy'])
        
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy}%")
        writer.writerow([total, correct, accuracy])
        print(f"Test results saved to {test_csv_path}")


if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer)
    test(model, test_loader)
