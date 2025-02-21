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
import numpy as np

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
    
    # 用于记录损失值
    losses = []
    epochs = []
    steps = []
    
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
                    # 记录损失值
                    losses.append(loss.item())
                    epochs.append(epoch)
                    steps.append(index)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, 'b-', label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Steps (x100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, f'training_loss_{timestamp}.png'))
    plt.close()
    
    save_path = "./MNIST/model/"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + "model.pth")
    print("Training complete")
    print(f"Training results saved to {train_csv_path}")

    if os.path.exists("./MNIST/model/model.pth"):
        model.load_state_dict(torch.load("./MNIST/model/model.pth"))


def visualize_predictions(model, test_loader, num_images=10):
    """可视化模型预测结果"""
    model.eval()
    results_path = "./MNIST/results/"
    os.makedirs(results_path, exist_ok=True)
    
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 创建图表
    fig = plt.figure(figsize=(15, 3))
    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx + 1)
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Pred: {predicted[idx]}\nTrue: {labels[idx]}',
                    color=('green' if predicted[idx] == labels[idx] else 'red'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_path, f'predictions_{timestamp}.png'))
    plt.close()


def test(model, test_loader):
    correct = 0
    total = 0
    results_path = "./MNIST/results/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_csv_path = os.path.join(results_path, f'test_results_{timestamp}.csv')
    
    # 用于记录每个类别的准确率
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad(), open(test_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Total_Images', 'Correct_Predictions', 'Accuracy'])
        
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 统计每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        
        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy}%")
        writer.writerow([total, correct, accuracy])
        print(f"Test results saved to {test_csv_path}")
    
    # 绘制每个类别的准确率柱状图
    plt.figure(figsize=(10, 6))
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
    plt.bar(range(10), class_accuracy)
    plt.title('Accuracy by Class')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(10))
    plt.ylim(0, 100)
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc, f'{acc:.1f}%', ha='center', va='bottom')
    plt.savefig(os.path.join(results_path, f'class_accuracy_{timestamp}.png'))
    plt.close()


if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer)
    test(model, test_loader)
    # 可视化一些预测结果
    visualize_predictions(model, test_loader)
