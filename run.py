import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import csv
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Sigmoid, CrossEntropyLoss
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


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

# 创建可视化文件夹
VIS_PATH = "./MNIST/visualizations/"
os.makedirs(VIS_PATH, exist_ok=True)


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


def visualize_dataset_samples():
    """可视化数据集中的示例数字"""
    # 为每个数字找到一个示例
    examples = {}
    for images, labels in train_loader:
        for img, label in zip(images, labels):
            if label.item() not in examples and len(examples) < 10:
                examples[label.item()] = img
        if len(examples) == 10:
            break

    # 创建图表
    fig = plt.figure(figsize=(20, 8))  # 调整比例
    for i in range(10):
        ax = fig.add_subplot(2, 5, i + 1)
        img = examples[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Digit: {i}', fontsize=16, pad=10)  # 增大字体并增加边距

    plt.tight_layout(pad=3.0)  # 增加子图之间的间距
    plt.savefig(os.path.join(VIS_PATH, 'dataset_samples.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("Dataset samples visualization saved")


def train(model, train_loader, criterion, optimizer):
    # 创建结果文件夹
    results_path = "./MNIST/results/"
    os.makedirs(results_path, exist_ok=True)

    # 创建CSV文件记录训练结果
    train_csv_path = os.path.join(results_path, 'training_results.csv')

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
                    losses.append(loss.item())
                    epochs.append(epoch)
                    steps.append(index)

    # 绘制训练损失曲线
    plt.figure(figsize=(12, 8))  # 调整图像大小
    plt.plot(range(len(losses)), losses, 'b-',
             label='Training Loss', linewidth=2)  # 增加线条宽度
    plt.title('Training Loss Over Time', fontsize=16, pad=20)  # 增大标题字体
    plt.xlabel('Steps (x100)', fontsize=14)  # 增大轴标签字体
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)  # 增大图例字体
    plt.grid(True)
    plt.xticks(fontsize=12)  # 增大刻度字体
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_PATH, 'training_loss.png'),
                dpi=200, bbox_inches='tight')
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

    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 获取预测结果
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 创建图表
    fig = plt.figure(figsize=(20, 8))  # 调整大小
    for idx in range(num_images):
        ax = fig.add_subplot(2, 5, idx + 1)  # 改为2行5列
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Pred: {predicted[idx]}\nTrue: {labels[idx]}',
                     color=('green' if predicted[idx]
                            == labels[idx] else 'red'),
                     fontsize=14, pad=10)  # 增大字体

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(VIS_PATH, 'predictions.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def test(model, test_loader):
    correct = 0
    total = 0
    results_path = "./MNIST/results/"
    test_csv_path = os.path.join(results_path, 'test_results.csv')

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
    plt.figure(figsize=(12, 8))  # 调整图像大小
    class_accuracy = [100 * class_correct[i] / class_total[i]
                      for i in range(10)]
    plt.bar(range(10), class_accuracy)
    plt.title('Accuracy by Class', fontsize=16, pad=20)  # 增大标题字体
    plt.xlabel('Digit', fontsize=14)  # 增大轴标签字体
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(range(10), fontsize=12)  # 增大刻度字体
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc, f'{acc:.1f}%', ha='center',
                 va='bottom', fontsize=12)  # 增大文本字体
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_PATH, 'class_accuracy.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 可视化数据集示例
    visualize_dataset_samples()

    # 训练和测试模型
    train(model, train_loader, criterion, optimizer)
    test(model, test_loader)

    # 可视化预测结果
    visualize_predictions(model, test_loader)
