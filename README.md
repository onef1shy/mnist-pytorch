# MNIST手写数字识别项目

这是一个使用PyTorch实现的MNIST手写数字识别项目。该项目使用卷积神经网络(CNN)来识别手写数字，实现了较高的识别准确率。

## 项目结构

```
MNIST/
├── data/           # 存放MNIST数据集
├── model/          # 存放训练好的模型
├── results/        # 存放训练和测试结果数据
├── visualizations/ # 存放所有可视化图像
├── run.py          # 主程序文件
└── README.md       # 项目说明文件
```

## 功能特点

- 使用PyTorch框架实现
- 采用CNN网络结构
- 包含三层卷积层和三层全连接层
- 使用Adam优化器
- 自动保存训练过程和测试结果到CSV文件
- 模型训练完成后自动保存
- 丰富的可视化功能

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- pillow

## 如何使用

1. 安装依赖：
```bash
pip install torch torchvision matplotlib pillow
```

2. 运行训练和测试：
```bash
python run.py
```

## 模型结构

- 输入层：28x28 灰度图像
- 卷积层1：32个3x3卷积核
- 卷积层2：64个3x3卷积核
- 卷积层3：128个3x3卷积核
- 全连接层1：128个神经元
- 全连接层2：64个神经元
- 输出层：10个神经元（对应0-9十个数字）

## 可视化功能

项目包含多种可视化功能，所有可视化结果保存在 `visualizations` 目录下：

1. 数据集示例（`dataset_samples.png`）
   - 展示数据集中的0-9数字示例
   - 帮助理解输入数据的形式

2. 训练过程（`training_loss_[timestamp].png`）
   - 显示训练过程中损失值的变化
   - 帮助监控模型训练的收敛情况

3. 预测结果（`predictions_[timestamp].png`）
   - 展示模型对测试集图像的预测结果
   - 正确预测显示为绿色，错误预测显示为红色

4. 分类准确率（`class_accuracy_[timestamp].png`）
   - 展示模型对每个数字（0-9）的识别准确率
   - 包含具体的准确率数值

## 结果保存

- 训练结果保存在 `results/training_results_[timestamp].csv`
- 测试结果保存在 `results/test_results_[timestamp].csv`
- 模型保存在 `model/model.pth` 