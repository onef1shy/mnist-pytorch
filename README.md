# MNIST手写数字识别项目

这是一个使用PyTorch实现的MNIST手写数字识别项目。该项目使用卷积神经网络(CNN)来识别手写数字，实现了较高的识别准确率。

## 项目结构

```
MNIST/
├── data/           # 存放MNIST数据集
├── model/          # 存放训练好的模型
├── results/        # 存放训练和测试结果
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

## 结果保存

- 训练结果保存在 `results/training_results_[timestamp].csv`
- 测试结果保存在 `results/test_results_[timestamp].csv`
- 模型保存在 `model/model.pth` 