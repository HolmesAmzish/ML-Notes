"""
Convolutional neural network model for prection of fashion-MNIST
Date: 2025-02-07
Author: Holmes Amzish
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层：输入1通道，输出32通道，卷积核大小3x3，padding=1 保证输出大小不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 最大池化层：2x2池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层：将卷积后的特征展平，连接至10个类别
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 输入大小根据卷积层计算得出
        self.fc2 = nn.Linear(512, 10)

        # Dropout 层：防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积 + 激活函数 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        # 展平多维输入成一维
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层 + 激活函数 + Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # 输出层
        x = self.fc2(x)
        return x


# 预处理和加载数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 创建模型实例
model = CNN()

# 使用GPU或者CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


if not os.path.exists('fashion_mnist_model_2.pth'):
    # 训练模型
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(X)

            loss = loss_fn(outputs, y)
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
    torch.save(model.state_dict(), 'fashion_mnist_model_2.pth')
    print("fashion_mnist_model_2 saved.")

model.load_state_dict(torch.load('fashion_mnist_model_2.pth'))

# 测试模型
model.eval()  # 设置为评估模式
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 显示测试结果中的一张图片和预测
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.unsqueeze(0).to(device)  # 为批处理添加一个额外的维度
    pred = model(x)
    predicted_class = pred.argmax(1).item()

plt.imshow(x.squeeze().cpu(), cmap="gray")  # 转到CPU以显示图像
plt.title(f"Predicted: {classes[predicted_class]}, Actual: {classes[y]}")
plt.show()
