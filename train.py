import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG19输入要求224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用预训练VGG19的均值和标准差
])

# 数据集加载（假设数据集已经按类别划分）
dataset = datasets.ImageFolder('setData/', transform=transform)

# 使用train_test_split划分训练集与验证集
train_data, val_data = train_test_split(dataset, test_size=0.2, stratify=dataset.targets)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 加载预训练的VGG19模型
model = torchvision.models.vgg19(pretrained=True)

# 替换最后的全连接层
model.classifier[6] = nn.Linear(4096, 2)  # 二分类，输出为2个类别

# 将模型移动到设备
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 训练函数
def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 清除梯度

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {running_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')


# 训练与验证
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_loader, criterion, optimizer, device, epochs=1)
    validate(model, val_loader, criterion, device)

# 保存模型
torch.save(model.state_dict(), 'vgg19_model.pth')
print("Model saved as 'vgg19_model.pth'")
