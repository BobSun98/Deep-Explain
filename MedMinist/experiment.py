import torch
from torch.utils.data import DataLoader
from medmnist import ChestMNIST, Evaluator
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from resnet import ResNet18
from torchvision.models import resnet18
# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = ChestMNIST(split="train", download=True, transform=transform,size=224, root="/home/bobsun/cambrige/MedMinist/data")
val_dataset = ChestMNIST(split="val", download=True, transform=transform,size=224, root="/home/bobsun/cambrige/MedMinist/data")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
num_classes = 14  # ChestMNIST 有 14 个类别
# model = resnet18(num_classes)
model = ResNet18(num_classes)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# 定义损失函数和优化器
# 假设已计算每个类别的正样本数量

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练函数
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

# 验证函数
def validate(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)  # 使用 sigmoid 输出概率


            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_preds).numpy()

    # 初始化 Evaluator 并计算指标
    evaluator = Evaluator("chestmnist", split='test')
    metrics = evaluator.evaluate(y_score, y_true)

    print("Validation AUC: %.3f, Accuracy: %.3f" % (metrics[0], metrics[1]))
    return metrics
# 训练与验证循环
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
    validate(model, val_loader, device)