import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import denv3
from denv3 import DEN_V3  # 确保您已经正确导入 DEN_V3 模型

# 解析命令行参数
parser = argparse.ArgumentParser(description='PyTorch DEN_V3 Training on ImageNet1K')
parser.add_argument('--data_dir', default='/home/bobsun/bob/data/imagenet1000', type=str, help='数据集路径')
parser.add_argument('--batch_size', default=468, type=int, help='批大小')
parser.add_argument('--epochs', default=10, type=int, help='训练的总轮数')
parser.add_argument('--lr', default=0.0002, type=float, help='初始学习率')
parser.add_argument('--min_lr', default=0.00002, type=float, help='学习率调度器的最小学习率')
parser.add_argument('--weight_decay', default=0.01, type=float, help='权重衰减系数')
parser.add_argument('--print_freq', default=200, type=int, help='信息打印频率')
parser.add_argument('--num_workers', default=8, type=int, help='数据加载线程数')
parser.add_argument('--gpu', default=None, type=int, help='使用的GPU ID')
parser.add_argument('--resume', default='', type=str, help='恢复训练的检查点路径')
parser.add_argument('--fine_tune_lr', default=1e-4, type=float, help='微调时使用的学习率')
args = parser.parse_args()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

from torchvision.transforms import AutoAugment, AutoAugmentPolicy
# 数据预处理
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
}

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                      pin_memory=True)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
num_classes = len(image_datasets['train'].classes)

model = denv3.DEN_V3()
model.load_state_dict(torch.load('checkpoints/checkpoint_epoch_1.pth', map_location=device)["state_dict"])

model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 使用微调学习率重新定义优化器
optimizer = optim.AdamW(model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

# 定义学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

# 可选地从检查点恢复
best_acc = 0.0
start_epoch = 0

# 训练函数
scaler = torch.cuda.amp.GradScaler()


def train(epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 自动混合精度前向传播
        with torch.cuda.amp.autocast():
            outputs = model(inputs)

            try:
                loss = 0
                if (type(outputs) is list):
                    for output in outputs:
                        loss += criterion(output, labels)
                else:
                    loss = criterion(outputs, labels)
            except Exception as e:
                print(e)
                continue
            # 计算预测
            _, preds = torch.max(outputs[-1], 1)

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if i % args.print_freq == 0:
            print(f'Epoch [{epoch}][{i}/{len(dataloaders["train"])}]\t Loss {loss.item():.4f}')

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


# 验证函数
def validate(epoch):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            weighted_output = 0.0  # 初始化加权输出

            if isinstance(outputs, list):
                count = 0
                for output in outputs:
                    count += 1
                    weight = 0.5 ** (len(outputs) - count)  # 加权因子
                    weighted_output += output * weight       # 加权求和
            else:
                weighted_output = outputs  # 如果不是列表，直接使用输出

            # 使用加权后的输出计算损失和准确率
            loss = criterion(weighted_output, labels)
            _, preds = torch.max(weighted_output, 1)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc



# 主训练循环（微调）
best_acc = 0.0  # 重置最佳精度
for epoch in range(start_epoch, args.epochs):
    print(f'=====')
    print(f'Epoch {epoch}/{args.epochs - 1}')
    print('-' * 10)

    train(epoch)
    val_acc = validate(epoch)

    scheduler.step()

    # 保存最佳模型
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)

    # 检查和创建保存目录
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # 保存当前模型
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, save_path)

    if is_best:
        best_model_path = os.path.join(save_dir, 'best_model_fine_tune_for_explain.pth')
        torch.save(model,best_model_path)
        print("已保存微调后的最佳模型")

print(f'微调完成。最佳验证精度: {best_acc:.4f}')
