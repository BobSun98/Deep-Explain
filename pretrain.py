import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from resnet import ResNet18

# 解析命令行参数
parser = argparse.ArgumentParser(description='PyTorch ResNet-18 Training on ImageNet1K')
parser.add_argument('--data_dir', default='/home/bobsun/bob/data/imagenet1000', type=str, help='数据集路径')
parser.add_argument('--batch_size', default=256, type=int, help='批大小')
parser.add_argument('--epochs', default=90, type=int, help='训练的总轮数')
parser.add_argument('--lr', default=0.001, type=float, help='初始学习率')
parser.add_argument('--min_lr', default=0.0, type=float, help='学习率调度器的最小学习率')
parser.add_argument('--weight_decay', default=0.01, type=float, help='权重衰减系数')
parser.add_argument('--print_freq', default=100, type=int, help='信息打印频率')
parser.add_argument('--num_workers', default=8, type=int, help='数据加载线程数')
parser.add_argument('--gpu', default=None, type=int, help='使用的GPU ID')
parser.add_argument('--resume', default='', type=str, help='恢复训练的检查点路径')
args = parser.parse_args()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')

# 数据预处理
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]),
}

# 加载数据集
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
num_classes = len(image_datasets['train'].classes)

# 定义模型
# 如果您有自定义的 ResNet-18 模型，请导入并实例化
# from your_model_file import ResNet18
# model = ResNet18(num_classes=num_classes)

# 这里使用 torchvision 提供的 ResNet-18 模型
model = ResNet18(num_classes=num_classes,inchannels=3)
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

# 定义学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

# 可选地从检查点恢复
best_acc = 0.0
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
        print(f"=> 加载检查点 '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> 加载完成 '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> 未找到检查点 '{args.resume}'")

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        _, preds = torch.max(outputs, 1)
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
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc

# # 主训练循环
# for epoch in range(start_epoch, args.epochs):
#     print(f'=====')
#     print(f'Epoch {epoch}/{args.epochs - 1}')
#     print('-' * 10)
#
#     train(epoch)
#     val_acc = validate(epoch)
#
#     scheduler.step()
#
#     # 保存检查点
#     is_best = val_acc > best_acc
#     best_acc = max(val_acc, best_acc)
#
#     checkpoint = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_acc': best_acc,
#         'optimizer': optimizer.state_dict(),
#         'scheduler': scheduler.state_dict(),
#     }
#
#     save_dir = 'checkpoints'
#     # if not os.path.exists(save_dir):
#         # os.makedirs(save_dir)
#     # save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
#     # torch.save(checkpoint, save_path)
#
#     if is_best:
#         best_model_path = os.path.join(save_dir, 'best_model.pth')
#         torch.save(model.state_dict(), best_model_path)
#         print("saved best model")
#
# print(f'训练完成。最佳验证精度: {best_acc:.4f}')

# 添加微调学习率的命令行参数
# 添加微调学习率的命令行参数
parser.add_argument('--fine_tune_lr', default=1e-5, type=float, help='微调时使用的学习率')

# 解析所有参数
args = parser.parse_args()

# 加载最佳模型权重
if os.path.isfile('checkpoints/best_model.pth'):
    print("=> 加载最佳模型 'best_model.pth' 进行微调")
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    print("=> 加载完成 'best_model.pth'")
else:
    print("=> 未找到 'best_model.pth'，请检查路径或文件名是否正确")
    exit()

# 使用微调学习率重新定义优化器
optimizer = optim.AdamW(model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

# 主训练循环（微调）
best_acc = 0.0  # 重置最佳精度
for epoch in range(10):
    print(f'=====')
    print(f'Epoch {epoch}/{10 - 1}')
    print('-' * 10)

    train(epoch)
    val_acc = validate(epoch)

    # 保存最佳模型
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)

    # 检查和创建保存目录
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    if is_best:
        best_model_path = os.path.join(save_dir, 'fine_tuned_best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print("已保存微调后的最佳模型")

print(f'微调完成。最佳验证精度: {best_acc:.4f}')