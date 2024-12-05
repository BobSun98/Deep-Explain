import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepExplanation(nn.Module):
    def __init__(self, channels, num_classes):
        super(DeepExplanation, self).__init__()
        self.pred_fc = nn.Linear(channels, num_classes,bias=False)
        self.W = nn.Linear(channels, channels,bias=False)
        # self.temperature = nn.Parameter(torch.tensor(1.0))
        self.activation = None
        self.activation_all = None
        self.deep_pred = None

    def forward(self, x, x_next):
        # 输入:上一层输出和下一层输出
        # 输出,连接到下一层的feature map以及深度监督


        x_squeeze = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x_next_squeeze = F.adaptive_avg_pool2d(x_next, 1).squeeze(-1).squeeze(-1)

        x_squeeze = x_squeeze + x_next_squeeze
        x_squeeze = F.relu(x_squeeze)

        x_squeeze = self.W(x_squeeze)

        attention_gate = torch.sigmoid(x_squeeze)
        attention_gate = attention_gate.unsqueeze(-1).unsqueeze(-1)
        x = x * attention_gate
        self.activation = x
        self.activation_all = x + x_next
        # x_squeeze = F.relu(x_squeeze)
        deep_pred = self.pred_fc(x_squeeze)
        self.deep_pred = deep_pred
        # temperature = torch.clamp(self.temperature, min=0.1)

        return x, deep_pred


class BasicBlock(nn.Module):
    expansion = 1  # 用于调整最后一个线性层的输入通道数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, num_classes=11):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 如果需要调整输入尺寸，就用downsample层
        self.deep_explain = DeepExplanation(out_channels, num_classes)

    def forward(self, input):
        pred_ = 0.0
        if type(input) == tuple:
            x = input[0]
            pred_ = input[1]
        else:
            x = input

        identity = x  # 保留输入以用于残差连接

        # 卷积 -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 卷积 -> BN
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要进行下采样或通道调整
        if self.downsample is not None:
            identity = self.downsample(x)
        identity, pred = self.deep_explain(identity, out)
        out += identity  # 加上输入
        out = F.relu(out)  # ReLU激活

        return (out, pred + pred_ * 0.8)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, inchannels=1):
        super(ResNet18, self).__init__()
        # 输入卷积层
        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 56
        self.num_classes = num_classes
        # 各个阶段的残差层
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)  # 56
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)  # 28
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # 14
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=1)  # 14

        # # 全局平均池化和全连接层
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes,bias=False)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # 如果输入和输出通道数或尺寸不匹配，需要进行下采样
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        # 第一个残差块需要可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, num_classes=self.num_classes))
        # 后续的残差块不改变通道数和尺寸
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, num_classes=self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积、批归一化和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # 各阶段的残差块

        x, pred = self.layer1((x, 0))
        x, pred = self.layer2((x, pred))
        x, pred = self.layer3((x, pred))
        x, pred = self.layer4((x, pred))

        # 全局平均池化和全连接
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(x)
        # x = self.fc(x)

        return pred


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


import torch
import torch.nn as nn

class NoInplaceBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(NoInplaceBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)  # 不使用 inplace
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()  # 防止原地操作

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  # 不使用 inplace

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # 避免使用 +=
        out = self.relu(out)  # 不使用 inplace

        return out

class NoInplaceResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(NoInplaceResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)  # 不使用 inplace
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 不使用 inplace
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def no_inplace_resnet18(num_classes=1000):
    return NoInplaceResNet(NoInplaceBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

if __name__ == '__main__':
    from torchsummary import summary
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DeepExplanation(in_channels=32, out_channels=64, num_classes=10)
    net = ResNet18(num_classes=9, inchannels=3)
    print(net)
    count_parameters(net)
    model = torchvision.models.resnet18(num_classes=9)
    count_parameters(model)
