from torchcam.utils import overlay_mask
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from medmnist import INFO

from PIL import Image, ImageDraw, ImageFont


def add_text_to_image(image, text, position=(10, 10), font_size=40, max_width=None, line_spacing=5,add_text=True):
    """
    在图像上添加支持自动换行的文本。

    参数:
    - image: PIL 图像对象
    - text: 要添加的文本
    - position: 文本在图像上的起始位置
    - font_size: 字体大小
    - max_width: 文本行的最大宽度，如果不指定，则使用图像宽度
    - line_spacing: 行间距

    返回:
    - 带有文本的图像
    """

    if not add_text:
        return image
    draw = ImageDraw.Draw(image)
    try:
        # 尝试加载系统默认字体
        font = ImageFont.truetype("arial", font_size)
    except IOError:
        # 使用默认字体
        font = ImageFont.load_default()

    # 如果没有指定最大宽度，则使用图像宽度
    if max_width is None:
        max_width = image.width - position[0] * 2  # 留出一定边距

    # 拆分文本，逐行添加
    lines = []
    words = text.split()
    current_line = ""
    for word in words:
        # 判断当前行加入下一个词后是否超过最大宽度
        test_line = f"{current_line} {word}".strip()
        left, top, right, bottom = font.getbbox(test_line)
        line_width = right - left
        if line_width <= max_width:
            current_line = test_line
        else:
            # 当前行宽度超限，将当前行加入到行列表并重置
            lines.append(current_line)
            current_line = word

    # 添加最后一行
    if current_line:
        lines.append(current_line)

    # 绘制每一行文本
    x, y = position
    for line in lines:
        draw.text((x, y), line, fill="white", font=font)
        _, _, _, line_height = font.getbbox(line)  # 获取当前行的高度
        y += line_height + line_spacing  # 增加行间距

    return image


def get_weight_with_sharpness(full_weight, y, sharpness, sharpness_n):
    # 确保 full_weight 是 PyTorch 张量
    full_weight_pt = full_weight
    weight_pt = full_weight_pt[y]

    # 转换为 NumPy 数组
    full_weight_np = full_weight_pt
    weight_np = weight_pt

    # 处理负向权重
    full_weight_n = -full_weight_np.copy()
    weight_n = -weight_np.copy()

    # 正向权重处理
    full_weight_np[full_weight_np < 0] = 0
    norm = np.sum(full_weight_np, axis=0)
    weight_np[weight_np < 0] = 0
    alpha = weight_np / (norm + 1e-8)
    weight_np[alpha < sharpness] = 0

    # 负向权重处理
    full_weight_n[full_weight_n < 0] = 0
    norm_n = np.sum(full_weight_n, axis=0)
    weight_n[weight_n < 0] = 0
    alpha_n = weight_n / (norm_n + 1e-8)
    weight_n[alpha_n < sharpness_n] = 0

    # 最终权重
    final_weight = weight_np - weight_n

    return final_weight


def generate_cam_with_overlay(fc_weights, feature_map, class_idx, input_image, non_negative=1.0, negative=0,
                              sharpness=0, sharpness_n=0):
    # print("class_idx:",class_idx)
    weights = fc_weights[class_idx].cpu().detach().numpy()
    if (sharpness > 0 or sharpness_n > 0):
        weights = get_weight_with_sharpness(fc_weights.cpu().detach().numpy(), class_idx, sharpness, sharpness_n)
    B, C, H, W = feature_map.shape
    feature_map = feature_map.cpu().detach().numpy()

    # 计算正向 CAM
    weights_positive = np.clip(weights, a_min=0, a_max=None)
    weights_negative = np.clip(-weights, a_min=0, a_max=None)
    weights_final = non_negative * weights_positive + negative * weights_negative
    cam_positive = np.sum(weights_final[:, None, None] * feature_map[0], axis=0)
    cam_return = cam_positive.copy()

    # 归一化 CAM
    min_val = np.percentile(cam_positive, 1)
    max_val = np.percentile(cam_positive, 99)
    cam_positive = (cam_positive - min_val) / (max_val - min_val)
    cam_positive[cam_positive < 0.1] = 0
    cam_positive[cam_positive > 1.0] = 1.0
    cam_positive_resized = cv2.resize(cam_positive, (input_image.shape[1], input_image.shape[0]))
    cam_return = cv2.resize(cam_return, (input_image.shape[1], input_image.shape[0]))

    # 将激活图转换为 PIL Image 格式
    activation_map = Image.fromarray(cam_positive_resized, mode='F')

    # 使用 overlay_mask 叠加
    overlay_image = overlay_mask(Image.fromarray(input_image), activation_map, alpha=0.5)

    return overlay_image, cam_return


def draw(img_list, input_image=None, cam_list=None, class_names=[], label=0, predicted_class="", confidence=0.0):
    # 累加所有 CAM，并进行归一化处理

    if input_image is not None:
        all_cam = np.sum(cam_list, axis=0)
        all_cam = (all_cam - all_cam.min()) / (all_cam.max() - all_cam.min())

        # 将叠加的 CAM 转换为 PIL Image 格式
        all_cam_img = Image.fromarray(all_cam, mode='F')
        overlay_all_cam = overlay_mask(Image.fromarray(input_image), all_cam_img, alpha=0.5)

        # 在原图上添加实际类别
        original_image_with_text = add_text_to_image(Image.fromarray(input_image), f"True Class: {class_names[label]}")
        # 在全叠加的 CAM 图上添加预测类别和置信度
        overlay_all_cam_with_text = add_text_to_image(overlay_all_cam, f"P: {predicted_class}, C: {confidence:.2f}")

        # 创建包含原图和叠加激活图的图像
        img_list.insert(0, [original_image_with_text, overlay_all_cam_with_text])
    layer_images = []

    # 拼接各层的激活图
    for i, layer in enumerate(img_list):
        widths, heights = zip(*(img.size for img in layer))
        total_width = sum(widths)
        max_height = max(heights)

        # 创建一个空白图像来存放水平拼接结果
        layer_image = Image.new('RGB', (total_width, max_height))

        # 逐个粘贴图像并添加对应层级的预测信息
        x_offset = 0
        for j, img in enumerate(layer):
            # 如果是每层的 CAM 图，添加层的预测文本
            if i > 0:  # 跳过原图和 all_cam 图
                text = f"layer{i}[{j}] Pred: {class_names[j]}, Conf: {confidence:.2f}"
                # img = add_text_to_image(img, text)
            layer_image.paste(img, (x_offset, 0))
            x_offset += img.width

        layer_images.append(layer_image)

    # 垂直拼接所有层
    widths, heights = zip(*(img.size for img in layer_images))
    max_width = max(widths)
    total_height = sum(heights)

    final_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for layer_image in layer_images:
        final_image.paste(layer_image, (0, y_offset))
        y_offset += layer_image.height

    return final_image


def concatImg(model, test_loader, idx, non_negative=1, negative=0, class_=-1, sharpness=0, sharpness_n=0,
              dataset_name="dermamnist", activation="activation",add_text=True):
    images, labels = next(iter(test_loader))
    img_tensor = images[idx].to(device).unsqueeze(0)
    label = labels[idx].item()
    class_names = [label for _, label in INFO[dataset_name]["label"].items()]
    print(f"True Class: {class_names[label]}")

    # label = 0
    # 获取模型输出
    output = model(img_tensor)
    # print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
    original_image = img_tensor.cpu().squeeze().numpy()
    if original_image.ndim == 2:  # 单通道 (H, W)
        display_image = (original_image * 0.5 + 0.5) * 255  # 转换到 [0, 255]
        input_image = np.stack([display_image] * 3, axis=-1)  # 转为三通道
    elif original_image.ndim == 3:  # 多通道 (C, H, W)
        input_image = (original_image.transpose(1, 2, 0) * 0.5 + 0.5) * 255

    # input_image = img_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    # input_image = (input_image * 0.5 + 0.5) * 255
    # input_image = input_image.astype('uint8')

    img_list = []
    cam_list = []
    for i in range(1, 5):
        layer_list = []
        for j in range(2):
            pred_fc = getattr(getattr(getattr(model, f'layer{i}')[j], 'deep_explain'), 'pred_fc').weight
            W = getattr(getattr(getattr(model, f'layer{i}')[j], 'deep_explain'), 'W').weight
            weight = pred_fc @ W
            feature_map = getattr(getattr(getattr(model, f'layer{i}')[j], 'deep_explain'), activation)
            logit = getattr(getattr(getattr(model, f'layer{i}')[j], 'deep_explain'), 'deep_pred')
            probabilities = F.softmax(logit, dim=1).cpu().detach().numpy()[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            # print(f"layer{i}[{j}] Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
            if not class_ == -1:
                label = class_
            overlay_image, activation_map = generate_cam_with_overlay(weight, feature_map, label, input_image,
                                                              non_negative, negative, sharpness, sharpness_n)
            overlay_image_with_text = add_text_to_image(overlay_image,
                                                        f"layer{i}[{j}] Pred: {predicted_class}, Conf: {confidence:.2f}")
            layer_list.append(overlay_image_with_text)
            cam_list.append(activation_map) # * (0.8 ** (8 - len(cam_list))))

        img_list.append(layer_list)

    probabilities = F.softmax(output, dim=1).cpu().detach().numpy()[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]
    all_img = draw(img_list, input_image, cam_list, class_names, label, predicted_class, confidence)
    return all_img


import torch
from torchcam.methods import GradCAM

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_grad_cam_image(model, test_loader, idx, target_layer='layer4', alpha=0.5, normalize_input=True):
    """
    使用 Grad-CAM 生成指定图像的热力图并叠加到原图上。

    参数:
    - model: 经过训练的模型
    - test_loader: 数据加载器，用于提供输入图像
    - idx: 图像在 batch 中的索引
    - target_layer: 目标层的名称，用于生成 Grad-CAM
    - alpha: 叠加热力图的透明度
    - normalize_input: 是否归一化输入图像（默认为 True）

    返回:
    - result: 叠加了 Grad-CAM 热力图的 PIL 图像
    - predicted_class_name: 预测的类别名称
    - true_class_name: 实际的类别名称
    """

    # 将模型设置为评估模式并加载到设备
    # model.eval()
    model.to(device)

    # 从 test_loader 中获取一批图像
    images, labels = next(iter(test_loader))

    # 获取指定索引的图像并转移到设备
    img_tensor = images[idx].to(device)
    # 如果是单通道图像，转换为三通道
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)
    # 添加批次维度
    input_tensor = img_tensor.unsqueeze(0)

    # 初始化 GradCAM 对象并指定目标层
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # 前向传播以记录激活
    output = model(input_tensor)

    # 获取预测的类别索引和名称
    predicted_class_idx = output.argmax(dim=1).item()
    class_names = [label for _, label in INFO["pathmnist"]["label"].items()]
    true_class_name = class_names[labels[idx].item()]
    predicted_class_name = class_names[predicted_class_idx]

    # 生成指定类别的 CAM 热力图
    activation_map = cam_extractor(predicted_class_idx, output)

    # 将输入图像转换为 PIL 格式
    if normalize_input:
        input_image = to_pil_image(img_tensor.cpu() * 0.5 + 0.5)  # 假设图像已经归一化到 [-1, 1]
    else:
        input_image = to_pil_image(img_tensor.cpu())

    # 可视化 Grad-CAM 结果
    activation_map = activation_map[0].squeeze().cpu().detach().numpy()
    result = overlay_mask(input_image, Image.fromarray(activation_map, mode='F'), alpha=alpha)

    # 打印实际和预测类别
    # print(f"True Class: {true_class_name}")
    # print(f"P: {predicted_class_name}, C: {output.softmax(dim=1)[0][predicted_class_idx].item():.2f}")

    return result, predicted_class_name, true_class_name


# 调用示例
# result_image, predicted_class_name, true_class_name = generate_grad_cam_image(model, test_loader, idx)
# result_image.show()  # 或者保存结果 result_image.save('grad_cam_result.jpg')

import shap
import torch
import numpy as np
from medmnist.dataset import INFO
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_shap_explanation(model, test_loader, idx, max_evals=5000, batch_size=50, masker_type="inpaint_telea",
                              class_name="dermamnist"):
    # 将模型加载到设备并设置为评估模式
    model.to(device)
    model.eval()

    # 定义模型预测函数
    def f(x):
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float()
        x = x.to(device)

        # 归一化（根据实际情况调整）
        # mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        # std = torch.tensor([0.5, 0.5, 0.5]).to(device)
        # x = (x - mean[:, None, None]) / std[:, None, None]

        # 这里不要使用 torch.no_grad()，让梯度计算保持开启
        outputs = model(x)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs.detach().cpu().numpy()  # 添加 .detach() 去除梯度跟踪

    # 从 test_loader 中获取一批图像
    batch = next(iter(test_loader))
    images, labels = batch
    X = images.permute(0, 2, 3, 1).cpu().numpy()

    # 选择 masker 类型
    masker = shap.maskers.Image(masker_type, X[0].shape)

    # 提取 PathMNIST 类别名称
    class_names = [label for _, label in INFO[class_name]["label"].items()]
    class_name = class_names[labels[idx].item()]

    # 创建解释器
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # 计算 SHAP 值
    shap_values = explainer(X[idx:idx + 1], max_evals=max_evals, batch_size=batch_size)

    # pixel_values = X[idx:idx + 1] * 0.5 + 0.5

    # 打印实际类别
    print(f"True Class: {class_name}")

    # 可视化 SHAP 值
    shap.image_plot(shap_values)

    return shap_values, class_name


def concat_images(images, cols=2):
    # 每个图像的宽度和高度（假设所有图像大小一致）
    width, height = images[0].size
    # 计算拼接图像的行数
    rows = (len(images) + cols - 1) // cols
    # 创建一个新的图像，用于拼接
    concat_img = Image.new('RGB', (width * cols, height * rows))

    for i, img in enumerate(images):
        # 计算放置位置
        x = (i % cols) * width
        y = (i // cols) * height
        # 将图像粘贴到对应位置
        concat_img.paste(img, (x, y))

    return concat_img
# 调用示例
# shap_values, class_name = generate_shap_explanation(model, test_loader, idx)