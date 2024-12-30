# avg_drop_computation.py

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import shap
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from IPython.display import HTML

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'

os.environ['HF_HOME'] = '/home/bobsun/cambrige/MedMinist/hugginface'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/bobsun/cambrige/MedMinist/hugginface'


# ==========================
# Helper Functions
# ==========================

def saliency_to_latex(tokens, saliency, cmap='viridis'):
    """
    将tokens和saliency scores转换为带有背景颜色和透明度的LaTeX代码，使用RGB颜色模型。
    """
    colormap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin=0, vmax=1)
    latex_tokens = []
    for token, score in zip(tokens, saliency):
        rgba = colormap(norm(score))
        r, g, b, _ = rgba
        opacity = 0.4 + 0.6 * score
        latex_token = (
            f'\\begin{{tikzpicture}}[baseline=(word.base)]\n'
            f'  \\node[fill={{rgb,1:red,{r:.2f}; green,{g:.2f}; blue,{b:.2f}}}, '
            f'opacity={opacity:.2f}, inner sep=1pt, rounded corners=2pt] (word) '
            f'{{\\strut {token}}};\n'
            f'\\end{{tikzpicture}}'
        )
        latex_tokens.append(latex_token)
    latex_line = ' '.join(latex_tokens)
    return latex_line


def render_text_with_saliency(tokens, saliency, cmap='Reds'):
    """
    Render text with tokens colored according to saliency scores.
    """
    norm = Normalize(vmin=np.min(saliency), vmax=np.max(saliency))
    cmap = cm.get_cmap('viridis')
    colors_hex = [plt.cm.colors.rgb2hex(cmap(norm(score))) for score in saliency]
    colored_tokens = [
        f'<span style="background-color:{color}; padding:2px; border-radius:3px;">{token}</span>'
        for token, color in zip(tokens, colors_hex)
    ]
    html_text = ' '.join(colored_tokens)
    return HTML(html_text)


def encode_text(tokenizer, text, max_length=71):
    """
    编码文本为模型输入格式。
    """
    return tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')


def perturb_input(tokenizer, input_ids, top_k_indices):
    """
    根据top_k_indices掩蔽输入中的重要token。
    """
    perturbed_ids = input_ids.clone()
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids('[MASK]')
    perturbed_ids[0, top_k_indices] = MASK_TOKEN_ID
    return perturbed_ids


# ==========================
# Load and Prepare Data
# ==========================

print("Loading AG News dataset...")
dataset = load_dataset('ag_news')
test_dataset = dataset['test'].select(range(512))  # 选择前512个测试样本

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def preprocess_function(examples):
    tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=71)
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': examples['label']
    }


print("Tokenizing dataset...")
tokenized_datasets = test_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
test_dataset = tokenized_datasets

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=71, return_tensors='pt')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

# ==========================
# Load Models
# ==========================

print("Loading BERT model for SHAP and evaluation...")
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
bert_model_path = 'bert_ag_news.pth'  # 确保该路径正确
bert_model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = bert_model.to(device)
bert_model.eval()

print("Loading DEN model...")
import importlib
import Transformer.DEN_BERT  # 确保Transformer/DEN_BERT.py存在且正确

importlib.reload(Transformer.DEN_BERT)

from Transformer.DEN_BERT import BertConfig, CustomBertForSequenceClassification

config = BertConfig(
    num_labels=4
)

den_model_path = 'den_ag_news.pth'
den_state_dict = torch.load(den_model_path, map_location='cpu')
den_model = CustomBertForSequenceClassification(config)
den_model.load_state_dict(den_state_dict, strict=True)
den_model = den_model.to(device)
den_model.eval()

# ==========================
# Initialize SHAP
# ==========================

print("Initializing SHAP...")
classifier_pipeline = pipeline(
    "text-classification",
    model=bert_model,
    tokenizer=tokenizer,
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

pmodel = shap.models.TransformersPipeline(classifier_pipeline, rescale_to_logits=False)
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(pmodel, masker)


# ==========================
# Define Saliency Functions for DEN
# ==========================

def bert_cam_one_layer(layer_idx, class_idx):
    layers = den_model.encoder.layer
    activation = layers[layer_idx].deb.activation.squeeze(0).cpu().detach().numpy()  # [70,768]
    W = layers[layer_idx].deb.W.weight
    pred_fc = layers[layer_idx].deb.pred_fc.weight
    weight = pred_fc @ W
    neuron = weight[class_idx].cpu().detach().numpy()  # [768]
    a = neuron * activation
    sailency = np.sum(a, axis=1)
    return sailency


def get_den_saliency(model, input_ids, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs
    probs = torch.nn.Softmax(dim=1)(logits)
    predictions = torch.argmax(probs, dim=-1).item()
    sailencies = []
    for i in range(12):  # 假设有12层
        sailency = bert_cam_one_layer(i, predictions)
        sailencies.append(sailency)
    sailencies = np.array(sailencies)
    # 只取前3层
    sailencies = sailencies[0:3] # 3
    # Sum the first 3 layers
    sailencies = np.sum(sailencies, axis=0)
    return sailencies


# ==========================
# Define Perturbation and AVG Drop Computation
# ==========================

def compute_avg_drop():
    den_drops = []
    shap_drops = []
    total_samples = len(test_dataloader)
    print(f"Computing AVG Drop over {total_samples} samples...")

    for idx, batch in enumerate(tqdm(test_dataloader, desc="Processing samples")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        label = labels.item()

        # 原始预测
        with torch.no_grad():
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # 修改这里，提取 logits
            probs = torch.nn.Softmax(dim=1)(logits)
            original_prob = probs[0, label].item()

        # DEN Saliency
        den_saliency = get_den_saliency(den_model, input_ids, attention_mask)
        # Normalize DEN saliency
        den_saliency_norm = (den_saliency - den_saliency.min()) / (den_saliency.max() - den_saliency.min() + 1e-10)

        # SHAP Saliency
        # 获取原始文本
        # 注意：tokenized_datasets 不再包含 'text' 列，所以需要从原始 dataset 中获取
        original_text = dataset['test'][idx]['text']
        shap_values = explainer([original_text])

        # 获取预测类别的索引
        predicted_label = np.argmax(probs.cpu().numpy(), axis=1)[0]
        shap_values_target = shap_values.values[0, :, predicted_label]
        # Normalize SHAP saliency
        shap_saliency_norm = (shap_values_target - shap_values_target.min()) / (
                    shap_values_target.max() - shap_values_target.min() + 1e-10)

        # Get tokens excluding padding
        attention_np = attention_mask.squeeze().detach().cpu().numpy()
        valid_indices = np.where(attention_np == 1)[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][valid_indices])
        den_saliency_norm = den_saliency_norm[:len(valid_indices)]
        shap_saliency_norm = shap_saliency_norm[:len(valid_indices)]

        # Define top K
        K = max(1, int(0.2 * len(valid_indices)))  # 10%的tokens，至少1个
        K = min(K, 20)  # 不超过10个

        # DEN Method
        den_topk_indices = np.argsort(-den_saliency_norm)[:K]
        perturbed_den_ids = perturb_input(tokenizer, input_ids, valid_indices[den_topk_indices])
        with torch.no_grad():
            outputs_den = bert_model(perturbed_den_ids, attention_mask=attention_mask)
            logits_den = outputs_den.logits  # 修改这里，提取 logits
            probs_den = torch.nn.Softmax(dim=1)(logits_den)
            perturbed_den_prob = probs_den[0, label].item()
        den_drop = original_prob - perturbed_den_prob
        den_drops.append(den_drop)

        # SHAP Method
        shap_topk_indices = np.argsort(-shap_saliency_norm)[:K]
        perturbed_shap_ids = perturb_input(tokenizer, input_ids, valid_indices[shap_topk_indices])
        with torch.no_grad():
            outputs_shap = bert_model(perturbed_shap_ids, attention_mask=attention_mask)
            logits_shap = outputs_shap.logits  # 修改这里，提取 logits
            probs_shap = torch.nn.Softmax(dim=1)(logits_shap)
            perturbed_shap_prob = probs_shap[0, label].item()
        shap_drop = original_prob - perturbed_shap_prob
        shap_drops.append(shap_drop)

        # Optional: Print debug information for first few samples
        if idx < 5:
            print(f"\nSample {idx + 1}:")
            print(f"Original Probability for true label ({label}): {original_prob:.4f}")
            print(f"Perturbed DEN Probability: {perturbed_den_prob:.4f}, Drop: {den_drop:.4f}")
            print(f"Perturbed SHAP Probability: {perturbed_shap_prob:.4f}, Drop: {shap_drop:.4f}")
            # Optionally render saliency (requires Jupyter)
            # display(render_text_with_saliency(tokens, den_saliency_norm))
            # display(render_text_with_saliency(tokens, shap_saliency_norm))

    avg_den_drop = np.mean(den_drops)
    avg_shap_drop = np.mean(shap_drops)

    print("\n==========================")
    print(f"Average DEN Drop: {avg_den_drop:.4f}")
    print(f"Average SHAP Drop: {avg_shap_drop:.4f}")
    print("==========================")


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    compute_avg_drop()