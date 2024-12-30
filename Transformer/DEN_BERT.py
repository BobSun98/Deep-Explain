import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import math
from transformers import BertTokenizer




# 定义配置类
class BertConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_labels=4
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels

# 定义 Embeddings 层
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, seq_length)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# 定义 Self-Attention 层
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # (batch_size, seq_length, num_heads, head_size)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  # (batch_size, seq_length, all_head_size)
        mixed_key_layer = self.key(hidden_states)      # (batch_size, seq_length, all_head_size)
        mixed_value_layer = self.value(hidden_states)  # (batch_size, seq_length, all_head_size)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch_size, num_heads, seq_length, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)      # (batch_size, num_heads, seq_length, head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch_size, num_heads, seq_length, head_size)

        # 计算注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (batch_size, num_heads, seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        attention_scores = attention_scores + attention_mask

        # 计算注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch_size, num_heads, seq_length, head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, num_heads, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch_size, seq_length, all_head_size)

        return context_layer, attention_probs

# 定义 Attention 输出层
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # (batch_size, seq_length, hidden_size)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接
        return hidden_states

# 定义 Attention 层
class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs

# 定义 Intermediate 层
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # (batch_size, seq_length, intermediate_size)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# 定义 Output 层
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # (batch_size, seq_length, hidden_size)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接
        return hidden_states


# 定义适用于BERT的DEB模块
class DeepExplanationBERT(nn.Module):
    def __init__(self, config ):
        super(DeepExplanationBERT, self).__init__()
        self.pred_fc = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.W = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.activation = None
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.deep_hidden = None
    def forward(self, x, x_next):
        """
        x: [batch_size, seq_length, hidden_size] - 上一层的激活
        x_next: [batch_size, seq_length, hidden_size] - 当前层的输出
        """
        # 全局平均池化（GAP）沿着序列长度维度进行平均
        x_next = self.dense(x_next) # 降维到 768
        x_squeeze = x.mean(dim=1)        # [batch_size, hidden_size]
        x_next_squeeze = x_next.mean(dim=1)  # [batch_size, hidden_size]

        # 结合x和x_next
        x_combined = x_squeeze + x_next_squeeze  # [batch_size, hidden_size]
        x_combined = F.relu(x_combined)
        # 通过线性层W
        a = self.W(x_combined)  # [batch_size, hidden_size]
        self.deep_hidden = x_combined
        b = torch.sigmoid(a)    # [batch_size, hidden_size]
        # Attention Gate: 调整b的维度以与x匹配
        b = b.unsqueeze(1)      # [batch_size, 1, hidden_size]
        # 使用广播机制将b应用于x

        activation_all = x + x_next
        self.activation = activation_all

        x_attended = activation_all * b       # [batch_size, seq_length, hidden_size]

        x_attended = self.dropout(x_attended)
        x_attended = self.LayerNorm(x_attended)

        # 深度监督
        deep_pred = self.pred_fc(a)  # [batch_size, num_classes]

        return x_attended, deep_pred

# 定义 Transformer 编码器层
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        # self.output = BertOutput(config)
        self.deb = DeepExplanationBERT(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output) #3072
        # layer_output = self.output(intermediate_output, attention_output) 不在这里残差了,整体暂时跳过
        layer_output, deep_pred = self.deb(attention_output,intermediate_output)
        return layer_output, attention_probs, deep_pred

# 定义 Transformer 编码器（堆叠多个编码器层）
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_attentions = []
        deep_preds = []
        for layer_module in self.layer:
            hidden_states, attention_probs,deep_pred = layer_module(hidden_states, attention_mask)
            all_attentions.append(attention_probs)
            deep_preds.append(deep_pred)
        return hidden_states, all_attentions, deep_preds

# 定义 Pooler 层
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 取 `[CLS]` token 的表示
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# 定义分类头
class BertClassificationHead(nn.Module):
    def __init__(self, config):
        super(BertClassificationHead, self).__init__()
        # 定义直接的 weight 和 bias 参数
        self.weight = nn.Parameter(torch.Tensor(config.num_labels, config.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(config.num_labels))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, pooled_output):
        logits = F.linear(pooled_output, self.weight, self.bias)
        return logits

# 定义自定义 BERT 模型
class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(CustomBertForSequenceClassification, self).__init__()
        self.num_labels = config.num_labels

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPooler(config)
        # self.classifier = BertClassificationHead(config)  # 使用修改后的分类头

    #     # 初始化权重
    #     self.init_weights()
    #
    # def init_weights(self):
    #     # 初始化分类头权重
    #     nn.init.xavier_uniform_(self.classifier.weight)
    #     nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 调整 attention_mask 的形状和值
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
        attention_mask = (1.0 - attention_mask) * -10000.0          # 将 1 转换为 0，0 转换为 -10000.0

        # 嵌入层
        embedding_output = self.embeddings(input_ids, token_type_ids=None)

        # 编码器
        encoder_output, attentions,deep_preds = self.encoder(embedding_output, attention_mask)

        if self.training:
            return deep_preds
        else: #eval 模式
            return deep_preds[-1]

        # # 池化层
        # pooled_output = self.pooler(encoder_output)
        #
        # # 分类头
        # logits = self.classifier(pooled_output)
        #
        # outputs = logits

        # return outputs  # (loss), logits


if __name__ == '__main__':
    import torch
    # 初始化配置
    config = BertConfig(
        num_labels=4
    )

    # 初始化模型
    model = CustomBertForSequenceClassification(config)
    model.train()  # 设置模型为评估模式

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义批次大小和序列长度
    batch_size = 1
    seq_length = 70
    vocab_size = config.vocab_size  # 通常为30522

    # 模拟 input_ids：随机整数在1到vocab_size-1之间（0通常是[PAD] token）
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_length))

    # 模拟 attention_mask：全1表示所有token都需要被注意
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

    # 将张量移动到设备
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    print("Logits:", logits)

# # 移动到设备（CPU 或 GPU）
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
#
# # 设置为评估模式
# model.eval()
#
# # 初始化分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# # 示例文本
# text = "The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)"
# inputs = tokenizer(text, padding='max_length', truncation=True, max_length=70, return_tensors='pt')
#
# input_ids = inputs['input_ids'].to(device)
# attention_mask = inputs['attention_mask'].to(device)
#
# # 模型前向传播
# with torch.no_grad():
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     logits = outputs[0]
#
# print(logits)
