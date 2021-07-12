# Various attention mechanism using Pytorch
# Author: Jermy
# Time: 2021-7-6
# Reference: https://github.com/bojone/attention/blob/master/attention_keras.py

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor, dropout
from torch.autograd import Variable
from torch.nn.modules.container import Sequential


class Configuration:
    vocab_size = 10000#default
    padding_idx = 0#default
    max_seq_len = 32
    embed_size = 300#is equal to d_model
    heads = 8
    hidden_size = 1024
    encoders = 6
    dropout = 0.0
    device = "cpu"
    num_classes = 10
    chosen_attention = "MultiHeadedAttention"


def to_mask(x: Tensor, mask: BoolTensor = None):
    # Default: using -1e9 mask
    # x.shape: (batch_size, seq_len, embed_dim)
    # mask.shape: (batch_size, seq_len, 1)
    if mask is None:
        return x
    if len(mask.size()) == 3 and mask.size(-1) == 1 and \
        mask.size(0) == x.size(0) and \
            mask.size(1) == x.size(1):
            x.masked_fill_(mask, value=torch.tensor(-1e9))#in-place
            return x
    else:
        raise ValueError("""Mask tensor does not match X tensor. See 
        https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html#torch.Tensor.masked_fill_ in detail""")

def attention(query: Tensor, key: Tensor, value: Tensor, mask: BoolTensor = None):
    # query.shape: (batch_size, len_q, dim_q)
    # key.shape: (batch_size, len_k, dim_q)
    # value.shape: (batch_size, len_k, dim_v)
    scale = math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale# (batch_size, len_q, len_k)
    if mask is not None:
        scores = to_mask(scores, mask=mask)
    scores_p = F.softmax(scores, dim=-1)
    attentioned_context = torch.matmul(scores_p, value)
    return attentioned_context


#多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout=0.0):
        #Take in model size and number of heads
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.dim_head = d_model // heads
        self.heads = heads
        self.fc_query = nn.Linear(d_model, d_model)#d_model --> heads * dim_head
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_final = nn.Linear(d_model, d_model)#heads * dim_head --> d_model
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x.shape: (batch_size, seq_len, embed_dim)
        batch_size = x.size(0)
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)# its shape: (batch_size, heads, seq_len, dim_head)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        # Apply attention on all the projected vectors in batch
        atted_x = attention(query, key, value, mask=mask)
        atted_x = atted_x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim_head)#after transpose, shape is (batch_size, seq_len, heads, dim_head)
        atted_x = self.fc_final(atted_x)#feature mapping and concatting
        atted_x = self.dropout(atted_x)
        # atted_x is context vectors
        # 残差连接
        final_x = atted_x + x
        final_x = self.layer_norm(final_x)
        return final_x, atted_x


# 参考文献：Generating Long Sequences with Sparse Transformers
# 空洞多头注意力机制：每个元素只跟相对距离为dilation倍数的元素有关联
class AtrousMultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dilation: int = 2,dropout=0.0):
        super(AtrousMultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.dilation = dilation
        self.dim_head = d_model // heads
        self.heads = heads
        self.fc_query = nn.Linear(d_model, d_model)#d_model --> heads * dim_head
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_final = nn.Linear(d_model, d_model)#heads * dim_head --> d_model
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        #x.shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size(0), x.size(1), x.size(-1)
        if seq_len % self.dilation == 0:
            padding_size = 0
        else:# != 0
            padding_size = (seq_len // self.dilation + 1) * self.dilation - seq_len
        assert (padding_size + seq_len) % self.dilation == 0
        # x = x.transpose(1, -1)#x.shape: (batch_size, embed_dim, seq_len)
        # x = F.pad(x, (0, padding_size), "constant", 0)#x.shape: (batch_size, embed_dim, seq_len+padding_size)
        # x = x.transpose(1, -1)#x.shape: (batch_size, seq_len+padding_size, embed_dim)
        copy_x = x.clone()#只有用户显式定义的tensor支持deepcopy协议，使用clone替代
        x = F.pad(x, (0, 0, 0, padding_size), "constant", 0)#x.shape: (batch_size, seq_len+padding_size, embed_dim)
        padded_seq_len = x.size(1)
        assert padded_seq_len == padding_size + seq_len
        x = x.view(-1, padded_seq_len // self.dilation, self.dilation, embed_dim)
        x = x.permute(0, 2, 1, 3)#x.shape: (batch_size, self.dilation, padded_seq_len // self.dilation, embed_dim)
        x = x.reshape(-1, padded_seq_len // self.dilation, embed_dim)#x.shape: (batch_size * self.dilation, padded_seq_len // self.dilation, embed_dim)
        query = self.fc_query(x)#their shape: (batch_size * self.dilation, padded_seq_len // self.dilation, embed_dim)
        key = self.fc_key(x)
        value = self.fc_value(x)
        #its shape: (batch_size * self.dilation, heads, padded_seq_len // self.dilation, dim_head)
        query = query.view(batch_size * self.dilation, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size * self.dilation, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size * self.dilation, -1, self.heads, self.dim_head).transpose(1, 2)
        # Apply attention
        atted_x = attention(query, key, value, mask=mask)
        #after transpose, shape is (batch_size * self.dilation, padded_seq_len // self.dilation, heads, dim_head)
        atted_x = atted_x.transpose(1, 2).contiguous()
        atted_x = atted_x.view(batch_size * self.dilation, -1, self.heads * self.dim_head)#(batch_size * self.dilation, padded_seq_len // self.dilation, embed_dim)
        # 恢复shape
        atted_x = atted_x.view(-1, self.dilation, padded_seq_len // self.dilation, embed_dim)
        atted_x = atted_x.permute(0, 2, 1, 3)
        # atted_x = atted_x.contiguous().view(-1, padded_seq_len, embed_dim)
        atted_x = atted_x.reshape(-1, padded_seq_len, embed_dim)
        if padding_size > 0:# != 0
            atted_x = atted_x[:, :-padding_size]

        # print(atted_x.size())#print次数与encoders数相同
        assert atted_x.size(0) == batch_size and atted_x.size(1) == seq_len and atted_x.size(-1) == embed_dim
        assert copy_x.size() == atted_x.size()
        # 全连接映射+残差连接
        atted_x = self.dropout(self.fc_final(atted_x))
        # atted_x is context vectors
        final_x = atted_x + copy_x
        final_x = self.layer_norm(final_x)
        return final_x, atted_x


# patch内元素间隔dilation默认为1
def extract_seq_patches(x: Tensor, kernel_size: int, dilation: int = 1):
    # x.shape: (batch_size, seq_len, embed_dim)
    seq_len, embed_dim = x.size(1), x.size(-1)
    patch_size = kernel_size + (dilation - 1) * (kernel_size - 1)
    padding_right = (patch_size - 1) // 2
    padding_left = patch_size - padding_right - 1
    # x = x.transpose(1, -1)#x.shape: (batch_size, embed_dim, seq_len)
    # padding_layer = nn.ConstantPad1d(padding=(padding_left, padding_right), value=0.0)
    # # https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html#torch.nn.ConstantPad1d
    # x = padding_layer(x)#也可以用F.pad()函数
    # x = x.transpose(1, -1)#x.shape: (batch_size, seq_len+padding_left+padding_right, embed_dim)
    x = F.pad(x, (0, 0, padding_left, padding_right), mode="constant", value=0.0)#x.shape: (batch_size, seq_len+padding_left+padding_right, embed_dim)
    x = [x[:, i: i+seq_len].detach().numpy() for i in range(0, patch_size, dilation)]
    x = torch.tensor(x)#x.shape: (kernel_size, batch_size, seq_len, embed_dim)
    # x = x.transpose(0, 1)#x.shape: (batch_size, kernel_size, seq_len, embed_dim)
    # x = x.transpose(1, 2)#x.shape: (batch_size, seq_len, kernel_size, embed_dim)
    x = x.permute(1, 2, 0, 3)#x.shape: (batch_size, seq_len, kernel_size, embed_dim)
    return x


# 局部多头注意力机制，每个元素只跟左右各neighbors的元素有关联，元素与元素的间隔dilation默认为1
class LocalMultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, neighbors = 2, dilation: int = 1, dropout = 0.0):
        super(LocalMultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        self.dilation = dilation
        self.neighbors = neighbors
        self.dim_head = d_model // heads
        self.heads = heads
        self.fc_query = nn.Linear(d_model, d_model)#d_model --> heads * dim_head
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_final = nn.Linear(d_model, d_model)#heads * dim_head --> d_model
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: BoolTensor = None):
        # x.shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len = x.size(0), x.size(1)
        kernel_size = 1 + 2 * self.neighbors
        patches_x = extract_seq_patches(x, kernel_size=kernel_size, dilation=self.dilation)#its shape: (batch_size, seq_len, kernel_size, embed_dim)
        x = x.view(x.size(0), x.size(1), 1, x.size(-1))#x.shape: (batch_size, seq_len, 1, embed_dim)
        query = self.fc_query(x)
        key = self.fc_key(patches_x)
        value = self.fc_value(patches_x)
        # 为了多头进行维度合并
        query = query.view(-1, 1, query.size(-1))#its shape: (batch_size * seq_len, 1, embed_dim)
        key = key.view(-1, kernel_size, key.size(-1))#its shape: (batch_size * seq_len, kernel_size, embed_dim)
        value = value.view(-1, kernel_size, value.size(-1))
        # 多头
        query = query.view(query.size(0), -1, self.heads, self.dim_head).transpose(1, 2)#its shape: (batch_size * seq_len, heads, 1, dim_head)
        key = key.view(key.size(0), -1, self.heads, self.dim_head).transpose(1, 2)#its shape: (batch_size * seq_len, heads, kernel_size, dim_head)
        value = value.view(value.size(0), -1, self.heads, self.dim_head).transpose(1, 2)
        # Apply attention
        atted_x = attention(query, key, value, mask=mask)#its shape: (batch_size * seq_len, heads, 1, dim_head)
        atted_x = atted_x.transpose(1, 2).contiguous()#its shape: (batch_size * seq_len, 1, heads, dim_head)
        atted_x = atted_x.view(-1, 1, self.heads * self.dim_head)
        atted_x = atted_x.view(batch_size, seq_len, -1)#its shape: (batch_size, seq_len, embed_dim)
        assert atted_x.size(-1) == self.heads * self.dim_head
        atted_x = self.fc_final(atted_x)#feature mapping and concatting
        atted_x = self.dropout(atted_x)
        # atted_x is context vectors
        # 残差连接
        x = x.view(batch_size, seq_len, -1)#x.shape: (batch_size, seq_len, embed_dim)
        final_x = atted_x + x
        final_x = self.layer_norm(final_x)
        return final_x, atted_x
        

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        # 残差连接
        out = out + x
        out = self.layer_norm(out)
        return out


# 位置Embedding
class Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 32, dropout: int = 0.0, device: str = "cpu" ):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([
            [pos / (10000 ** (i / d_model)) if i % 2 == 0 else pos / (10000 ** ((i - 1) / d_model)) for i in range(d_model)] for pos in range(max_seq_len)
        ])
        # self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pos in range(max_seq_len)])#与上面等价
        # 偶数维度用sin 奇数维度用cos
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        #x.shape: (batch_size, max_seq_len, d_model)
        #self.pe: (max_seq_len, d_model)
        #广播机制
        # out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = x + Variable(self.pe, requires_grad=False).to(self.device)#与上面等价
        out = self.dropout(out)
        return out


class Encoder_layer(nn.Module):
    def __init__(self,  d_model: int, heads: int, hidden: int, dropout: int = 0.0, chosen_attention: str = "MultiHeadedAttention"):
        super(Encoder_layer, self).__init__()
        if chosen_attention == "MultiHeadedAttention":
            self.attention = MultiHeadedAttention(d_model=d_model, heads=heads, dropout=dropout)
        elif chosen_attention == "AtrousMultiHeadedAttention":
            self.attention = AtrousMultiHeadedAttention(d_model=d_model, heads=heads, dropout=dropout)
        elif chosen_attention == "LocalMultiHeadedAttention":
            self.attention = LocalMultiHeadedAttention(d_model=d_model, heads=heads, neighbors=2, dilation=1, dropout=dropout)
        else:
            raise ValueError("Attention Mechanism Does Not Match")
        self.feed_forward = Position_wise_Feed_Forward(d_model=d_model, hidden=hidden, dropout=dropout)

    def forward(self, x: Tensor):
        out, context = self.attention(x)
        out = self.feed_forward(out)
        return out

def Encoder_layer_func(d_model: int, heads: int, hidden: int, dropout: int = 0.0):
    return Sequential(
        MultiHeadedAttention(d_model=d_model, heads=heads, dropout=dropout),
        Position_wise_Feed_Forward(d_model=d_model, hidden=hidden, dropout=dropout)
    )


class Transformer_Encoder(nn.Module):
    def __init__(self, config = Configuration, embedding_pretrained=None):
        super(Transformer_Encoder, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embedding_pretrained, freeze=False)
        else:#None
            self.embedding = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.padding_idx)

        self.position_embedding = Positional_Encoding(config.embed_size, config.max_seq_len, config.dropout, config.device)
        # self.encoder = Encoder_layer_func(config.embed_size, config.heads, config.hidden_size, config.dropout)
        self.encoder = Encoder_layer(config.embed_size, config.heads, config.hidden_size, config.dropout, config.chosen_attention)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder) for _ in range(config.encoders)
        ])
        # 假设下游是一个文本分类任务,num_classes = 10
        # 第一种做法：拼接
        self.fc = nn.Linear(config.max_seq_len * config.embed_size, config.num_classes)
        # 第二种做法： mean
        self.fc_mean = nn.Linear(config.embed_size, config.num_classes)

    def forward(self, x):
        #x.shape: (batch_size, max_seq_len)
        x = self.embedding(x)#x.shape: (batch_size, max_seq_len, embed_size)
        x = self.position_embedding(x)#增加了位置信息
        for encoder in self.encoders:
            x = encoder(x)#x.shape: (batch_size, max_seq_len, embed_size)
        # #第一种做法
        # x = x.view(x.size(0), -1)#x.shape: (batch_size, max_seq_len * embed_size)
        # out = self.fc(x)#out.shape: (batch_size, num_classes)
        #第二种做法
        x = x.mean(dim=1)#x.shape: (batch_size, embed_size)
        out = self.fc_mean(x)
        return out


if __name__=="__main__":
    # Testing Transformer-encoder
    Configuration.heads = 10
    Configuration.chosen_attention = "LocalMultiHeadedAttention"
    m = Transformer_Encoder(config=Configuration)
    x = torch.LongTensor([
        [i for i in range(32)],
        [i+10 for i in range(32)],
        [i+100 for i in range(32)]
    ])
    out = m(x)
    print(out)


