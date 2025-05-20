import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import argparse
#import data_prepare_eav as data_prepare
#import data_prepare_faced as data_prepare
import data_prepare_seed as data_prepare
import math
from loss import CustomLoss
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from scipy.stats import entropy
from torch.nn.functional import one_hot
from datetime import datetime

class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x): # note we carefully add the positional encoding, omitted
        x = x #+ self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class MultiScaleAttention(nn.Module):###scales?
    """
    多尺度特征注意力（去除相对位置编码和掩码），并加入 Lipschitz 注意力限制
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, scales=[1, 2, 4], lipschitz_num=1.0):
        super(MultiScaleAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scales = scales
        self.scaling = self.head_dim ** -0.5
        self.lipschitz_num = lipschitz_num

        # QKV 投影
        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.v_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim).double(),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim).double()
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim).double()

    def lipschitz_normalize(self, scores, query, key):
        """
        Lipschitz 归一化
        """
        # 计算 query 和 key 的 F-范数
        query_norm = torch.norm(query, dim=-1, keepdim=True)  # (batch_size, num_heads, seq_len, 1)
        key_norm = torch.norm(key, dim=-1, keepdim=True)  # (batch_size, num_heads, seq_len, 1)

        # 计算最大范数
        max_query_norm = query_norm.max(dim=-2, keepdim=True)[0]  # (batch_size, num_heads, 1, 1)
        max_key_norm = key_norm.max(dim=-2, keepdim=True)[0]  # (batch_size, num_heads, 1, 1)

        # Lipschitz 归一化
        lipschitz_denominator = torch.clamp(max_query_norm * max_key_norm, min=1e-6)  # 避免除零
        normalized_scores = scores / lipschitz_denominator

        # 应用 Lipschitz 数限制
        normalized_scores = normalized_scores * self.lipschitz_num
        return normalized_scores

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        query = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.einsum("bhld,bhmd->bhlm", query, key) * self.scaling

        # 应用 Lipschitz 归一化
        scores = self.lipschitz_normalize(scores, query, key)

        # 应用 Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        # 计算输出
        output = torch.einsum("bhlm,bhmd->bhld", attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output
        
class TransformerBlock(nn.Module):
    """
    改进版 TransformerBlock，去除相对位置编码和掩码
    """
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, scales=[1, 2, 4, 8],lipschitz_num=1.0):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.scales = scales
        self.lipschitz_num=lipschitz_num

        # 多尺度特征注意力
        self.attention = MultiScaleAttention(embed_dim, num_heads, dropout, scales,lipschitz_num=self.lipschitz_num)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward).double(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim).double()
        )

        # 层归一化
        self.layernorm1 = nn.LayerNorm(embed_dim).double()
        self.layernorm2 = nn.LayerNorm(embed_dim).double()

    def forward(self, x):
        # 多尺度特征注意力
        attn_output = self.attention(x)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        x = self.layernorm1(x + attn_output)

        # 前馈网络
        mlp_output = self.mlp(x)
        mlp_output = F.dropout(mlp_output, p=self.dropout, training=self.training)
        x = self.layernorm2(x + mlp_output)

        return x
    
class DilatedAttentionBlock_attention(nn.Module):
    def __init__(self, in_channels, seq_length, out_channels, dropout_rate=0.0, num_heads=2, lipschitz_num=1.0):
        super(DilatedAttentionBlock_attention, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"
        self.lipschitz_num = lipschitz_num

        # 时间注意力模块
        self.q_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.k_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.v_linear_time = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        ).double()
        self.out_linear_time = nn.Linear(out_channels, out_channels).double()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm_time = nn.LayerNorm(in_channels).double()

    def forward(self, x_in):
        batch_size, in_channels, seq_length = x_in.shape
        x = x_in.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, in_channels)

        # 时间注意力
        Q_time = self.q_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K_time = self.k_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V_time = self.v_linear_time(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores_time = torch.matmul(Q_time, K_time.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用 Lipschitz 约束
        # 计算注意力分数的范数
        score_norm = torch.norm(attention_scores_time, dim=-1, keepdim=True)
        # 归一化注意力分数z
        attention_scores_time = attention_scores_time / (score_norm * self.lipschitz_num + 1e-12)

        # 计算注意力权重
        attention_weights_time = F.softmax(attention_scores_time, dim=-1)

        # 计算注意力输出
        attention_output_time = torch.matmul(attention_weights_time, V_time).transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attention_output_time = self.out_linear_time(attention_output_time)
        attention_output_time = self.norm_time(attention_output_time)

        # 恢复形状并应用激活函数和 Dropout
        attention_output_time = attention_output_time.permute(0, 2, 1)  # (batch_size, out_channels, seq_length)
        combined_output = self.elu(attention_output_time)
        combined_output = self.dropout(combined_output)
        combined_output = combined_output + x_in  # 残差连接

        return combined_output

class ResidualMLP(nn.Module):
    """
    带残差连接的多层感知机
    """
    def __init__(self, input_dim, hidden_dim):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, input_dim).double()
        self.dropout = nn.Dropout(0.1).double()

    def forward(self, x, lipschitz_num=1.0):
        # 动态调整权重以满足 Lipschitz 约束
        self.fc1.weight.data = self.fc1.weight.data / self.fc1.weight.data.norm() * lipschitz_num
        self.fc2.weight.data = self.fc2.weight.data / self.fc2.weight.data.norm() * lipschitz_num

        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.dropout(out)
        return residual + out  # 添加残差连接

class FeatureAttentionModule(nn.Module):
    """
    特征注意力模块，计算输入矩阵中特征维度之间的相关性
    """
    def __init__(self, input_dim, hidden_dim, lipschitz_num=1.0):
        super(FeatureAttentionModule, self).__init__()
        self.query_mlp = ResidualMLP(input_dim, hidden_dim)
        self.key_mlp = ResidualMLP(input_dim, hidden_dim)
        self.value_mlp = ResidualMLP(input_dim, hidden_dim)
        self.lipschitz_num = lipschitz_num

    def forward(self, matrix):
        # 获取输入矩阵的维度
        seq_len, input_dim = matrix.shape

        # 通过 MLP 获取 Q, K, V
        Q = self.query_mlp(matrix.T, lipschitz_num=self.lipschitz_num)  # (input_dim, seq_len)
        K = self.key_mlp(matrix.T, lipschitz_num=self.lipschitz_num)    # (input_dim, seq_len)
        V = self.value_mlp(matrix.T, lipschitz_num=self.lipschitz_num)  # (input_dim, seq_len)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(input_dim, dtype=torch.float64))
        attention_scores = F.softmax(attention_scores, dim=-1)  # 归一化注意力分数

        # 应用注意力权重到 V
        attention_output = torch.matmul(attention_scores, V)  # (input_dim, input_dim)

        # 返回注意力输出和注意力分数
        return attention_output.T, attention_scores

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution Layer
    """
    def __init__(self, in_features, out_features):
        """
        初始化图卷积层
        :param in_features: 输入特征的维度
        :param out_features: 输出特征的维度
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).double()
        # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化权重矩阵
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        """
        前向传播
        :param x: 节点特征矩阵，形状为 (N, in_features)，N 是节点数量
        :param adj: 邻接矩阵，形状为 (N, N)
        :return: 更新后的节点特征矩阵
        """
        # 1. 邻接矩阵归一化
        # 添加自连接
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        # 计算度矩阵
        degree = adj.sum(1)
        # 计算归一化因子
        D_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = D_inv_sqrt.view(-1, 1)
        # 归一化邻接矩阵
        adj_normalized = D_inv_sqrt * adj * D_inv_sqrt.t()

        # 2. 图卷积操作
        support = torch.mm(x, self.weight.to(x.device))  # 特征矩阵与权重矩阵相乘
        out = torch.mm(adj_normalized, support)  # 邻接矩阵与特征矩阵相乘

        # 3. 应用非线性激活函数
        out = F.relu(out)
        return out

class EnhancedNSAM(nn.Module):
    def __init__(self, num_channels: int = 30, seq_length: int = 250, sampling_rate: float = 250.0, lipschitz_num: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        self.lipschitz_num = lipschitz_num  # Lipschitz 常数

        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # 注意力机制
        self.channel_attention = nn.Sequential(
            nn.Linear(num_channels, num_channels, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(num_channels, num_channels, dtype=torch.float64),
            nn.Sigmoid()
        )

        self.spectral_attention = nn.Sequential(
            nn.Linear(len(self.bands), len(self.bands), dtype=torch.float64),
            nn.GELU(),
            nn.Linear(len(self.bands), len(self.bands), dtype=torch.float64),
            nn.Softmax(dim=-1)
        )

        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.norm = nn.LayerNorm(seq_length, dtype=torch.float64)

    def apply_lipschitz_constraint(self):
        """
        应用 Lipschitz 约束，限制线性层的权重谱范数。
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    weight_norm = torch.linalg.svdvals(module.weight)[0]  # 计算谱范数
                    module.weight.data = module.weight.data / weight_norm * self.lipschitz_num

    def lipschitz_attention(self, scores):
        """
        应用 Lipschitz 约束的注意力分数归一化。
        """
        # 归一化注意力分数，使其满足 Lipschitz 约束
        scores_norm = torch.norm(scores, dim=-1, keepdim=True)
        return scores / (scores_norm + 1e-6) * self.lipschitz_num

    def get_band_mask(self, freqs: torch.Tensor, band: str) -> torch.Tensor:
        low, high = self.bands[band]
        return ((freqs >= low) & (freqs <= high)).to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape [batch_size, num_channels, seq_length]
        Returns:
            output: 增强后的信号
            extracted_features: 提取的特征，包括频带特征、能量、注意力权重等
        """
        # 确保输入数据类型为 double
        x = x.to(dtype=torch.float64)
        identity = x
        batch_size = x.shape[0]

        # 应用 Lipschitz 约束
        self.apply_lipschitz_constraint()

        # FFT变换
        X = torch.fft.rfft(x, dim=-1)  # [batch_size, num_channels, freq]
        freqs = torch.fft.rfftfreq(self.seq_length, 1 / self.sampling_rate).to(x.device, dtype=torch.float64)

        # 频带分解和处理
        band_features = {}
        band_powers = []

        for band in self.bands.keys():
            mask = self.get_band_mask(freqs, band).to(x.device)
            X_band = X * mask.unsqueeze(0).unsqueeze(0)
            band_features[band] = X_band
            power = torch.sum(torch.abs(X_band).pow(2), dim=-1)  # [batch_size, num_channels]
            band_powers.append(power)

        # [batch_size, num_channels, num_bands]
        band_powers = torch.stack(band_powers, dim=-1)

        # 通道注意力
        channel_weights = self.channel_attention(band_powers.mean(dim=-1))
        channel_weights = channel_weights.unsqueeze(-1)  # [batch_size, num_channels, 1]

        # 频谱注意力
        spectral_input = band_powers.mean(dim=1)  # [batch_size, num_bands]
        spectral_scores = self.spectral_attention(spectral_input)  # [batch_size, num_bands]

        # 应用 Lipschitz 约束的注意力分数归一化
        spectral_weights = self.lipschitz_attention(spectral_scores)

        # 特征重组
        X_combined = torch.zeros_like(X, dtype=torch.complex128)  # 修改为复数张量
        for i, band in enumerate(self.bands.keys()):
            channel_weights_complex = channel_weights.to(dtype=torch.complex128)
            spectral_weights_complex = spectral_weights[:, i:i+1].unsqueeze(1).to(dtype=torch.complex128)
            X_combined += (band_features[band] * channel_weights_complex * spectral_weights_complex)

        # 返回时域
        output = torch.fft.irfft(X_combined, n=self.seq_length, dim=-1)
        output = self.norm(output)

        # 残差连接
        alpha = torch.sigmoid(self.alpha)
        output = alpha * output + (1 - alpha) * identity

        # 提取的特征
        extracted_features = {
            "band_features": band_features,  # 频带特征
            "band_powers": band_powers,      # 频带能量
            "channel_weights": channel_weights,  # 通道注意力权重
            "spectral_weights": spectral_weights  # 频谱注意力权重
        }

        return output, extracted_features

def normalize_tensor(tensor):
    # 计算每个样本的均值和标准差（沿着特征维度）
    mean = tensor.mean(dim=1, keepdim=True)  # 形状为 (128, 1)
    std = tensor.std(dim=1, keepdim=True)   # 形状为 (128, 1)
    transformed_tensor = torch.exp(tensor) - 1
    # 归一化：(x - mean) / std
    normalized_tensor = (tensor - mean) / (std + 1e-8)  # 防止除以0
    return normalized_tensor
'''
def normalize_tensor(tensor, lipschitz_constant, eps=1e-8):
    """
    对张量进行归一化，并引入Lipschitz约束。
    
    参数:
        tensor (torch.Tensor): 输入张量，形状为 (batch_size, feature_dim)。
        lipschitz_constant (float): Lipschitz常数，用于控制归一化的梯度变化。
        eps (float): 防止除以零的小常数。
    
    返回:
        torch.Tensor: 归一化后的张量。
    """
    # 计算每个样本的均值和标准差（沿着特征维度）
    mean = tensor.mean(dim=1, keepdim=True)  # 形状为 (batch_size, 1)
    std = tensor.std(dim=1, keepdim=True)    # 形状为 (batch_size, 1)
    
    # 原始归一化：(x - mean) / std
    normalized_tensor = (tensor - mean) / (std + eps)
    
    # Lipschitz约束：限制归一化后的梯度变化
    # 计算归一化后的梯度范数
    grad_norm = torch.norm(normalized_tensor, dim=1, keepdim=True)  # 形状为 (batch_size, 1)
    
    # 限制梯度范数不超过Lipschitz常数
    lipschitz_factor = torch.clamp(grad_norm / lipschitz_constant, min=1.0)
    
    # 应用Lipschitz约束
    lipschitz_normalized_tensor = normalized_tensor / (lipschitz_factor + eps)
    
    return lipschitz_normalized_tensor
'''
class GCN_energy(nn.Module):
    def __init__(self, in_features, out_features,dropout=0.1):
        super(GCN_energy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).double()
        nn.init.xavier_uniform_(self.weight)
        self.dropout=nn.Dropout(dropout)

    def forward(self, features, adj):
        """
        :param features: 输入特征，形状为 (batchsize, channel, energy)
        :param adj: 邻接矩阵，形状为 (channel, channel)
        :return: 输出特征，形状为 (batchsize, channel, out_features)
        """
        batchsize, channel, energy = features.size()
        # 添加自环
        adj_with_self_loop = adj + torch.eye(channel).to(adj.device).unsqueeze(0).expand(batchsize, -1, -1).double()
        # 计算度矩阵 D
        D = torch.diag_embed(torch.pow(adj_with_self_loop.sum(dim=2), -0.5))
        # 归一化邻接矩阵
        adj_normalized = torch.bmm(torch.bmm(D, adj_with_self_loop), D)
        # 图卷积操作
        output = torch.bmm(adj_normalized, features)  # (batchsize, channel, energy)
        output = torch.bmm(output, self.weight.unsqueeze(0).expand(batchsize, -1, -1).to(output.device))  # (batchsize, channel, out_features)
        output=self.dropout(output)
        output = F.relu(output)
        return output
    
class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel,timepoint, dropout=0.1, num_class=5,lipschitz_num=1.0):
        super().__init__()
        self.channel=eeg_channel
        self.time=timepoint
        self.num_class=num_class
        self.lipschitz_num=lipschitz_num
        
        self.conv_c = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        ).double()
        
        self.transformer_conv_channel = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout,scales=[1, 2, 4],lipschitz_num=self.lipschitz_num),
        ).double()
        self.dab_conv_channel=nn.Sequential(
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.3,4,self.lipschitz_num),
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.2,4,self.lipschitz_num),
            DilatedAttentionBlock_attention(eeg_channel*2,timepoint,eeg_channel*2,0.1,4,self.lipschitz_num),
        )
        self.mlp_conv1 = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, num_class),
        ).double()
        self.attention_relation_channel=FeatureAttentionModule(timepoint,timepoint,self.lipschitz_num)

        self.gcn_channel=GraphConvolution(timepoint,timepoint)

        self.c_x = nn.Parameter(torch.tensor(1.0))

        self.nsam=EnhancedNSAM(eeg_channel,timepoint,timepoint//10,self.lipschitz_num)
        self.final=nn.Sequential(
            nn.LayerNorm(5+6*eeg_channel, dtype=torch.float64),
            nn.Linear(5+6*eeg_channel, (5+6*eeg_channel)//2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear((5+6*eeg_channel)//2, num_class),
        ).double()

        self.attention_relation_channel_energy=FeatureAttentionModule(5,5,self.lipschitz_num)
        self.gcn_energy=GCN_energy(5,5,dropout)
        self.c_energy = nn.Parameter(torch.tensor(1.0))
        self.band_energy_out=nn.Sequential(
            nn.LayerNorm(5*eeg_channel, dtype=torch.float64),
            nn.Linear(5*eeg_channel, 5*eeg_channel//2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(5*eeg_channel//2, num_class),
        ).double()

        self.band_magnitudes_out=nn.Sequential(
            nn.LayerNorm(eeg_channel*5*(timepoint//2+1), dtype=torch.float64),
            nn.Linear(eeg_channel*5*(timepoint//2+1), eeg_channel*5*(timepoint//2+1)//10),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel*5*(timepoint//2+1)//10, num_class),
        ).double()

        self.out1 = nn.Parameter(torch.tensor(1.0))
        self.out2 = nn.Parameter(torch.tensor(1.0))
        self.out3 = nn.Parameter(torch.tensor(1.0))
        self.out4 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = x.to(dtype=torch.float64)#batchsize,channel,time
        batchsize,channel,time=x.shape
        
        x_c = self.conv_c(x)
        channel_list=[]
        attention_output_list = []
        for b in x_c:
            attention_output,channel_relation=self.attention_relation_channel(b.T)
            channel_list.append(channel_relation)
            attention_output_list.append(attention_output)
        channel_adj = torch.stack(channel_list, dim=0)  # batchsize, time, time
        channel_adj = channel_adj / channel_adj.sum(dim=-1, keepdim=True)  # 归一化邻接矩阵
        attention_output_stack = torch.stack(attention_output_list, dim=0).permute(0,2,1)
        # 处理 x_conv1
        x_conv1 = self.transformer_conv_channel(x_c.permute(0, 2, 1)).permute(0, 2, 1)
        x_conv1 = self.dab_conv_channel(x_conv1)
        gcn_channel_list=[]
        for b in range(batchsize):
            x_gcn = self.gcn_channel(x_conv1[b, :, :], channel_adj[b,:,:])  # 图卷积，输入为通道特征和邻接矩阵
            x_gcn = x_gcn.permute(1, 0)  # 调整维度
            gcn_channel_list.append(x_gcn)
        x_gcn_c = torch.stack(gcn_channel_list, dim=0)
        x_gcn_c = x_gcn_c.permute(0,2,1)
        x_conv1 = x_conv1 + self.c_x*x_gcn_c 
        x_conv1 = x_conv1.mean(dim=2)
        x_out1 = self.mlp_conv1(x_conv1)

        x_nsam,x_nsam_features=self.nsam(x)
        x_nsam_band_feature=x_nsam_features["band_features"]
        x_nsam_band_powers=x_nsam_features["band_powers"]
        x_nsam_channel_weights=x_nsam_features["channel_weights"]
        x_nsam_spectral_weights=x_nsam_features["spectral_weights"]
        x_nsam_band_powers_flat = x_nsam_band_powers.view(x_nsam_band_powers.size(0), -1)  # [batch_size, num_channels * num_bands]
        # 展平 channel_weights
        x_nsam_channel_weights_flat = x_nsam_channel_weights.squeeze(-1)  # [batch_size, num_channels]
        # 融合所有特征
        x_nsam_combined_features = torch.cat([x_nsam_band_powers_flat, x_nsam_channel_weights_flat, x_nsam_spectral_weights], dim=1)
        x_out2=self.final(x_nsam_combined_features)

        x_nsam_band_energy=[]
        for band, feature in x_nsam_band_feature.items():
            energy = torch.sum(torch.abs(feature) ** 2, dim=-1)  # [batch_size, num_channels]
            x_nsam_band_energy.append(energy)
        x_band_energies = torch.stack(x_nsam_band_energy, dim=-1)
        channel_list_energy=[]
        attention_output_list_energy=[]
        for b in x_band_energies:
            attention_energy,channel_relation=self.attention_relation_channel_energy(b.T)
            channel_list_energy.append(channel_relation)
            attention_output_list_energy.append(attention_energy)
        channel_adj_energy = torch.stack(channel_list_energy, dim=0)  # batchsize, time, time
        channel_adj_energy = channel_adj_energy/ channel_adj_energy.sum(dim=-1, keepdim=True)  # 归一化邻接矩阵
        attention_output_stack_energy = torch.stack(attention_output_list_energy, dim=0).permute(0,2,1)
        x_band_energies_conv = self.gcn_energy(x_band_energies,channel_adj_energy)
        x_band_energies = x_band_energies + self.c_energy*x_band_energies_conv
        x_band_energies_flat = x_band_energies.view(x_band_energies.size(0), -1)
        x_out3=self.band_energy_out(x_band_energies_flat)

        
        x_nasm_band_magnitudes = []
        for band, feature in x_nsam_band_feature.items():
            magnitude = torch.abs(feature)  # [batch_size, num_channels, freq_bins]
            x_nasm_band_magnitudes.append(magnitude)
        x_band_magnitudes = torch.stack(x_nasm_band_magnitudes, dim=-1)
        x_band_magnitudes_flat = x_band_magnitudes.view(x_band_magnitudes.size(0), -1)
        x_out4=self.band_magnitudes_out(x_band_magnitudes_flat)

        normalized_tensor1 = normalize_tensor(x_out1)
        normalized_tensor2 = normalize_tensor(x_out2)
        normalized_tensor3 = normalize_tensor(x_out3)
        normalized_tensor4 = normalize_tensor(x_out4)
        '''
        normalized_tensor1 = normalize_tensor(x_out1,lipschitz_constant=self.lipschitz_num)
        normalized_tensor2 = normalize_tensor(x_out2,lipschitz_constant=self.lipschitz_num)
        normalized_tensor3 = normalize_tensor(x_out3,lipschitz_constant=self.lipschitz_num)
        normalized_tensor4 = normalize_tensor(x_out4,lipschitz_constant=self.lipschitz_num)
        '''
        x_concat = torch.stack((
            self.out1*normalized_tensor1, 
            self.out2*normalized_tensor2, 
            self.out3*normalized_tensor3, 
            self.out4*normalized_tensor4,
        ), dim=0)
        x_out = torch.mean(x_concat, dim=0)
        return x_out

class EEGModelTrainer:
    def __init__(self,  train_dataloader, val_dataloader, model = [], sub = '', lr=0.001, batch_size = 64):
        if model:
            self.model = model
        else:
            self.model = EEGClassificationModel(eeg_channel=30)

        #self.tr, self.tr_y, self.te, self.te_y = DATA
        self.batch_size = batch_size
        self.test_acc = float()

        '''
        self.train_dataloader = self._prepare_dataloader(self.tr, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te, self.te_y, shuffle=False)
        '''
        self.train_dataloader = train_dataloader
        self.test_dataloader = val_dataloader

        self.initial_lr = lr
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = CustomLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)

        # Automatically use GPU if available, else fallback to CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)
        model = torch.compile(self.model)
        model.to(self.device)
        print(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        accuracies = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
                accuracies.extend((predicted == labels).cpu().numpy())
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.2f}')
        return accuracy, predictions

    def calculate_f1(self, labels, preds, num_classes):
        """
        使用PyTorch计算F1分数
        """
        eps = 1e-7  # 防止分母为零
        labels_onehot = one_hot(labels, num_classes=num_classes).float()
        preds_onehot = one_hot(preds, num_classes=num_classes).float()
    
        tp = torch.sum(labels_onehot * preds_onehot, dim=0)  # 真正例
        fp = torch.sum(preds_onehot * (1 - labels_onehot), dim=0)  # 假正例
        fn = torch.sum(labels_onehot * (1 - preds_onehot), dim=0)  # 假负例
    
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
    
        return f1.mean().item()  # 返回平均F1分数

    def train(self, epochs=25, lr=None, freeze=False,num_class=5, log_dir="logs",sub=-1):
        # 获取当前时间并格式化为“年月日_小时分钟”
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        log_filename = f"train_log_sub{sub}_{current_time}.txt"
        log_path = os.path.join(log_dir, log_filename)
    
        # 创建日志文件所在的文件夹（如果不存在）
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 打开日志文件，准备写入
        with open(log_path, "w") as log_file:
            lr = lr if lr is not None else self.initial_lr
            if lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
    
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
            # Freeze or unfreeze model parameters based on the freeze flag
            for param in self.model.parameters():
                param.requires_grad = not freeze
    
            # Wrap the model with DataParallel
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                log_file.write(f"GPU: {torch.cuda.device_count()}\n")
    
            ###
            best_val_accuracy = 0.0
            best_epoch = 0
            best_val_f1 = 0.0
            best_model_state = None
            ###
    
            for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
                # Variables to store performance metrics
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                noise_std = 0.0
    
                # Training phase
                self.model.train()
                for inputs, labels in self.train_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    noise = torch.randn_like(inputs) * noise_std  # noise_std is the standard deviation of the noise
                    noisy_inputs = inputs + noise
                    self.optimizer.zero_grad()
                    outputs = self.model(noisy_inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
    
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    label = torch.argmax(labels, dim=1)
                    correct_predictions += (predicted == label).sum().item()
    
                train_loss = running_loss / len(self.train_dataloader.dataset)
                train_accuracy = correct_predictions / total_predictions
    
                # Validation phase
                self.model.eval()
                running_val_loss = 0.0
                val_correct_predictions = 0
                val_total_predictions = 0
                val_labels = []  # 在每个epoch的验证阶段初始化
                val_preds = []   # 在每个epoch的验证阶段初始化
                with torch.no_grad():
                    for inputs, labels in self.test_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        val_loss = self.criterion(outputs, labels)
                        running_val_loss += val_loss.item() * inputs.size(0)
    
                        _, predicted = torch.max(outputs.data, 1)
                        val_total_predictions += labels.size(0)
                        label = torch.argmax(labels, dim=1)
                        val_correct_predictions += (predicted == label).sum().item()
    
                        val_labels.append(label.cpu())  # 收集真实标签
                        val_preds.append(predicted.cpu())  # 收集预测标签
    
                val_loss = running_val_loss / len(self.test_dataloader.dataset)
                val_accuracy = val_correct_predictions / val_total_predictions
    
                # 将列表中的张量拼接成一个大张量
                val_labels = torch.cat(val_labels, dim=0)
                val_preds = torch.cat(val_preds, dim=0)
    
                # 计算F1分数（使用PyTorch实现）
                val_f1 = self.calculate_f1(val_labels, val_preds, num_classes=num_class)
    
                if val_accuracy > best_val_accuracy:
                    # 如果当前验证集准确率更高，更新最佳模型状态
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    best_val_f1 = val_f1
                    best_model_state = self.model.state_dict()
                elif val_accuracy == best_val_accuracy:
                    # 如果准确率相同，但 F1 分数更高，也更新最佳模型状态
                    if val_f1 > best_val_f1:
                        best_val_accuracy = val_accuracy
                        best_epoch = epoch + 1
                        best_val_f1 = val_f1
                        best_model_state = self.model.state_dict()
    
                log_content = (
                    f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                    f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}, Test F1 Score: {val_f1:.4f}\n'
                )
                #print(log_content, end="")
                log_file.write(log_content)
    
            final_log_content = (
                f"Best Test Accuracy: {best_val_accuracy:.4f} at Epoch {best_epoch}/{epochs}, with F1 Score: {best_val_f1:.4f}\n"
            )
            print(final_log_content, end="")
            log_file.write(final_log_content)
    
        self.test_acc = val_accuracy
        # Load the best model state for testing
        self.model.load_state_dict(best_model_state)
        print("Loaded the best model for testing.")
        return self.model
    
def convert_hot_encoding(label):
    """
    将长度为batch_size的10分类独热编码转换为5分类独热编码。
    每个样本的10分类独热编码按顺序两两一组重新组合。
    同一组中的两个只要有一个是1，则5分类中对应的组为1。
    
    参数:
        label (numpy.ndarray): 形状为 (batch_size, 10) 的独热编码数组。
    
    返回:
        numpy.ndarray: 形状为 (batch_size, 5) 的5分类独热编码数组。
    """
    # 检查输入是否为二维数组且第二维长度为10
    if not isinstance(label, np.ndarray) or label.ndim != 2 or label.shape[1] != 10:
        raise ValueError("输入的独热编码必须是二维数组，且第二维长度为10。")
    
    batch_size = label.shape[0]
    new_label = np.zeros((batch_size, 5), dtype=label.dtype)
    
    # 对每个样本进行转换
    for i in range(batch_size):
        for j in range(5):
            # 检查每组中是否有1
            if label[i, 2*j] == 1 or label[i, 2*j+1] == 1:
                new_label[i, j] = 1
    
    return new_label
def resample_data(data, original_time_points, new_time_points):
    """
    使用 numpy 的插值方法对数据进行重采样，适用于时间维度在第二个维度的情况。

    参数:
        data (numpy.ndarray): 输入数据，形状为 [num_samples, original_time_points, channels]。
        original_time_points (int): 原始数据的时间点数量。
        new_time_points (int): 重采样后的时间点数量。

    返回:
        numpy.ndarray: 重采样后的数据，形状为 [num_samples, new_time_points, channels]。
    """
    num_samples, _, channels = data.shape

    # 创建原始时间序列的索引
    original_indices = np.linspace(0, original_time_points - 1, original_time_points)
    # 创建新的时间序列的索引
    new_indices = np.linspace(0, original_time_points - 1, new_time_points)

    # 初始化重采样后的数据数组
    resampled_data = np.zeros((num_samples, new_time_points, channels))

    # 对每个样本和每个通道进行重采样
    for i in range(num_samples):
        for j in range(channels):
            # 提取单个通道的时间序列
            channel_data = data[i, :, j]  # 时间维度在第二个维度
            # 检查原始数据和索引的长度是否一致
            if len(channel_data) != original_time_points:
                raise ValueError(f"通道数据长度 {len(channel_data)} 与原始时间点数量 {original_time_points} 不一致。")
            # 使用 numpy 的插值方法进行重采样
            resampled_channel_data = np.interp(new_indices, original_indices, channel_data)
            # 将重采样后的数据存回数组
            resampled_data[i, :, j] = resampled_channel_data

    return resampled_data
def filter_odd_even(data, labels, invert=False):
    """
    沿第一个维度（len）筛选数据和对应的标签，保留奇数或偶数索引的数据。
    
    参数:
        data (numpy.ndarray): 输入数据，形状为 (len, time, channel)
        labels (numpy.ndarray): 输入标签，形状为 (len, label)
        invert (bool): 是否取反。如果为 False，则保留奇数索引的数据；如果为 True，则保留偶数索引的数据。
    
    返回:
        tuple: (filtered_data, filtered_labels)，处理后的数据和标签
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("输入数据必须是形状为 (len, time, channel) 的三维 numpy 数组")
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ValueError("输入标签必须是形状为 (len, label) 的二维 numpy 数组")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("数据和标签的第一个维度长度必须一致")
    
    # 根据 invert 参数选择保留奇数或偶数索引
    if invert:
        print("active mode")
        # 保留偶数索引的数据和标签
        filtered_data = data[1::2, :, :]
        filtered_labels = labels[1::2, :]
    else:
        print("passive mode")
        # 保留奇数索引的数据和标签
        filtered_data = data[::2, :, :]
        filtered_labels = labels[::2, :]
    
    return filtered_data, filtered_labels

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int, help="Subject number")
parser.add_argument('--n_ses', default=1, type=int, help="Number of sessions")
parser.add_argument('--sfreq', default=250, type=int, help="Resampling frequency")
parser.add_argument("--preprocessed_max_subject", type=int, default=1, help="Maximum subject number for preprocessed data check, default is 42")
args = parser.parse_args()

if __name__ == "__main__":
    # Toy dataste
    '''
    data = torch.randn(1000, 30, 500)
    labels = torch.randint(0, 5, (1000,))
    '''

    random_split_seed=100
    cut_len=300 #eav 500 seed 1500 faced 500
    train_deep=300 #eav 500 seed 300 faced 500
    num_class=3 #eav5 seed3 faced9
    eeg_channel=62 #eav 30 seed 62 faced 32
    batch_size = 80 #available eav 128 seed 128
    shift_step = 25
    '''
    data_path='/root/autodl-tmp/EAV/perprocess'#/False
    dataset_path='/root/autodl-tmp/EAV/ori'
    data_path='/root/autodl-tmp/FACED/perprocess'
    dataset_path='/root/autodl-tmp/FACED'
    '''
    data_path='/root/autodl-tmp/SEED/perprocess'
    dataset_path='/root/autodl-tmp/SEED'
    data_list,label_list=data_prepare.load_data_from_file(args,dataset_path,data_path,data_path)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', default=1, type=int, help="Subject number")
    parser.add_argument('--n_ses', default=1, type=int, help="Number of sessions")
    parser.add_argument('--sfreq', default=250, type=int, help="Resampling frequency")
    parser.add_argument("--preprocessed_max_subject", type=int, default=45, help="Maximum subject number for preprocessed data check, eav max 42,faced max 122,seed max 15")
    args = parser.parse_args()

    data_list,label_list=data_prepare.load_data_from_file(args,dataset_path,data_path,data_path)
    '''
    dataset = TensorDataset(data, labels)

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    '''
    if len(data_list)==len(label_list):
        start=0
        fin=len(data_list)
        print(f"\n total: {fin-start}")
        for i in range(start,fin):
        #for i in range(0,10):
            print(f"\n subject {i+1}")
            
            subject_data=np.transpose(data_list[i], (2, 0, 1))
            subject_label=np.transpose(label_list[i], (1, 0))

            #subject_data,subject_label=filter_odd_even(subject_data,subject_label,True)#false 135 true 024 only eav

            subject_data_extend,subject_label_extend=data_prepare.cut_and_extend_data(subject_data,subject_label,cut_len)
            #subject_data_extend,subject_label_extend=data_prepare.cut_and_extend_data_new(subject_data,subject_label,cut_len,shift_step)
            
            #subject_data=subject_data_extend[:,:train_deep,:]
            subject_data=resample_data(subject_data_extend,cut_len,train_deep)
            subject_label=subject_label_extend
            
            #subject_label=convert_hot_encoding(subject_label)#only eav
            subject_data=np.transpose(subject_data, (0, 2, 1))
            '''
            subject_label=np.transpose(subject_label, (0, 1))
            subject_label = np.expand_dims(subject_label, axis=0)
            subject_label = subject_label[:0]
            '''
            # 假设 subject_data 是特征数据，subject_label 是标签数据
            # subject_label 必须是整数类型
            subject_label = subject_label.astype(int)
            
            # 初始化 StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            
            # 获取分层划分的索引
            for train_index, test_index in sss.split(subject_data, subject_label):
                train_data, test_data = subject_data[train_index], subject_data[test_index]
                train_labels, test_labels = subject_label[train_index], subject_label[test_index]
            
            # 将划分后的数据转换为 TensorDataset
            train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
            test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

            '''
            dataset = TensorDataset(torch.from_numpy(subject_data).float(), torch.from_numpy(subject_label).float())

            train_size = int(0.7 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            '''

            # 获取训练集和测试集的标签
            train_labels = [torch.argmax(train_dataset[i][1]).item() for i in range(len(train_dataset))]
            test_labels = [torch.argmax(test_dataset[i][1]).item() for i in range(len(test_dataset))]
            
            # 统计标签分布
            train_label_distribution = Counter(train_labels)
            test_label_distribution = Counter(test_labels)
            
            # 打印标签分布
            print("Train Dataset Label Distribution:")
            for label, count in sorted(train_label_distribution.items()):
                print(f"Label {label}: {count} samples ({count / len(train_labels) * 100:.2f}%)", end="   ")
            
            print("\n\nTest Dataset Label Distribution:")
            for label, count in sorted(test_label_distribution.items()):
                print(f"Label {label}: {count} samples ({count / len(test_labels) * 100:.2f}%)", end="   ")
            '''
            train_data,train_label,test_data,test_label=data_prepare.split_dataset(subject_data,subject_label,0.2,random_split_seed)#len,time,channel
            train_data=np.transpose(train_data, (0, 2, 1))
            test_data=np.transpose(test_data, (0, 2, 1))#len,channels,time
            train_label=np.transpose(train_label, (0, 1))
            #train_label = np.expand_dims(train_label, axis=0)
            #train_label = train_label[:0]
            test_label=np.transpose(test_label, (0, 1))
            #test_label = np.expand_dims(test_label, axis=0)
            #test_label = test_label[:0]
            
            #dataset_train=data_prepare.Dataset_prepare(train_data,train_label)
            #dataset_test=data_prepare.Dataset_prepare(test_data,test_label)
            dataset_train=TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
            dataset_test=TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
            '''

            # Create DataLoaders for training and validation sets
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
            model = EEGClassificationModel(eeg_channel=eeg_channel,timepoint=train_deep,num_class=num_class)  # Example: 64 EEG channels
            trainer = EEGModelTrainer(train_dataloader, test_dataloader,model)
            trainer.train(epochs=150,num_class=num_class,log_dir="/root/autodl-tmp/code/new/report",sub=i+1)


