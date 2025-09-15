#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
残差生成器

基于DiT-1D架构，专门学习实际分布与Mayo-Lewis理论分布的偏差
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class TimestepEmbedder(nn.Module):
    """时间步嵌入器"""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation层"""
    
    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.scale_shift_table = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_size * 2, bias=True)
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_size]
            cond: 条件特征 [batch_size, cond_dim]
        """
        scale_shift = self.scale_shift_table(cond)  # [B, hidden_size * 2]
        scale, shift = scale_shift.chunk(2, dim=1)  # [B, hidden_size] each
        
        # 添加序列维度
        scale = scale.unsqueeze(1)  # [B, 1, hidden_size]
        shift = shift.unsqueeze(1)  # [B, 1, hidden_size]
        
        return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block
    
    结合自注意力和条件调制的Transformer块
    """
    
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 cond_dim: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 film_each_layer: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.film_each_layer = film_each_layer
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 自注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # FiLM调制层
        if film_each_layer:
            self.film1 = FiLMLayer(cond_dim, hidden_size)
            self.film2 = FiLMLayer(cond_dim, hidden_size)
        else:
            self.film1 = None
            self.film2 = None
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_size]
            cond: 条件特征 [batch_size, cond_dim]
        """
        # 自注意力分支
        if self.film1 is not None:
            norm1_out = self.film1(self.norm1(x), cond)
        else:
            norm1_out = self.norm1(x)
        
        attn_out, _ = self.attn(norm1_out, norm1_out, norm1_out)
        x = x + attn_out
        
        # MLP分支
        if self.film2 is not None:
            norm2_out = self.film2(self.norm2(x), cond)
        else:
            norm2_out = self.norm2(x)
        
        mlp_out = self.mlp(norm2_out)
        x = x + mlp_out
        
        return x


class ResidualGenerator(nn.Module):
    """
    残差生成器
    
    基于DiT-1D架构，专门学习实际分布与Mayo-Lewis理论分布的偏差
    使用扩散模型框架进行训练和推理
    """
    
    def __init__(self,
                 bins: int = 50,
                 cond_dim: int = 128,
                 hidden_size: int = 256,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 film_each_layer: bool = True,
                 learn_sigma: bool = False):
        """
        初始化残差生成器
        
        Args:
            bins: 分布bins数量
            cond_dim: 条件特征维度（包含理论分布）
            hidden_size: 隐藏层大小
            num_layers: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            dropout: Dropout率
            film_each_layer: 是否每层都使用FiLM
            learn_sigma: 是否学习方差
        """
        super().__init__()
        
        self.bins = bins
        self.cond_dim = cond_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learn_sigma = learn_sigma
        
        # 时间步嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 输入投影（将每个bin映射到hidden_size维度）
        self.x_embedder = nn.Linear(1, hidden_size)  # 每个bin值单独嵌入
        
        # 位置编码（对于1D分布）
        self.pos_embed = nn.Parameter(torch.zeros(1, bins, hidden_size))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                cond_dim=cond_dim + hidden_size,  # 条件 + 时间嵌入
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                film_each_layer=film_each_layer
            )
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_layer = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 输出投影：从[B, bins, hidden_size]到[B, bins, 1]或[B, bins, 2]
        output_channels = 2 if learn_sigma else 1
        self.linear = nn.Linear(hidden_size, output_channels, bias=True)
        
        # 初始化权重
        self.initialize_weights()
    
    def initialize_weights(self):
        """初始化模型权重"""
        # 初始化线性层
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 零初始化最终输出层（重要！）
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                cond: torch.Tensor,
                theoretical_dist: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 噪声输入 [batch_size, bins]
            t: 时间步 [batch_size]
            cond: 条件特征 [batch_size, cond_dim - bins]
            theoretical_dist: 理论分布 [batch_size, bins]
            
        Returns:
            预测的残差或v参数 [batch_size, bins] 或 [batch_size, 2*bins]
        """
        # 时间步嵌入
        t_emb = self.t_embedder(t)  # [B, hidden_size]
        
        # 组合条件（原条件 + 理论分布）
        combined_cond = torch.cat([cond, theoretical_dist], dim=1)  # [B, cond_dim]
        
        # 组合条件和时间嵌入
        full_cond = torch.cat([combined_cond, t_emb], dim=1)  # [B, cond_dim + hidden_size]
        
        # 输入嵌入：将[B, bins]转换为[B, bins, hidden_size]
        x = x.unsqueeze(-1)  # [B, bins, 1]
        x = self.x_embedder(x)  # [B, bins, hidden_size]
        
        # 添加位置编码
        x = x + self.pos_embed  # [B, bins, hidden_size]
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x, full_cond)
        
        # 最终层归一化
        x = self.final_layer(x)
        
        # 输出投影
        x = self.linear(x)  # [B, bins, output_channels]
        
        if self.learn_sigma:
            # 分离均值和方差：[B, bins, 2] -> [B, bins], [B, bins]
            mean, logvar = x.chunk(2, dim=-1)  # [B, bins, 1] each
            return mean.squeeze(-1), logvar.squeeze(-1)  # [B, bins] each
        else:
            return x.squeeze(-1)  # [B, bins]
    
    def get_condition_dim(self) -> int:
        """获取完整条件维度（不包括理论分布）"""
        return self.cond_dim - self.bins


class NoiseSchedule:
    """扩散噪声调度器"""
    
    def __init__(self, 
                 T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = 'cosine'):
        """
        初始化噪声调度器
        
        Args:
            T: 扩散步数
            beta_start: 起始beta值
            beta_end: 结束beta值
            schedule_type: 调度类型 ('linear' or 'cosine')
        """
        self.T = T
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, T)
        elif schedule_type == 'cosine':
            # 余弦调度
            steps = T + 1
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # v-参数化所需的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def to(self, device):
        """移动到指定设备"""
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                     'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod']:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def q_sample_vparam(x_start: torch.Tensor, 
                   t: torch.Tensor, 
                   schedule: NoiseSchedule,
                   noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    v-参数化的前向扩散过程
    
    Args:
        x_start: 原始数据 [B, ...]
        t: 时间步 [B]
        schedule: 噪声调度器
        noise: 可选的噪声张量
        
    Returns:
        x_t: 加噪后的数据
        v_target: v参数化的目标
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    device = x_start.device
    schedule = schedule.to(device)

    # 获取时间步对应的系数
    sqrt_alphas_cumprod_t = schedule.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
    
    # 前向过程
    x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # v-参数化目标
    v_target = sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
    
    return x_t, v_target


def test_residual_generator():
    """测试残差生成器"""
    print("🧪 测试ResidualGenerator...")
    
    # 测试参数
    batch_size = 4
    bins = 30
    cond_dim = 64  # 不包括理论分布
    
    # 创建模型
    model = ResidualGenerator(
        bins=bins,
        cond_dim=cond_dim + bins,  # 包括理论分布
        hidden_size=128,
        num_layers=4,
        num_heads=8
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    x = torch.randn(batch_size, bins)
    t = torch.randint(0, 1000, (batch_size,))
    cond = torch.randn(batch_size, cond_dim)
    theoretical_dist = torch.softmax(torch.randn(batch_size, bins), dim=1)
    
    # 前向传播
    output = model(x, t, cond, theoretical_dist)
    
    print(f"输入形状: {x.shape}")
    print(f"时间步形状: {t.shape}")
    print(f"条件形状: {cond.shape}")
    print(f"理论分布形状: {theoretical_dist.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试噪声调度器
    print("\n🧪 测试NoiseSchedule...")
    
    schedule = NoiseSchedule(T=100, schedule_type='cosine')
    
    # 测试v-参数化采样
    x_start = torch.softmax(torch.randn(batch_size, bins), dim=1)
    t_sample = torch.randint(0, 100, (batch_size,))
    
    x_t, v_target = q_sample_vparam(x_start, t_sample, schedule)
    
    print(f"原始数据形状: {x_start.shape}")
    print(f"加噪数据形状: {x_t.shape}")
    print(f"v目标形状: {v_target.shape}")
    print(f"原始数据求和: {torch.sum(x_start, dim=1)[:3]}")
    print(f"加噪数据求和: {torch.sum(x_t, dim=1)[:3]}")
    
    # 测试损失计算
    v_pred = model(x_t, t_sample, cond, theoretical_dist)
    loss = F.mse_loss(v_pred, v_target)
    
    print(f"v预测形状: {v_pred.shape}")
    print(f"MSE损失: {loss.item():.6f}")
    
    print("✅ ResidualGenerator测试通过!")


if __name__ == "__main__":
    test_residual_generator()
