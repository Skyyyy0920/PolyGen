#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件编码器

基于PolyGen-F06C的ConditionEncoder，适配双生成器需求
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConditionEncoder(nn.Module):
    """
    条件编码器
    
    将聚合物条件特征编码为潜在表示，用于指导残差生成器
    """
    
    def __init__(self,
                 in_dim: int = 17,
                 d_model: int = 128,
                 proj_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 temperature: float = 0.10):
        """
        初始化条件编码器
        
        Args:
            in_dim: 输入特征维度
            d_model: 模型隐藏维度
            proj_dim: 投影维度
            num_layers: Transformer层数
            dropout: Dropout率
            temperature: 对比学习温度参数
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影头
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, proj_dim)
        )
        
        # 条件嵌入输出
        self.condition_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, cond: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            cond: 条件特征 [batch_size, in_dim]
            return_features: 是否返回中间特征
            
        Returns:
            包含各种嵌入的字典
        """
        batch_size = cond.size(0)
        
        # 输入投影
        x = self.input_projection(cond)  # [B, d_model]
        
        # 添加序列维度用于Transformer
        x = x.unsqueeze(1)  # [B, 1, d_model]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [B, 1, d_model]
        encoded = encoded.squeeze(1)  # [B, d_model]
        
        # 生成不同的嵌入
        proj_emb = self.projection_head(encoded)  # [B, proj_dim] - 用于对比学习
        cond_emb = self.condition_head(encoded)   # [B, d_model] - 用于条件生成
        
        # L2归一化投影嵌入
        proj_emb = F.normalize(proj_emb, p=2, dim=1)
        
        results = {
            'proj_emb': proj_emb,      # 对比学习嵌入
            'cond_emb': cond_emb,      # 条件生成嵌入
        }
        
        if return_features:
            results.update({
                'input_features': cond,
                'projected_features': x.squeeze(1),
                'encoded_features': encoded
            })
        
        return results
    
    def compute_contrastive_loss(self, 
                                proj_emb1: torch.Tensor, 
                                proj_emb2: torch.Tensor,
                                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            proj_emb1: 第一组投影嵌入 [B, proj_dim]
            proj_emb2: 第二组投影嵌入 [B, proj_dim]
            labels: 可选的标签，用于监督对比学习
            
        Returns:
            对比损失
        """
        batch_size = proj_emb1.size(0)
        device = proj_emb1.device
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(proj_emb1, proj_emb2.T) / self.temperature
        
        if labels is None:
            # 无监督对比学习：对角线为正样本
            labels = torch.arange(batch_size, device=device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def get_embedding_dim(self) -> int:
        """获取条件嵌入维度"""
        return self.d_model


class SequenceEncoder(nn.Module):
    """
    序列编码器
    
    专门用于编码聚合物序列，生成序列级别的表示
    """
    
    def __init__(self,
                 vocab_size: int = 3,  # 0: pad, 1: A, 2: B
                 d_model: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 max_length: int = 1000,
                 dropout: float = 0.1):
        """
        初始化序列编码器
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            max_length: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(1, max_length, d_model) * 0.02
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 池化层
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        
    def tokenize_sequences(self, sequences_batch: list) -> torch.Tensor:
        """
        将序列批次转换为token张量
        
        Args:
            sequences_batch: 序列批次，每个元素是序列列表
            
        Returns:
            Token张量 [batch_size, max_seq_len]
        """
        batch_tokens = []
        
        for sequences in sequences_batch:
            # 合并序列
            combined_seq = ''.join(sequences)[:self.max_length]
            
            # 转换为token
            tokens = []
            for char in combined_seq:
                if char == 'A':
                    tokens.append(1)
                elif char == 'B':
                    tokens.append(2)
                else:
                    tokens.append(0)  # padding
            
            # 填充到最大长度
            while len(tokens) < self.max_length:
                tokens.append(0)
            
            batch_tokens.append(tokens)
        
        return torch.tensor(batch_tokens, dtype=torch.long)
    
    def forward(self, sequences_batch: list) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            sequences_batch: 序列批次
            
        Returns:
            编码结果字典
        """
        # Token化
        tokens = self.tokenize_sequences(sequences_batch)  # [B, L]
        device = next(self.parameters()).device
        tokens = tokens.to(device)
        
        batch_size, seq_len = tokens.shape
        
        # Token嵌入
        x = self.token_embedding(tokens)  # [B, L, d_model]
        
        # 位置编码
        if seq_len <= self.max_length:
            pos_enc = self.position_encoding[:, :seq_len, :]
        else:
            # 处理超长序列
            pos_enc = self.position_encoding.repeat(1, (seq_len // self.max_length) + 1, 1)[:, :seq_len, :]
        
        x = x + pos_enc
        x = self.dropout(x)
        
        # 创建padding mask
        padding_mask = (tokens == 0)  # [B, L]
        
        # Transformer编码
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)  # [B, L, d_model]
        
        # 池化得到序列级表示
        # 只对非padding位置进行池化
        mask = (~padding_mask).float().unsqueeze(-1)  # [B, L, 1]
        masked_encoded = encoded * mask
        
        # 平均池化
        seq_lengths = mask.sum(dim=1, keepdim=True)  # [B, 1, 1]
        seq_repr = masked_encoded.sum(dim=1) / (seq_lengths.squeeze(-1) + 1e-8)  # [B, d_model]
        
        return {
            'sequence_embedding': seq_repr,
            'token_embeddings': encoded,
            'attention_mask': ~padding_mask
        }


def test_condition_encoder():
    """测试条件编码器"""
    print("🧪 测试ConditionEncoder...")
    
    # 测试参数
    batch_size = 4
    in_dim = 17
    
    # 创建编码器
    encoder = ConditionEncoder(
        in_dim=in_dim,
        d_model=128,
        proj_dim=256,
        num_layers=3
    )
    
    # 生成测试数据
    cond_features = torch.randn(batch_size, in_dim)
    
    # 前向传播
    results = encoder(cond_features, return_features=True)
    
    print(f"输入形状: {cond_features.shape}")
    print(f"投影嵌入形状: {results['proj_emb'].shape}")
    print(f"条件嵌入形状: {results['cond_emb'].shape}")
    
    # 测试对比学习损失
    proj_emb1 = results['proj_emb']
    proj_emb2 = torch.randn_like(proj_emb1)
    proj_emb2 = F.normalize(proj_emb2, p=2, dim=1)
    
    contrastive_loss = encoder.compute_contrastive_loss(proj_emb1, proj_emb2)
    print(f"对比损失: {contrastive_loss.item():.4f}")
    
    print("✅ ConditionEncoder测试通过!")
    
    # 测试序列编码器
    print("\n🧪 测试SequenceEncoder...")
    
    seq_encoder = SequenceEncoder(d_model=64, num_layers=2)
    
    # 测试序列
    sequences_batch = [
        ['AAABBB', 'BBAABB'],
        ['ABABAB', 'BABABA'],
        ['AAAAAA', 'BBBBBB'],
        ['ABABABAB']
    ]
    
    seq_results = seq_encoder(sequences_batch)
    
    print(f"序列嵌入形状: {seq_results['sequence_embedding'].shape}")
    print(f"Token嵌入形状: {seq_results['token_embeddings'].shape}")
    
    print("✅ SequenceEncoder测试通过!")


if __name__ == "__main__":
    test_condition_encoder()
