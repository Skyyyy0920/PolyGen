#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应融合模块

动态调整Mayo-Lewis理论分布与残差修正分布的权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class AdaptiveFusionModule(nn.Module):
    """
    自适应融合模块
    
    根据条件特征和理论置信度，动态调整理论分布与残差修正分布的权重
    """
    
    def __init__(self,
                 cond_dim: int,
                 mayo_param_dim: int = 6,  # f_A, f_B, p_AA, p_BB, p_AB, p_BA
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        初始化自适应融合模块
        
        Args:
            cond_dim: 条件特征维度
            mayo_param_dim: Mayo-Lewis参数维度
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.cond_dim = cond_dim
        self.mayo_param_dim = mayo_param_dim
        self.hidden_dim = hidden_dim
        
        # 置信度预测网络
        confidence_layers = []
        input_dim = cond_dim + mayo_param_dim
        
        for i in range(num_layers):
            if i == 0:
                confidence_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            elif i == num_layers - 1:
                confidence_layers.append(nn.Linear(hidden_dim, 1))
            else:
                confidence_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
        
        self.confidence_predictor = nn.Sequential(*confidence_layers)
        
        # 分布质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(cond_dim + mayo_param_dim + 2,  # +2 for distribution statistics
            hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 权重调制网络
        self.weight_modulator = nn.Sequential(
            nn.Linear(cond_dim + mayo_param_dim + 1,  # +1 for confidence
            hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 输出3个权重：theoretical, residual, combined
            nn.Softmax(dim=-1)
        )
    
    def extract_mayo_lewis_features(self, mayo_params_batch: list) -> torch.Tensor:
        """
        从Mayo-Lewis参数字典中提取特征张量
        
        Args:
            mayo_params_batch: Mayo-Lewis参数字典列表
            
        Returns:
            Mayo-Lewis特征张量 [batch_size, mayo_param_dim]
        """
        features = []
        
        for params in mayo_params_batch:
            param_vector = [
                params.get('f_A', 0.5),
                params.get('f_B', 0.5),
                params.get('p_AA', 0.3),
                params.get('p_BB', 0.3),
                params.get('p_AB', 0.2),
                params.get('p_BA', 0.2)
            ]
            features.append(param_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_distribution_statistics(self, 
                                      theoretical_dist: torch.Tensor,
                                      residual_dist: torch.Tensor) -> torch.Tensor:
        """
        计算分布统计量
        
        Args:
            theoretical_dist: 理论分布 [batch_size, bins]
            residual_dist: 残差修正分布 [batch_size, bins]
            
        Returns:
            分布统计量 [batch_size, 2]
        """
        # 计算分布的熵
        theoretical_entropy = -torch.sum(
            theoretical_dist * torch.log(theoretical_dist + 1e-8), dim=1
        )
        residual_entropy = -torch.sum(
            residual_dist * torch.log(residual_dist + 1e-8), dim=1
        )
        
        return torch.stack([theoretical_entropy, residual_entropy], dim=1)
    
    def predict_confidence(self, 
                          cond: torch.Tensor,
                          mayo_params_batch: list) -> torch.Tensor:
        """
        预测Mayo-Lewis理论的置信度
        
        Args:
            cond: 条件特征 [batch_size, cond_dim]
            mayo_params_batch: Mayo-Lewis参数字典列表
            
        Returns:
            置信度 [batch_size]
        """
        device = cond.device
        
        # 提取Mayo-Lewis特征
        mayo_features = self.extract_mayo_lewis_features(mayo_params_batch).to(device)
        
        # 组合特征
        combined_features = torch.cat([cond, mayo_features], dim=1)
        
        # 预测置信度
        confidence_logits = self.confidence_predictor(combined_features)
        confidence = torch.sigmoid(confidence_logits.squeeze(-1))
        
        return confidence
    
    def assess_distribution_quality(self,
                                  cond: torch.Tensor,
                                  mayo_params_batch: list,
                                  theoretical_dist: torch.Tensor,
                                  residual_dist: torch.Tensor) -> torch.Tensor:
        """
        评估分布质量
        
        Args:
            cond: 条件特征
            mayo_params_batch: Mayo-Lewis参数
            theoretical_dist: 理论分布
            residual_dist: 残差修正分布
            
        Returns:
            质量评分 [batch_size]
        """
        device = cond.device
        
        # 提取特征
        mayo_features = self.extract_mayo_lewis_features(mayo_params_batch).to(device)
        dist_stats = self.compute_distribution_statistics(theoretical_dist, residual_dist)
        
        # 组合特征
        combined_features = torch.cat([cond, mayo_features, dist_stats], dim=1)
        
        # 评估质量
        quality_score = self.quality_assessor(combined_features).squeeze(-1)
        
        return quality_score
    
    def compute_adaptive_weights(self,
                               cond: torch.Tensor,
                               mayo_params_batch: list,
                               confidence: torch.Tensor) -> torch.Tensor:
        """
        计算自适应权重
        
        Args:
            cond: 条件特征
            mayo_params_batch: Mayo-Lewis参数
            confidence: 置信度
            
        Returns:
            权重 [batch_size, 3] - [theoretical_weight, residual_weight, combined_weight]
        """
        device = cond.device
        
        # 提取Mayo-Lewis特征
        mayo_features = self.extract_mayo_lewis_features(mayo_params_batch).to(device)
        
        # 组合特征
        combined_features = torch.cat([cond, mayo_features, confidence.unsqueeze(-1)], dim=1)
        
        # 计算权重
        weights = self.weight_modulator(combined_features)
        
        return weights
    
    def forward(self,
                theoretical_dist: torch.Tensor,
                residual_corrected_dist: torch.Tensor,
                cond: torch.Tensor,
                mayo_params_batch: list,
                fusion_strategy: str = 'adaptive') -> Dict[str, torch.Tensor]:
        """
        前向传播 - 自适应融合分布
        
        Args:
            theoretical_dist: Mayo-Lewis理论分布 [batch_size, bins]
            residual_corrected_dist: 残差修正分布 [batch_size, bins]
            cond: 条件特征 [batch_size, cond_dim]
            mayo_params_batch: Mayo-Lewis参数字典列表
            fusion_strategy: 融合策略 ('adaptive', 'confidence', 'weighted')
            
        Returns:
            融合结果字典
        """
        # 1. 预测置信度
        confidence = self.predict_confidence(cond, mayo_params_batch)
        
        # 2. 评估分布质量
        quality_score = self.assess_distribution_quality(
            cond, mayo_params_batch, theoretical_dist, residual_corrected_dist
        )
        
        # 3. 根据策略进行融合
        if fusion_strategy == 'confidence':
            # 简单的置信度加权
            confidence_weight = confidence.unsqueeze(-1)  # [B, 1]
            fused_dist = (confidence_weight * theoretical_dist + 
                         (1 - confidence_weight) * residual_corrected_dist)
            
        elif fusion_strategy == 'weighted':
            # 基于质量的加权
            quality_weight = quality_score.unsqueeze(-1)  # [B, 1]
            fused_dist = (quality_weight * theoretical_dist + 
                         (1 - quality_weight) * residual_corrected_dist)
            
        elif fusion_strategy == 'adaptive':
            # 自适应权重融合
            weights = self.compute_adaptive_weights(cond, mayo_params_batch, confidence)
            
            # 三种分布的加权组合
            w_theo, w_resid, w_comb = weights.unbind(-1)  # [B] each
            
            # 创建组合分布（理论和残差的平均）
            combined_dist = 0.5 * theoretical_dist + 0.5 * residual_corrected_dist
            
            # 最终融合
            fused_dist = (w_theo.unsqueeze(-1) * theoretical_dist +
                         w_resid.unsqueeze(-1) * residual_corrected_dist +
                         w_comb.unsqueeze(-1) * combined_dist)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # 4. 重新归一化
        fused_dist = fused_dist / (torch.sum(fused_dist, dim=1, keepdim=True) + 1e-8)
        
        return {
            'fused_distribution': fused_dist,
            'confidence': confidence,
            'quality_score': quality_score,
            'theoretical_dist': theoretical_dist,
            'residual_corrected_dist': residual_corrected_dist,
            'fusion_weights': weights if fusion_strategy == 'adaptive' else None
        }
    
    def compute_fusion_loss(self,
                          fused_dist: torch.Tensor,
                          target_dist: torch.Tensor,
                          confidence: torch.Tensor,
                          quality_score: torch.Tensor,
                          alpha: float = 0.1,
                          beta: float = 0.05) -> Dict[str, torch.Tensor]:
        """
        计算融合损失
        
        Args:
            fused_dist: 融合分布
            target_dist: 目标分布
            confidence: 置信度
            quality_score: 质量评分
            alpha: 置信度正则化权重
            beta: 质量正则化权重
            
        Returns:
            损失字典
        """
        # 主要重建损失
        recon_loss = F.kl_div(
            torch.log(fused_dist + 1e-8), 
            target_dist, 
            reduction='batchmean'
        )
        
        # 置信度正则化（鼓励适度置信）
        confidence_reg = alpha * torch.mean((confidence - 0.5) ** 2)
        
        # 质量正则化（鼓励高质量预测）
        quality_reg = beta * torch.mean((1 - quality_score) ** 2)
        
        # 总损失
        total_loss = recon_loss + confidence_reg + quality_reg
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'confidence_regularization': confidence_reg,
            'quality_regularization': quality_reg
        }


def test_fusion_module():
    """测试自适应融合模块"""
    print("🧪 测试AdaptiveFusionModule...")
    
    # 测试参数
    batch_size = 4
    bins = 30
    cond_dim = 64
    
    # 创建模块
    fusion_module = AdaptiveFusionModule(
        cond_dim=cond_dim,
        mayo_param_dim=6,
        hidden_dim=128,
        num_layers=3
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in fusion_module.parameters()):,}")
    
    # 生成测试数据
    theoretical_dist = torch.softmax(torch.randn(batch_size, bins), dim=1)
    residual_corrected_dist = torch.softmax(torch.randn(batch_size, bins), dim=1)
    cond = torch.randn(batch_size, cond_dim)
    
    # 模拟Mayo-Lewis参数
    mayo_params_batch = [
        {'f_A': 0.4, 'f_B': 0.6, 'p_AA': 0.3, 'p_BB': 0.4, 'p_AB': 0.15, 'p_BA': 0.15},
        {'f_A': 0.5, 'f_B': 0.5, 'p_AA': 0.35, 'p_BB': 0.35, 'p_AB': 0.15, 'p_BA': 0.15},
        {'f_A': 0.6, 'f_B': 0.4, 'p_AA': 0.4, 'p_BB': 0.3, 'p_AB': 0.15, 'p_BA': 0.15},
        {'f_A': 0.3, 'f_B': 0.7, 'p_AA': 0.25, 'p_BB': 0.45, 'p_AB': 0.15, 'p_BA': 0.15}
    ]
    
    # 测试不同融合策略
    strategies = ['confidence', 'weighted', 'adaptive']
    
    for strategy in strategies:
        print(f"\n测试融合策略: {strategy}")
        
        results = fusion_module(
            theoretical_dist=theoretical_dist,
            residual_corrected_dist=residual_corrected_dist,
            cond=cond,
            mayo_params_batch=mayo_params_batch,
            fusion_strategy=strategy
        )
        
        print(f"  融合分布形状: {results['fused_distribution'].shape}")
        print(f"  置信度范围: [{results['confidence'].min():.3f}, {results['confidence'].max():.3f}]")
        print(f"  质量评分范围: [{results['quality_score'].min():.3f}, {results['quality_score'].max():.3f}]")
        print(f"  融合分布求和: {torch.sum(results['fused_distribution'], dim=1)[:3]}")
        
        if results['fusion_weights'] is not None:
            weights = results['fusion_weights']
            print(f"  权重分布: 理论={weights[:, 0].mean():.3f}, "
                  f"残差={weights[:, 1].mean():.3f}, 组合={weights[:, 2].mean():.3f}")
    
    # 测试损失计算
    print(f"\n测试损失计算:")
    target_dist = torch.softmax(torch.randn(batch_size, bins), dim=1)
    
    loss_dict = fusion_module.compute_fusion_loss(
        fused_dist=results['fused_distribution'],
        target_dist=target_dist,
        confidence=results['confidence'],
        quality_score=results['quality_score']
    )
    
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value.item():.6f}")
    
    print("✅ AdaptiveFusionModule测试通过!")


if __name__ == "__main__":
    test_fusion_module()
