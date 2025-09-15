#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双生成器主模型

整合Mayo-Lewis理论计算、残差生成器和自适应融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..data.mayo_lewis import MayoLewisCalculator
from .condition_encoder import ConditionEncoder, SequenceEncoder
from .residual_generator import ResidualGenerator, NoiseSchedule, q_sample_vparam
from .fusion_module import AdaptiveFusionModule


class DualGeneratorModel(nn.Module):
    """
    双生成器聚合物块分布生成模型
    
    架构组成：
    1. Mayo-Lewis理论计算器（确定性，无参数）
    2. 条件编码器（编码实验条件和序列信息）
    3. 残差生成器（基于扩散模型，学习理论偏差）
    4. 自适应融合模块（动态组合理论和残差分布）
    """
    
    def __init__(self,
                 bins: int = 50,
                 condition_dim: int = 17,

                 cond_encoder_d_model: int = 128,
                 cond_encoder_proj_dim: int = 256,
                 cond_encoder_layers: int = 3,

                 seq_encoder_d_model: int = 64,
                 seq_encoder_layers: int = 2,

                 residual_hidden_size: int = 256,
                 residual_num_layers: int = 8,
                 residual_num_heads: int = 8,

                 fusion_hidden_dim: int = 128,
                 fusion_num_layers: int = 3,

                 diffusion_steps: int = 1000,
                 noise_schedule: str = 'cosine',

                 dropout: float = 0.1,
                 temperature: float = 0.1):
        super().__init__()
        
        self.bins = bins
        self.condition_dim = condition_dim
        self.diffusion_steps = diffusion_steps
        
        # 1. Mayo-Lewis理论计算器（无参数）
        self.mayo_lewis_calc = MayoLewisCalculator(max_length=bins)
        
        # 2. 条件编码器
        self.condition_encoder = ConditionEncoder(
            in_dim=condition_dim,
            d_model=cond_encoder_d_model,
            proj_dim=cond_encoder_proj_dim,
            num_layers=cond_encoder_layers,
            dropout=dropout,
            temperature=temperature
        )
        
        # 3. 序列编码器
        self.sequence_encoder = SequenceEncoder(
            vocab_size=3,  # 0: pad, 1: A, 2: B
            d_model=seq_encoder_d_model,
            num_layers=seq_encoder_layers,
            max_length=500,  # 设置合理的最大序列长度
            dropout=dropout
        )
        
        # 4. 残差生成器
        residual_cond_dim = (cond_encoder_d_model + seq_encoder_d_model + bins)
        self.residual_generator = ResidualGenerator(
            bins=bins,
            cond_dim=residual_cond_dim,
            hidden_size=residual_hidden_size,
            num_layers=residual_num_layers,
            num_heads=residual_num_heads,
            dropout=dropout
        )
        
        # 5. 自适应融合模块
        fusion_cond_dim = cond_encoder_d_model + seq_encoder_d_model
        self.fusion_module = AdaptiveFusionModule(
            cond_dim=fusion_cond_dim,
            mayo_param_dim=6,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_num_layers,
            dropout=dropout
        )
        
        # 6. 扩散噪声调度器
        self.noise_schedule = NoiseSchedule(
            T=diffusion_steps,
            schedule_type=noise_schedule
        )
        
        print(f"🏗️ DualGeneratorModel初始化完成:")
        print(f"  总参数数量: {self.count_parameters():,}")
        print(f"  分布bins: {bins}")
        print(f"  扩散步数: {diffusion_steps}")
        print(f"  噪声调度: {noise_schedule}")
    
    def count_parameters(self) -> int:
        """计算模型总参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_theoretical_distributions(self, sequences_batch: List[List[str]]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        获取Mayo-Lewis理论分布
        
        Args:
            sequences_batch: 批量序列数据
            
        Returns:
            theoretical_dists: 理论分布 [batch_size, bins]
            mayo_params: Mayo-Lewis参数列表
        """
        theoretical_dists = self.mayo_lewis_calc.batch_calculate_distributions(sequences_batch)
        
        mayo_params = []
        for sequences in sequences_batch:
            params = self.mayo_lewis_calc.extract_sequence_statistics(sequences)
            mayo_params.append(params)
        
        return theoretical_dists, mayo_params
    
    def encode_conditions(self, 
                         condition_features: torch.Tensor,
                         sequences_batch: List[List[str]]) -> Dict[str, torch.Tensor]:
        """
        编码条件特征和序列信息
        
        Args:
            condition_features: 条件特征 [batch_size, condition_dim]
            sequences_batch: 序列批次
            
        Returns:
            编码结果字典
        """
        # 条件编码
        cond_results = self.condition_encoder(condition_features)
        
        # 序列编码
        seq_results = self.sequence_encoder(sequences_batch)
        
        # 组合编码
        combined_embedding = torch.cat([
            cond_results['cond_emb'],
            seq_results['sequence_embedding']
        ], dim=1)
        
        return {
            'condition_embedding': cond_results['cond_emb'],
            'sequence_embedding': seq_results['sequence_embedding'],
            'combined_embedding': combined_embedding,
            'projection_embedding': cond_results['proj_emb']
        }
    
    def forward_residual_generator(self,
                                 x: torch.Tensor,
                                 t: torch.Tensor,
                                 combined_embedding: torch.Tensor,
                                 theoretical_dist: torch.Tensor) -> torch.Tensor:
        """
        残差生成器前向传播
        
        Args:
            x: 噪声输入 [batch_size, bins]
            t: 时间步 [batch_size]
            combined_embedding: 组合嵌入 [batch_size, combined_dim]
            theoretical_dist: 理论分布 [batch_size, bins]
            
        Returns:
            残差预测 [batch_size, bins]
        """
        return self.residual_generator(x, t, combined_embedding, theoretical_dist)
    
    def compute_residual_loss(self,
                            residual_targets: torch.Tensor,
                            condition_features: torch.Tensor,
                            sequences_batch: List[List[str]],
                            theoretical_dists: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算残差生成器损失
        
        Args:
            residual_targets: 残差目标 [batch_size, bins]
            condition_features: 条件特征
            sequences_batch: 序列批次
            theoretical_dists: 理论分布
            
        Returns:
            损失字典
        """
        device = residual_targets.device
        batch_size = residual_targets.size(0)
        
        # 编码条件
        encoding_results = self.encode_conditions(condition_features, sequences_batch)
        combined_embedding = encoding_results['combined_embedding']
        
        # 随机时间步
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        
        # v-参数化扩散采样
        x_t, v_target = q_sample_vparam(residual_targets, t, self.noise_schedule.to(device))
        
        # 残差生成器预测
        v_pred = self.forward_residual_generator(x_t, t, combined_embedding, theoretical_dists)
        
        # MSE损失
        residual_loss = F.mse_loss(v_pred, v_target)
        
        return {
            'residual_loss': residual_loss,
            'v_pred': v_pred,
            'v_target': v_target,
            'encoding_results': encoding_results
        }
    
    def sample_residual_distribution(self,
                                   condition_features: torch.Tensor,
                                   sequences_batch: List[List[str]],
                                   theoretical_dists: torch.Tensor,
                                   num_steps: int = 50,
                                   guidance_scale: float = 1.0,
                                   temperature: float = 1.0) -> torch.Tensor:
        """
        采样残差分布
        
        Args:
            condition_features: 条件特征
            sequences_batch: 序列批次
            theoretical_dists: 理论分布
            num_steps: DDIM采样步数
            guidance_scale: 引导强度
            temperature: 采样温度
            
        Returns:
            采样的残差分布 [batch_size, bins]
        """
        device = condition_features.device
        batch_size = condition_features.size(0)
        
        # 编码条件
        encoding_results = self.encode_conditions(condition_features, sequences_batch)
        combined_embedding = encoding_results['combined_embedding']
        
        # DDIM采样（简化版本）
        schedule = self.noise_schedule.to(device)
        
        # 初始噪声
        x = torch.randn(batch_size, self.bins, device=device)
        
        # 采样步骤
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # 预测v
            with torch.no_grad():
                v_pred = self.forward_residual_generator(x, t_batch, combined_embedding, theoretical_dists)
            
            # DDIM更新步骤（简化）
            alpha_t = schedule.alphas_cumprod[t]
            alpha_prev = schedule.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # 计算x0预测
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            x0_pred = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * v_pred
            
            if i < len(timesteps) - 1:
                # 计算下一步
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
                
                x = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * v_pred
            else:
                x = x0_pred
        
        return x
    
    def forward(self,
                condition_features: torch.Tensor,
                sequences_batch: List[List[str]],
                mode: str = 'inference',
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        主前向传播函数
        
        Args:
            condition_features: 条件特征 [batch_size, condition_dim]
            sequences_batch: 序列批次
            mode: 模式 ('training', 'inference')
            **kwargs: 其他参数
            
        Returns:
            结果字典
        """
        device = condition_features.device
        
        # 1. 获取Mayo-Lewis理论分布
        theoretical_dists, mayo_params = self.get_theoretical_distributions(sequences_batch)
        theoretical_dists = theoretical_dists.to(device)
        
        if mode == 'training':
            # 训练模式：计算损失
            residual_targets = kwargs.get('residual_targets')
            if residual_targets is None:
                raise ValueError("Training mode requires residual_targets")
            
            loss_results = self.compute_residual_loss(
                residual_targets, condition_features, sequences_batch, theoretical_dists
            )
            
            return {
                'theoretical_distributions': theoretical_dists,
                'mayo_parameters': mayo_params,
                **loss_results
            }
        
        elif mode == 'inference':
            # 推理模式：生成分布
            
            # 2. 采样残差分布
            residual_dist = self.sample_residual_distribution(
                condition_features, sequences_batch, theoretical_dists,
                num_steps=kwargs.get('num_steps', 50),
                guidance_scale=kwargs.get('guidance_scale', 1.0),
                temperature=kwargs.get('temperature', 1.0)
            )
            
            # 3. 组合分布（理论 + 残差）
            residual_corrected_dist = theoretical_dists + residual_dist
            
            # 确保非负并归一化
            residual_corrected_dist = F.relu(residual_corrected_dist)
            residual_corrected_dist = residual_corrected_dist / (
                torch.sum(residual_corrected_dist, dim=1, keepdim=True) + 1e-8
            )
            
            # 4. 编码条件用于融合
            encoding_results = self.encode_conditions(condition_features, sequences_batch)
            
            # 5. 自适应融合
            fusion_results = self.fusion_module(
                theoretical_dist=theoretical_dists,
                residual_corrected_dist=residual_corrected_dist,
                cond=encoding_results['combined_embedding'],
                mayo_params_batch=mayo_params,
                fusion_strategy=kwargs.get('fusion_strategy', 'adaptive')
            )
            
            return {
                'theoretical_distributions': theoretical_dists,
                'residual_distributions': residual_dist,
                'residual_corrected_distributions': residual_corrected_dist,
                'mayo_parameters': mayo_params,
                'encoding_results': encoding_results,
                **fusion_results
            }
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        return {
            'model_name': 'DualGeneratorModel',
            'total_parameters': self.count_parameters(),
            'bins': self.bins,
            'condition_dim': self.condition_dim,
            'diffusion_steps': self.diffusion_steps,
            'components': {
                'mayo_lewis_calculator': 'MayoLewisCalculator (no parameters)',
                'condition_encoder': f'{sum(p.numel() for p in self.condition_encoder.parameters()):,} params',
                'sequence_encoder': f'{sum(p.numel() for p in self.sequence_encoder.parameters()):,} params',
                'residual_generator': f'{sum(p.numel() for p in self.residual_generator.parameters()):,} params',
                'fusion_module': f'{sum(p.numel() for p in self.fusion_module.parameters()):,} params'
            }
        }


def test_dual_generator_model():
    """测试双生成器模型"""
    print("🧪 测试DualGeneratorModel...")
    
    # 测试参数
    batch_size = 2  # 减小批次大小以节省内存
    bins = 30
    condition_dim = 17
    
    # 创建模型
    model = DualGeneratorModel(
        bins=bins,
        condition_dim=condition_dim,
        cond_encoder_d_model=64,  # 减小模型大小
        residual_hidden_size=128,
        residual_num_layers=4,
        fusion_hidden_dim=64,
        diffusion_steps=100  # 减少扩散步数
    )
    
    # 显示模型信息
    model_info = model.get_model_info()
    print(f"模型信息:")
    for key, value in model_info.items():
        if key == 'components':
            print(f"  {key}:")
            for comp_name, comp_info in value.items():
                print(f"    {comp_name}: {comp_info}")
        else:
            print(f"  {key}: {value}")
    
    # 生成测试数据
    condition_features = torch.randn(batch_size, condition_dim)
    sequences_batch = [
        ['AAABBBAAABBB', 'BBAABBAA'],
        ['ABABABABAB', 'BABABA']
    ]
    
    # 测试推理模式
    print(f"\n测试推理模式:")
    with torch.no_grad():
        results = model(
            condition_features=condition_features,
            sequences_batch=sequences_batch,
            mode='inference',
            num_steps=10,  # 减少采样步数
            fusion_strategy='adaptive'
        )
    
    print(f"  理论分布形状: {results['theoretical_distributions'].shape}")
    print(f"  残差分布形状: {results['residual_distributions'].shape}")
    print(f"  融合分布形状: {results['fused_distribution'].shape}")
    print(f"  置信度: {results['confidence']}")
    print(f"  质量评分: {results['quality_score']}")
    
    # 检查分布归一化
    fused_sums = torch.sum(results['fused_distribution'], dim=1)
    print(f"  融合分布求和: {fused_sums}")
    
    # 测试训练模式
    print(f"\n测试训练模式:")
    
    # 生成模拟残差目标
    residual_targets = torch.randn(batch_size, bins) * 0.1  # 小的残差
    
    train_results = model(
        condition_features=condition_features,
        sequences_batch=sequences_batch,
        mode='training',
        residual_targets=residual_targets
    )
    
    print(f"  残差损失: {train_results['residual_loss'].item():.6f}")
    print(f"  v预测形状: {train_results['v_pred'].shape}")
    print(f"  v目标形状: {train_results['v_target'].shape}")
    
    print("✅ DualGeneratorModel测试通过!")


if __name__ == "__main__":
    test_dual_generator_model()
