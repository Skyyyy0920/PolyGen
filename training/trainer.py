#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双生成器训练器

实现完整的训练流程，包括条件编码器预训练和双生成器联合训练
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.dual_generator import DualGeneratorModel
from ..data.dataset import DualPolyDataset, collate_dual_poly
from .utils import EvaluationMetrics


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本配置
    experiment_name: str = "dual_generator_experiment"
    output_dir: str = "outputs"
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # 数据配置
    csv_path: str = "PolyGen-F06C/data/copolymer.csv"
    max_samples: Optional[int] = None
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    
    # 模型配置
    bins: int = 50
    condition_dim: int = 17
    cond_encoder_d_model: int = 128
    residual_hidden_size: int = 256
    residual_num_layers: int = 8
    diffusion_steps: int = 1000
    
    # 训练配置
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # 预训练配置
    pretrain_condition_encoder: bool = True
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    
    # 验证和保存
    validate_every: int = 5
    save_every: int = 10
    num_workers: int = 0
    
    # 损失权重
    residual_loss_weight: float = 1.0
    fusion_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.05
    
    # 推理配置
    inference_num_steps: int = 50
    inference_guidance_scale: float = 1.0
    inference_temperature: float = 1.0


class DualGeneratorTrainer:
    """双生成器训练器"""
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        
        # 设置设备
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🖥️ 使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # 初始化日志
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化优化器
        self.optimizer = self._build_optimizer()
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 评估器
        self.evaluator = EvaluationMetrics()
        
        print(f"🎯 训练器初始化完成:")
        print(f"  实验名称: {config.experiment_name}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  训练样本: {len(self.train_loader.dataset)}")
        print(f"  验证样本: {len(self.val_loader.dataset)}")
        print(f"  测试样本: {len(self.test_loader.dataset)}")
    
    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """构建数据加载器"""
        print("📂 构建数据加载器...")
        
        # 训练集
        train_dataset = DualPolyDataset(
            csv_path=self.config.csv_path,
            max_length=self.config.bins,
            max_samples=self.config.max_samples,
            split='train',
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )
        
        # 验证集
        val_dataset = DualPolyDataset(
            csv_path=self.config.csv_path,
            max_length=self.config.bins,
            max_samples=self.config.max_samples,
            split='val',
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )
        
        # 测试集
        test_dataset = DualPolyDataset(
            csv_path=self.config.csv_path,
            max_length=self.config.bins,
            max_samples=self.config.max_samples,
            split='test',
            test_ratio=self.config.test_ratio,
            val_ratio=self.config.val_ratio
        )
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_dual_poly,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_dual_poly,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_dual_poly,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _build_model(self) -> DualGeneratorModel:
        """构建模型"""
        print("🏗️ 构建双生成器模型...")
        
        model = DualGeneratorModel(
            bins=self.config.bins,
            condition_dim=self.config.condition_dim,
            cond_encoder_d_model=self.config.cond_encoder_d_model,
            residual_hidden_size=self.config.residual_hidden_size,
            residual_num_layers=self.config.residual_num_layers,
            diffusion_steps=self.config.diffusion_steps
        ).to(self.device)
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def pretrain_condition_encoder(self):
        """预训练条件编码器"""
        if not self.config.pretrain_condition_encoder:
            print("⏭️ 跳过条件编码器预训练")
            return
        
        print(f"🎯 开始条件编码器预训练 ({self.config.pretrain_epochs} epochs)...")
        
        # 创建预训练优化器
        pretrain_optimizer = optim.AdamW(
            self.model.condition_encoder.parameters(),
            lr=self.config.pretrain_lr,
            weight_decay=self.config.weight_decay
        )
        
        self.model.condition_encoder.train()
        
        for epoch in range(self.config.pretrain_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.train_loader:
                condition_features = batch['condition_features'].to(self.device)
                
                # 前向传播
                results = self.model.condition_encoder(condition_features)
                
                # 对比学习损失（自监督）
                proj_emb1 = results['proj_emb']
                
                # 创建增强版本（添加噪声）
                noise = torch.randn_like(condition_features) * 0.1
                augmented_features = condition_features + noise
                augmented_results = self.model.condition_encoder(augmented_features)
                proj_emb2 = augmented_results['proj_emb']
                
                # 计算对比损失
                contrastive_loss = self.model.condition_encoder.compute_contrastive_loss(
                    proj_emb1, proj_emb2
                )
                
                # 反向传播
                pretrain_optimizer.zero_grad()
                contrastive_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.condition_encoder.parameters(), 
                    self.config.gradient_clip_norm
                )
                pretrain_optimizer.step()
                
                epoch_loss += contrastive_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"  预训练 Epoch {epoch+1:2d}: 对比损失 = {avg_loss:.6f}")
            
            # 记录日志
            self.writer.add_scalar('Pretrain/ContrastiveLoss', avg_loss, epoch)
        
        print("✅ 条件编码器预训练完成!")
        
        # 保存预训练模型
        pretrain_path = self.output_dir / "condition_encoder_pretrained.pt"
        torch.save({
            'model_state_dict': self.model.condition_encoder.state_dict(),
            'optimizer_state_dict': pretrain_optimizer.state_dict(),
            'epoch': self.config.pretrain_epochs,
            'loss': avg_loss
        }, pretrain_path)
        
        print(f"💾 预训练模型已保存: {pretrain_path}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'residual_loss': 0.0,
            'fusion_loss': 0.0,
            'contrastive_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            condition_features = batch['condition_features'].to(self.device)
            residual_targets = batch['residual_targets'].to(self.device)
            block_distributions = batch['block_distributions'].to(self.device)
            sequences_batch = batch['sequences']
            mayo_params_batch = batch['mayo_lewis_params']
            
            # 前向传播 - 训练模式
            train_results = self.model(
                condition_features=condition_features,
                sequences_batch=sequences_batch,
                mode='training',
                residual_targets=residual_targets
            )
            
            # 计算残差损失
            residual_loss = train_results['residual_loss']
            
            # 推理模式获取融合结果
            with torch.no_grad():
                inference_results = self.model(
                    condition_features=condition_features,
                    sequences_batch=sequences_batch,
                    mode='inference',
                    num_steps=10  # 减少步数以加速训练
                )
            
            # 计算融合损失
            fusion_loss_dict = self.model.fusion_module.compute_fusion_loss(
                fused_dist=inference_results['fused_distribution'],
                target_dist=block_distributions,
                confidence=inference_results['confidence'],
                quality_score=inference_results['quality_score']
            )
            fusion_loss = fusion_loss_dict['total_loss']
            
            # 计算对比损失
            encoding_results = train_results['encoding_results']
            proj_emb = encoding_results['projection_embedding']
            
            # 创建对比目标（简化版本）
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if proj_emb.size(0) > 1:
                # 使用批次内的对比学习
                contrastive_loss = self.model.condition_encoder.compute_contrastive_loss(
                    proj_emb, proj_emb.roll(1, dims=0)
                )
            
            # 总损失
            total_loss = (
                self.config.residual_loss_weight * residual_loss +
                self.config.fusion_loss_weight * fusion_loss +
                self.config.contrastive_loss_weight * contrastive_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
            self.optimizer.step()
            
            # 累积损失
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['residual_loss'] += residual_loss.item()
            epoch_losses['fusion_loss'] += fusion_loss.item()
            epoch_losses['contrastive_loss'] += contrastive_loss.item()
            num_batches += 1
            
            self.global_step += 1
            
            # 记录训练日志
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}: "
                      f"Total={total_loss.item():.4f}, "
                      f"Residual={residual_loss.item():.4f}, "
                      f"Fusion={fusion_loss.item():.4f}")
                
                # 记录到tensorboard
                self.writer.add_scalar('Train/BatchLoss', total_loss.item(), self.global_step)
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'mse': 0.0
        }
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                condition_features = batch['condition_features'].to(self.device)
                block_distributions = batch['block_distributions'].to(self.device)
                sequences_batch = batch['sequences']
                
                # 推理
                results = self.model(
                    condition_features=condition_features,
                    sequences_batch=sequences_batch,
                    mode='inference',
                    num_steps=self.config.inference_num_steps
                )
                
                fused_dist = results['fused_distribution']
                
                # 计算损失
                kl_loss = torch.nn.functional.kl_div(
                    torch.log(fused_dist + 1e-8),
                    block_distributions,
                    reduction='batchmean'
                )
                mse_loss = torch.nn.functional.mse_loss(fused_dist, block_distributions)
                
                val_losses['kl_divergence'] += kl_loss.item()
                val_losses['mse'] += mse_loss.item()
                val_losses['total_loss'] += kl_loss.item() + mse_loss.item()
                
                # 收集预测和目标
                all_predictions.append(fused_dist.cpu())
                all_targets.append(block_distributions.cpu())
                
                num_batches += 1
        
        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # 计算详细评估指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        detailed_metrics = self.evaluator.compute_metrics(
            all_predictions.numpy(),
            all_targets.numpy()
        )
        
        val_losses.update(detailed_metrics)
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # 保存最新检查点
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 最佳模型已保存: {best_path}")
    
    def train(self):
        """完整训练流程"""
        print(f"🚀 开始双生成器训练...")
        print(f"  总epochs: {self.config.num_epochs}")
        print(f"  批次大小: {self.config.batch_size}")
        print(f"  学习率: {self.config.learning_rate}")
        
        # 预训练条件编码器
        self.pretrain_condition_encoder()
        
        # 主训练循环
        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            print(f"\n📈 Epoch {epoch}/{self.config.num_epochs}")
            
            # 训练
            train_losses = self.train_epoch()
            
            # 记录训练损失
            for loss_name, loss_value in train_losses.items():
                self.writer.add_scalar(f'Train/{loss_name}', loss_value, epoch)
            
            epoch_time = time.time() - start_time
            
            print(f"  训练损失: Total={train_losses['total_loss']:.6f}, "
                  f"Residual={train_losses['residual_loss']:.6f}, "
                  f"时间={epoch_time:.1f}s")
            
            # 验证
            if epoch % self.config.validate_every == 0:
                print(f"  🔍 验证中...")
                val_losses = self.validate()
                
                # 记录验证损失
                for loss_name, loss_value in val_losses.items():
                    self.writer.add_scalar(f'Val/{loss_name}', loss_value, epoch)
                
                print(f"  验证损失: KL={val_losses['kl_divergence']:.6f}, "
                      f"MSE={val_losses['mse']:.6f}")
                
                # 检查是否是最佳模型
                current_val_loss = val_losses['total_loss']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
                    print(f"  🎉 新的最佳验证损失: {self.best_val_loss:.6f}")
                
                # 保存检查点
                if epoch % self.config.save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
        
        print(f"\n✅ 训练完成!")
        print(f"  最佳验证损失: {self.best_val_loss:.6f}")
        print(f"  输出目录: {self.output_dir}")
        
        # 关闭tensorboard
        self.writer.close()


def test_trainer():
    """测试训练器"""
    print("🧪 测试DualGeneratorTrainer...")
    
    # 创建测试配置
    config = TrainingConfig(
        experiment_name="test_dual_generator",
        csv_path="PolyGen-F06C/data/copolymer.csv",
        max_samples=50,  # 限制样本数
        bins=20,
        batch_size=2,
        num_epochs=2,
        pretrain_epochs=1,
        validate_every=1,
        save_every=1,
        diffusion_steps=10,
        cond_encoder_d_model=32,
        residual_hidden_size=64,
        residual_num_layers=2
    )
    
    try:
        # 创建训练器
        trainer = DualGeneratorTrainer(config)
        
        # 运行训练
        trainer.train()
        
        print("✅ DualGeneratorTrainer测试通过!")
        
    except Exception as e:
        print(f"❌ DualGeneratorTrainer测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trainer()
