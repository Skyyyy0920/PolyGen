#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练和评估工具函数
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error


class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self):
        pass
    
    def kl_divergence(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        """
        计算KL散度
        
        Args:
            p: 真实分布 [batch_size, bins] 或 [bins]
            q: 预测分布 [batch_size, bins] 或 [bins]
            eps: 数值稳定性参数
            
        Returns:
            KL散度值
        """
        # 确保非负并归一化
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        
        if p.ndim == 1:
            p = p / np.sum(p)
            q = q / np.sum(q)
            return np.sum(p * np.log(p / q))
        else:
            # 批量计算
            p = p / np.sum(p, axis=1, keepdims=True)
            q = q / np.sum(q, axis=1, keepdims=True)
            return np.mean(np.sum(p * np.log(p / q), axis=1))
    
    def js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算JS散度（对称版本的KL散度）
        
        Args:
            p: 真实分布
            q: 预测分布
            
        Returns:
            JS散度值
        """
        # 计算中点分布
        m = 0.5 * (p + q)
        
        # JS散度 = 0.5 * KL(p||m) + 0.5 * KL(q||m)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
    
    def earth_mover_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算Earth Mover's Distance (Wasserstein距离)
        
        Args:
            p: 真实分布
            q: 预测分布
            
        Returns:
            EMD值
        """
        if p.ndim == 1:
            # 单个分布
            x = np.arange(len(p))
            return wasserstein_distance(x, x, p, q)
        else:
            # 批量计算
            emds = []
            x = np.arange(p.shape[1])
            for i in range(p.shape[0]):
                emd = wasserstein_distance(x, x, p[i], q[i])
                emds.append(emd)
            return np.mean(emds)
    
    def mean_squared_error(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算均方误差"""
        return mean_squared_error(p.flatten(), q.flatten())
    
    def mean_absolute_error(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算平均绝对误差"""
        return np.mean(np.abs(p - q))
    
    def peak_position_accuracy(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        计算峰值位置准确率
        
        Args:
            p: 真实分布
            q: 预测分布
            
        Returns:
            峰值位置匹配的比例
        """
        if p.ndim == 1:
            return float(np.argmax(p) == np.argmax(q))
        else:
            peak_p = np.argmax(p, axis=1)
            peak_q = np.argmax(q, axis=1)
            return np.mean(peak_p == peak_q)
    
    def distribution_moments(self, dist: np.ndarray) -> Dict[str, float]:
        """
        计算分布的统计矩
        
        Args:
            dist: 分布 [batch_size, bins] 或 [bins]
            
        Returns:
            统计矩字典
        """
        if dist.ndim == 1:
            x = np.arange(1, len(dist) + 1)  # 块长度从1开始
            
            # 归一化
            dist_norm = dist / np.sum(dist)
            
            # 计算各阶矩
            mean = np.sum(x * dist_norm)
            variance = np.sum((x - mean) ** 2 * dist_norm)
            std = np.sqrt(variance)
            
            # 偏度和峰度
            if std > 0:
                skewness = np.sum(((x - mean) / std) ** 3 * dist_norm)
                kurtosis = np.sum(((x - mean) / std) ** 4 * dist_norm) - 3
            else:
                skewness = 0.0
                kurtosis = 0.0
            
            return {
                'mean': mean,
                'std': std,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
        else:
            # 批量计算平均值
            moments_list = [self.distribution_moments(dist[i]) for i in range(dist.shape[0])]
            
            avg_moments = {}
            for key in moments_list[0].keys():
                avg_moments[key] = np.mean([m[key] for m in moments_list])
            
            return avg_moments
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            predictions: 预测分布 [batch_size, bins]
            targets: 真实分布 [batch_size, bins]
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 基本误差指标
        metrics['kl_divergence'] = self.kl_divergence(targets, predictions)
        metrics['js_divergence'] = self.js_divergence(targets, predictions)
        metrics['earth_mover_distance'] = self.earth_mover_distance(targets, predictions)
        metrics['mse'] = self.mean_squared_error(targets, predictions)
        metrics['mae'] = self.mean_absolute_error(targets, predictions)
        
        # 峰值位置准确率
        metrics['peak_accuracy'] = self.peak_position_accuracy(targets, predictions)
        
        # 分布矩对比
        target_moments = self.distribution_moments(targets)
        pred_moments = self.distribution_moments(predictions)
        
        for key in target_moments.keys():
            metrics[f'target_{key}'] = target_moments[key]
            metrics[f'pred_{key}'] = pred_moments[key]
            metrics[f'{key}_error'] = abs(target_moments[key] - pred_moments[key])
        
        return metrics
    
    def compute_mayo_lewis_comparison(self, 
                                    predictions: np.ndarray,
                                    targets: np.ndarray,
                                    theoretical: np.ndarray) -> Dict[str, float]:
        """
        计算与Mayo-Lewis理论的对比指标
        
        Args:
            predictions: 模型预测 [batch_size, bins]
            targets: 真实分布 [batch_size, bins]
            theoretical: Mayo-Lewis理论分布 [batch_size, bins]
            
        Returns:
            对比指标字典
        """
        metrics = {}
        
        # 模型 vs 真实
        metrics['model_vs_target_kl'] = self.kl_divergence(targets, predictions)
        metrics['model_vs_target_emd'] = self.earth_mover_distance(targets, predictions)
        
        # 理论 vs 真实
        metrics['theory_vs_target_kl'] = self.kl_divergence(targets, theoretical)
        metrics['theory_vs_target_emd'] = self.earth_mover_distance(targets, theoretical)
        
        # 模型 vs 理论
        metrics['model_vs_theory_kl'] = self.kl_divergence(theoretical, predictions)
        metrics['model_vs_theory_emd'] = self.earth_mover_distance(theoretical, predictions)
        
        # 改进度计算
        kl_improvement = (metrics['theory_vs_target_kl'] - metrics['model_vs_target_kl']) / metrics['theory_vs_target_kl'] * 100
        emd_improvement = (metrics['theory_vs_target_emd'] - metrics['model_vs_target_emd']) / metrics['theory_vs_target_emd'] * 100
        
        metrics['kl_improvement_percent'] = kl_improvement
        metrics['emd_improvement_percent'] = emd_improvement
        
        return metrics


def create_learning_rate_scheduler(optimizer: torch.optim.Optimizer, 
                                 scheduler_type: str = 'cosine',
                                 **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 调度器参数
        
    Returns:
        学习率调度器
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """计算模型梯度范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def log_model_weights(model: torch.nn.Module, writer, step: int):
    """记录模型权重分布到tensorboard"""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f'weights/{name}', param.data, step)
            writer.add_histogram(f'gradients/{name}', param.grad.data, step)


def save_predictions_visualization(predictions: np.ndarray,
                                 targets: np.ndarray,
                                 theoretical: np.ndarray,
                                 save_path: str,
                                 num_samples: int = 9):
    """
    保存预测结果可视化
    
    Args:
        predictions: 预测分布
        targets: 真实分布
        theoretical: 理论分布
        save_path: 保存路径
        num_samples: 可视化样本数
    """
    import matplotlib.pyplot as plt
    
    num_samples = min(num_samples, predictions.shape[0])
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        x = np.arange(1, predictions.shape[1] + 1)
        
        ax.plot(x, targets[i], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(x, predictions[i], 'b-', linewidth=2, label='Prediction', alpha=0.7)
        ax.plot(x, theoretical[i], 'r--', linewidth=2, label='Mayo-Lewis', alpha=0.7)
        
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Block Length')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_evaluation_metrics():
    """测试评估指标"""
    print("🧪 测试EvaluationMetrics...")
    
    evaluator = EvaluationMetrics()
    
    # 生成测试数据
    batch_size = 5
    bins = 20
    
    # 真实分布（几何分布）
    x = np.arange(1, bins + 1)
    targets = []
    for i in range(batch_size):
        p = 0.2 + 0.1 * i  # 不同的参数
        dist = (1 - p) * (p ** (x - 1))
        dist = dist / np.sum(dist)
        targets.append(dist)
    targets = np.stack(targets)
    
    # 预测分布（添加噪声）
    predictions = targets + np.random.normal(0, 0.02, targets.shape)
    predictions = np.maximum(predictions, 0)
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
    
    # 理论分布（稍有偏差）
    theoretical = targets * 0.9 + 0.1 * np.ones_like(targets) / bins
    theoretical = theoretical / np.sum(theoretical, axis=1, keepdims=True)
    
    # 计算指标
    print("基本评估指标:")
    basic_metrics = evaluator.compute_metrics(predictions, targets)
    for key, value in basic_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nMayo-Lewis对比指标:")
    comparison_metrics = evaluator.compute_mayo_lewis_comparison(predictions, targets, theoretical)
    for key, value in comparison_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("✅ EvaluationMetrics测试通过!")


if __name__ == "__main__":
    test_evaluation_metrics()
