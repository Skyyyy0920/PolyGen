#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
与原始PolyGen-F06C的对比实验

实现完整的基准测试，对比双生成器方法与原始方法的性能
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入原始PolyGen-F06C模块
sys.path.append(str(Path(__file__).parent.parent.parent / "PolyGen-F06C"))
try:
    from data.dataset import ChainSetDataset, collate_fn_set_transformer
    from data.block_dist import mayo_lewis_from_sequence
    from src.encoder import ConditionEncoder as OriginalConditionEncoder
    from src.diffusion import DiT1D, NoiseSchedule as OriginalNoiseSchedule, ddim_sample, hist_to_logits, logits_to_hist
except ImportError as e:
    print(f"警告: 无法导入原始PolyGen-F06C模块: {e}")
    print("请确保PolyGen-F06C目录存在且包含必要文件")

# 导入双生成器模块
from ..models.dual_generator import DualGeneratorModel
from ..data.dataset import DualPolyDataset, collate_dual_poly
from ..training.utils import EvaluationMetrics


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 数据配置
    csv_path: str = "PolyGen-F06C/data/copolymer.csv"
    max_samples: Optional[int] = 1000
    test_ratio: float = 0.2
    batch_size: int = 8
    
    # 模型配置
    bins: int = 50
    
    # 原始模型配置
    original_cond_ckpt: Optional[str] = None
    original_diffusion_ckpt: Optional[str] = None
    
    # 双生成器配置
    dual_model_ckpt: Optional[str] = None
    
    # 评估配置
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    temperature: float = 1.0
    
    # 输出配置
    output_dir: str = "outputs/benchmark_comparison"
    save_visualizations: bool = True
    num_visualization_samples: int = 12


class PolyGenComparison:
    """PolyGen对比实验类"""
    
    def __init__(self, config: BenchmarkConfig):
        """
        初始化对比实验
        
        Args:
            config: 基准测试配置
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / "benchmark_config.json", 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        # 初始化评估器
        self.evaluator = EvaluationMetrics()
        
        # 加载数据
        self.test_loader = self._load_test_data()
        
        print(f"🔬 PolyGen对比实验初始化完成")
        print(f"  设备: {self.device}")
        print(f"  测试样本数: {len(self.test_loader.dataset)}")
        print(f"  输出目录: {self.output_dir}")
    
    def _load_test_data(self) -> DataLoader:
        """加载测试数据"""
        print("📂 加载测试数据...")
        
        # 使用双生成器数据集格式
        test_dataset = DualPolyDataset(
            csv_path=self.config.csv_path,
            max_length=self.config.bins,
            max_samples=self.config.max_samples,
            split='test',
            test_ratio=self.config.test_ratio
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_dual_poly,
            num_workers=0
        )
        
        return test_loader
    
    def load_original_model(self) -> Optional[Tuple]:
        """
        加载原始PolyGen-F06C模型
        
        Returns:
            (condition_encoder, diffusion_model, noise_schedule) 或 None
        """
        if not self.config.original_cond_ckpt or not self.config.original_diffusion_ckpt:
            print("⚠️ 未提供原始模型检查点路径，跳过原始模型评估")
            return None
        
        try:
            print("🔧 加载原始PolyGen-F06C模型...")
            
            # 加载条件编码器
            cond_ckpt = torch.load(self.config.original_cond_ckpt, map_location=self.device, weights_only=False)
            cond_args = cond_ckpt.get("args", {})
            
            condition_encoder = OriginalConditionEncoder(
                in_dim=int(cond_args.get("cond_in_dim", 17)),
                d_model=int(cond_args.get("d_model", 128)),
                proj_dim=int(cond_args.get("proj_dim", 256)),
                num_layers=int(cond_args.get("num_layers", 3)),
                dropout=cond_args.get("dropout", 0.1),
                temperature=float(cond_args.get("temperature", 0.10)),
            ).to(self.device)
            
            condition_encoder.load_state_dict(cond_ckpt["model"], strict=False)
            condition_encoder.eval()
            
            # 加载扩散模型
            diffusion_ckpt = torch.load(self.config.original_diffusion_ckpt, map_location=self.device, weights_only=False)
            diffusion_args = diffusion_ckpt.get("args", {})
            
            diffusion_model = DiT1D(
                bins=self.config.bins,
                cond_dim=int(cond_args.get("d_model", 128)),
                d_model=int(diffusion_args.get("dit_d_model", 256)),
                n_layers=int(diffusion_args.get("dit_layers", 8)),
                n_heads=int(diffusion_args.get("dit_heads", 8)),
                dropout=diffusion_args.get("dropout", 0.1),
                film_each_layer=diffusion_args.get("film_each_layer", True)
            ).to(self.device)
            
            diffusion_model.load_state_dict(diffusion_ckpt["model"], strict=False)
            diffusion_model.eval()
            
            # 噪声调度器
            noise_schedule = OriginalNoiseSchedule(
                T=int(diffusion_args.get("T", 1000))
            ).to(self.device)
            
            print("✅ 原始模型加载完成")
            return condition_encoder, diffusion_model, noise_schedule
            
        except Exception as e:
            print(f"❌ 加载原始模型失败: {e}")
            return None
    
    def load_dual_generator_model(self) -> Optional[DualGeneratorModel]:
        """
        加载双生成器模型
        
        Returns:
            DualGeneratorModel 或 None
        """
        if not self.config.dual_model_ckpt:
            print("⚠️ 未提供双生成器模型检查点路径，跳过双生成器评估")
            return None
        
        try:
            print("🔧 加载双生成器模型...")
            
            checkpoint = torch.load(self.config.dual_model_ckpt, map_location=self.device, weights_only=False)
            
            # 从检查点获取配置
            model_config = checkpoint.get("config", {})
            
            # 创建模型
            dual_model = DualGeneratorModel(
                bins=self.config.bins,
                condition_dim=model_config.get("condition_dim", 17),
                cond_encoder_d_model=model_config.get("cond_encoder_d_model", 128),
                residual_hidden_size=model_config.get("residual_hidden_size", 256),
                residual_num_layers=model_config.get("residual_num_layers", 8),
                diffusion_steps=model_config.get("diffusion_steps", 1000)
            ).to(self.device)
            
            # 加载权重
            dual_model.load_state_dict(checkpoint["model_state_dict"])
            dual_model.eval()
            
            print("✅ 双生成器模型加载完成")
            return dual_model
            
        except Exception as e:
            print(f"❌ 加载双生成器模型失败: {e}")
            return None
    
    def evaluate_original_model(self, original_models: Tuple) -> Dict[str, any]:
        """
        评估原始PolyGen-F06C模型
        
        Args:
            original_models: (condition_encoder, diffusion_model, noise_schedule)
            
        Returns:
            评估结果字典
        """
        condition_encoder, diffusion_model, noise_schedule = original_models
        
        print("📊 评估原始PolyGen-F06C模型...")
        
        all_predictions = []
        all_targets = []
        all_theoretical = []
        all_mayo_params = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                condition_features = batch['condition_features'].to(self.device)
                block_distributions = batch['block_distributions'].to(self.device)
                sequences_batch = batch['sequences']
                
                batch_size = condition_features.size(0)
                
                start_time = time.time()
                
                # 条件编码
                cond_results = condition_encoder(cond=condition_features)
                cond_emb = cond_results["cond_emb"]
                
                # DDIM采样
                z0, _ = ddim_sample(
                    model=diffusion_model,
                    cond=cond_emb,
                    schedule=noise_schedule,
                    steps=self.config.num_inference_steps,
                    guidance=None,
                    bins=self.config.bins,
                    tau=self.config.temperature
                )
                
                # 转换为分布
                predictions = logits_to_hist(z0, tau=self.config.temperature)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time / batch_size)  # 每个样本的时间
                
                # 计算Mayo-Lewis理论分布
                theoretical_dists = []
                mayo_params = []
                for sequences in sequences_batch:
                    theo_dist = mayo_lewis_from_sequence(sequences, max_length=self.config.bins)
                    theoretical_dists.append(theo_dist)
                    
                    # 提取Mayo-Lewis参数（简化版本）
                    all_seq = ''.join(sequences)
                    f_A = all_seq.count('A') / len(all_seq) if all_seq else 0.5
                    mayo_params.append({'f_A': f_A})
                
                theoretical_dists = torch.tensor(np.stack(theoretical_dists), dtype=torch.float32)
                
                # 收集结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(block_distributions.cpu().numpy())
                all_theoretical.append(theoretical_dists.numpy())
                all_mayo_params.extend(mayo_params)
                
                if batch_idx % 10 == 0:
                    print(f"  处理批次 {batch_idx+1}/{len(self.test_loader)}")
        
        # 合并所有结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_theoretical = np.concatenate(all_theoretical, axis=0)
        
        # 计算评估指标
        metrics = self.evaluator.compute_metrics(all_predictions, all_targets)
        mayo_comparison = self.evaluator.compute_mayo_lewis_comparison(
            all_predictions, all_targets, all_theoretical
        )
        
        # 性能统计
        avg_inference_time = np.mean(inference_times)
        
        results = {
            'model_type': 'PolyGen-F06C Original',
            'metrics': metrics,
            'mayo_lewis_comparison': mayo_comparison,
            'performance': {
                'avg_inference_time_per_sample': avg_inference_time,
                'total_samples': len(all_predictions)
            },
            'predictions': all_predictions,
            'targets': all_targets,
            'theoretical': all_theoretical,
            'mayo_parameters': all_mayo_params
        }
        
        print(f"✅ 原始模型评估完成")
        print(f"  KL散度: {metrics['kl_divergence']:.6f}")
        print(f"  EMD: {metrics['earth_mover_distance']:.6f}")
        print(f"  平均推理时间: {avg_inference_time:.4f}s/样本")
        
        return results
    
    def evaluate_dual_generator(self, dual_model: DualGeneratorModel) -> Dict[str, any]:
        """
        评估双生成器模型
        
        Args:
            dual_model: 双生成器模型
            
        Returns:
            评估结果字典
        """
        print("📊 评估双生成器模型...")
        
        all_predictions = []
        all_targets = []
        all_theoretical = []
        all_residual_corrected = []
        all_confidences = []
        all_quality_scores = []
        all_mayo_params = []
        
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                condition_features = batch['condition_features'].to(self.device)
                block_distributions = batch['block_distributions'].to(self.device)
                sequences_batch = batch['sequences']
                
                batch_size = condition_features.size(0)
                
                start_time = time.time()
                
                # 推理
                results = dual_model(
                    condition_features=condition_features,
                    sequences_batch=sequences_batch,
                    mode='inference',
                    num_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    temperature=self.config.temperature,
                    fusion_strategy='adaptive'
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time / batch_size)
                
                # 收集结果
                all_predictions.append(results['fused_distribution'].cpu().numpy())
                all_targets.append(block_distributions.cpu().numpy())
                all_theoretical.append(results['theoretical_distributions'].cpu().numpy())
                all_residual_corrected.append(results['residual_corrected_distributions'].cpu().numpy())
                all_confidences.append(results['confidence'].cpu().numpy())
                all_quality_scores.append(results['quality_score'].cpu().numpy())
                all_mayo_params.extend(results['mayo_parameters'])
                
                if batch_idx % 10 == 0:
                    print(f"  处理批次 {batch_idx+1}/{len(self.test_loader)}")
        
        # 合并所有结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_theoretical = np.concatenate(all_theoretical, axis=0)
        all_residual_corrected = np.concatenate(all_residual_corrected, axis=0)
        all_confidences = np.concatenate(all_confidences, axis=0)
        all_quality_scores = np.concatenate(all_quality_scores, axis=0)
        
        # 计算评估指标
        metrics = self.evaluator.compute_metrics(all_predictions, all_targets)
        mayo_comparison = self.evaluator.compute_mayo_lewis_comparison(
            all_predictions, all_targets, all_theoretical
        )
        
        # 性能统计
        avg_inference_time = np.mean(inference_times)
        
        results = {
            'model_type': 'DualGenerator',
            'metrics': metrics,
            'mayo_lewis_comparison': mayo_comparison,
            'performance': {
                'avg_inference_time_per_sample': avg_inference_time,
                'total_samples': len(all_predictions)
            },
            'predictions': all_predictions,
            'targets': all_targets,
            'theoretical': all_theoretical,
            'residual_corrected': all_residual_corrected,
            'confidences': all_confidences,
            'quality_scores': all_quality_scores,
            'mayo_parameters': all_mayo_params,
            'dual_generator_specific': {
                'avg_confidence': float(np.mean(all_confidences)),
                'avg_quality_score': float(np.mean(all_quality_scores)),
                'confidence_std': float(np.std(all_confidences)),
                'quality_score_std': float(np.std(all_quality_scores))
            }
        }
        
        print(f"✅ 双生成器评估完成")
        print(f"  KL散度: {metrics['kl_divergence']:.6f}")
        print(f"  EMD: {metrics['earth_mover_distance']:.6f}")
        print(f"  平均推理时间: {avg_inference_time:.4f}s/样本")
        print(f"  平均置信度: {np.mean(all_confidences):.3f}")
        print(f"  平均质量评分: {np.mean(all_quality_scores):.3f}")
        
        return results
    
    def run_comparison(self) -> Dict[str, any]:
        """
        运行完整对比实验
        
        Returns:
            对比结果字典
        """
        print(f"🚀 开始PolyGen对比实验...")
        
        comparison_results = {
            'config': self.config.__dict__,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'results': {}
        }
        
        # 评估原始模型
        original_models = self.load_original_model()
        if original_models is not None:
            original_results = self.evaluate_original_model(original_models)
            comparison_results['results']['original'] = original_results
        
        # 评估双生成器模型
        dual_model = self.load_dual_generator_model()
        if dual_model is not None:
            dual_results = self.evaluate_dual_generator(dual_model)
            comparison_results['results']['dual_generator'] = dual_results
        
        # 对比分析
        if 'original' in comparison_results['results'] and 'dual_generator' in comparison_results['results']:
            comparison_analysis = self._analyze_comparison(
                comparison_results['results']['original'],
                comparison_results['results']['dual_generator']
            )
            comparison_results['comparison_analysis'] = comparison_analysis
        
        # 保存结果
        self._save_results(comparison_results)
        
        return comparison_results
    
    def _analyze_comparison(self, original_results: Dict, dual_results: Dict) -> Dict[str, any]:
        """分析对比结果"""
        print("🔍 分析对比结果...")
        
        orig_metrics = original_results['metrics']
        dual_metrics = dual_results['metrics']
        
        # 计算改进度
        improvements = {}
        key_metrics = ['kl_divergence', 'earth_mover_distance', 'mse', 'mae']
        
        for metric in key_metrics:
            if metric in orig_metrics and metric in dual_metrics:
                orig_val = orig_metrics[metric]
                dual_val = dual_metrics[metric]
                
                if orig_val != 0:
                    improvement = (orig_val - dual_val) / orig_val * 100
                    improvements[f'{metric}_improvement_percent'] = improvement
        
        # 性能对比
        orig_time = original_results['performance']['avg_inference_time_per_sample']
        dual_time = dual_results['performance']['avg_inference_time_per_sample']
        
        time_ratio = dual_time / orig_time if orig_time != 0 else float('inf')
        
        # Mayo-Lewis对比分析
        orig_mayo = original_results['mayo_lewis_comparison']
        dual_mayo = dual_results['mayo_lewis_comparison']
        
        mayo_analysis = {}
        for key in ['kl_improvement_percent', 'emd_improvement_percent']:
            if key in orig_mayo and key in dual_mayo:
                mayo_analysis[f'original_{key}'] = orig_mayo[key]
                mayo_analysis[f'dual_{key}'] = dual_mayo[key]
                mayo_analysis[f'{key}_difference'] = dual_mayo[key] - orig_mayo[key]
        
        analysis = {
            'metric_improvements': improvements,
            'performance_comparison': {
                'original_inference_time': orig_time,
                'dual_inference_time': dual_time,
                'time_ratio': time_ratio,
                'dual_is_faster': time_ratio < 1.0
            },
            'mayo_lewis_analysis': mayo_analysis,
            'summary': self._generate_summary(improvements, time_ratio, mayo_analysis)
        }
        
        return analysis
    
    def _generate_summary(self, improvements: Dict, time_ratio: float, mayo_analysis: Dict) -> str:
        """生成对比摘要"""
        summary_lines = ["=== PolyGen对比实验摘要 ==="]
        
        # 性能改进
        kl_improvement = improvements.get('kl_divergence_improvement_percent', 0)
        emd_improvement = improvements.get('earth_mover_distance_improvement_percent', 0)
        
        if kl_improvement > 0:
            summary_lines.append(f"✅ 双生成器KL散度改进: {kl_improvement:.2f}%")
        else:
            summary_lines.append(f"❌ 双生成器KL散度下降: {-kl_improvement:.2f}%")
        
        if emd_improvement > 0:
            summary_lines.append(f"✅ 双生成器EMD改进: {emd_improvement:.2f}%")
        else:
            summary_lines.append(f"❌ 双生成器EMD下降: {-emd_improvement:.2f}%")
        
        # 推理时间
        if time_ratio < 1.0:
            summary_lines.append(f"⚡ 双生成器推理速度提升: {(1/time_ratio - 1)*100:.1f}%")
        else:
            summary_lines.append(f"🐌 双生成器推理速度下降: {(time_ratio - 1)*100:.1f}%")
        
        # Mayo-Lewis对比
        dual_mayo_kl = mayo_analysis.get('dual_kl_improvement_percent', 0)
        orig_mayo_kl = mayo_analysis.get('original_kl_improvement_percent', 0)
        
        summary_lines.append(f"📐 原始模型 vs Mayo-Lewis: {orig_mayo_kl:.2f}% KL改进")
        summary_lines.append(f"🔄 双生成器 vs Mayo-Lewis: {dual_mayo_kl:.2f}% KL改进")
        
        return "\n".join(summary_lines)
    
    def _save_results(self, results: Dict[str, any]):
        """保存对比结果"""
        print("💾 保存对比结果...")
        
        # 保存JSON结果（移除numpy数组）
        json_results = {}
        for key, value in results.items():
            if key == 'results':
                json_results[key] = {}
                for model_name, model_results in value.items():
                    json_results[key][model_name] = {
                        k: v for k, v in model_results.items() 
                        if k not in ['predictions', 'targets', 'theoretical', 'residual_corrected']
                    }
            else:
                json_results[key] = value
        
        with open(self.output_dir / "comparison_results.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # 保存numpy数组
        if 'results' in results:
            for model_name, model_results in results['results'].items():
                model_dir = self.output_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                for array_name in ['predictions', 'targets', 'theoretical', 'residual_corrected']:
                    if array_name in model_results:
                        np.save(model_dir / f"{array_name}.npy", model_results[array_name])
        
        print(f"✅ 结果已保存到: {self.output_dir}")


def create_benchmark_config() -> BenchmarkConfig:
    """创建基准测试配置"""
    return BenchmarkConfig(
        csv_path="PolyGen-F06C/data/copolymer.csv",
        max_samples=500,  # 限制样本数用于快速测试
        bins=50,
        batch_size=8,
        num_inference_steps=50,
        output_dir="outputs/benchmark_comparison",
        # 注意: 需要提供实际的模型检查点路径
        original_cond_ckpt=None,  # "path/to/original/condition_encoder.pt"
        original_diffusion_ckpt=None,  # "path/to/original/diffusion_model.pt"
        dual_model_ckpt=None  # "path/to/dual_generator/best_model.pt"
    )


def main():
    """主函数"""
    print("🔬 PolyGen对比实验")
    
    # 创建配置
    config = create_benchmark_config()
    
    # 运行对比实验
    comparison = PolyGenComparison(config)
    results = comparison.run_comparison()
    
    # 显示摘要
    if 'comparison_analysis' in results:
        print("\n" + results['comparison_analysis']['summary'])
    
    print(f"\n✅ 对比实验完成!")
    print(f"📁 结果保存在: {config.output_dir}")


if __name__ == "__main__":
    main()
