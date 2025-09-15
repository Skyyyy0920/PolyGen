#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双生成器模型评估脚本

对比评估双生成器模型与baseline模型的性能
"""

import os
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 导入你的模型组件
from DualPolyGen.models.dual_generator import DualGeneratorModel
from DualPolyGen.data.fixed_dataset import PolyDataset, collate_fixed_poly
from DualPolyGen.data.mayo_lewis import MayoLewisCalculator


def set_seed(seed: int = 42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算KL散度"""
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    return (p * (p.log() - q.log())).sum(dim=-1)


def emd_1d(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算1D Earth Mover's Distance (Wasserstein距离)"""
    cdf_p = p.cumsum(dim=-1)
    cdf_q = q.cumsum(dim=-1)
    return (cdf_p - cdf_q).abs().sum(dim=-1)


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算Jensen-Shannon散度"""
    eps = 1e-8
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def load_dual_generator_model(model_path: str, device: torch.device, args) -> DualGeneratorModel:
    """加载双生成器模型"""
    print(f"加载双生成器模型: {model_path}")

    # 创建模型
    model = DualGeneratorModel(
        bins=args.bins,
        condition_dim=args.condition_dim,
        cond_encoder_d_model=args.cond_dim,
        residual_hidden_size=args.residual_dim,
        residual_num_layers=args.residual_layers,
        diffusion_steps=args.diffusion_steps
    ).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


def evaluate_dual_generator(model: DualGeneratorModel, test_loader, device: torch.device, args):
    """评估双生成器模型"""
    model.eval()
    results = []

    print(f"开始评估双生成器模型...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"处理批次 {batch_idx + 1}/{len(test_loader)}")

            # 获取数据
            condition_features = batch['condition_features'].to(device)
            sequences_batch = batch['sequences']
            block_distributions = batch['block_distributions'].to(device)

            # 模型推理
            results_dict = model(
                condition_features=condition_features,
                sequences_batch=sequences_batch,
                mode='inference',
                num_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                temperature=args.inference_temp,
                fusion_strategy=args.fusion_strategy
            )

            # 获取预测分布
            pred_distributions = results_dict['fused_distribution']
            theoretical_distributions = results_dict['theoretical_distributions']
            residual_distributions = results_dict['residual_distributions']

            # 计算指标
            kl_scores = kl_divergence(block_distributions, pred_distributions)
            emd_scores = emd_1d(block_distributions, pred_distributions)
            js_scores = jensen_shannon_divergence(block_distributions, pred_distributions)

            # 理论分布的指标（用于对比）
            kl_theory = kl_divergence(block_distributions, theoretical_distributions)
            emd_theory = emd_1d(block_distributions, theoretical_distributions)

            # 保存结果
            batch_size = condition_features.size(0)
            for i in range(batch_size):
                sample_idx = batch_idx * args.batch_size + i

                results.append({
                    'sample_idx': sample_idx,
                    'kl_fused': kl_scores[i].item(),
                    'emd_fused': emd_scores[i].item(),
                    'js_fused': js_scores[i].item(),
                    'kl_theory': kl_theory[i].item(),
                    'emd_theory': emd_theory[i].item(),
                    'pred_dist': pred_distributions[i].cpu().numpy(),
                    'true_dist': block_distributions[i].cpu().numpy(),
                    'theory_dist': theoretical_distributions[i].cpu().numpy(),
                    'residual_dist': residual_distributions[i].cpu().numpy(),
                    'sequences': sequences_batch[i],
                    'condition_features': condition_features[i].cpu().numpy(),
                    'confidence': results_dict.get('confidence', torch.ones(batch_size))[i].item(),
                    'quality_score': results_dict.get('quality_score', torch.ones(batch_size))[i].item()
                })

    return results


def create_overlay_plot(true_dist: np.ndarray, pred_dist: np.ndarray, theory_dist: np.ndarray,
                        sequences: List[str], save_path: Path, title: str = "",
                        show_residual: bool = False, residual_dist: np.ndarray = None):
    """创建对比图"""
    max_bins = len(true_dist)
    x = np.arange(1, max_bins + 1)

    plt.figure(figsize=(8, 6), dpi=150)
    width = 0.35  # 增加柱宽，因为现在只有两组柱状图

    # 真实分布 (ground truth)
    plt.bar(x - width/2, true_dist, width=width, label="Ground Truth",
            alpha=0.8, color='#2E8B57')

    # 双生成器预测
    plt.bar(x + width/2, pred_dist, width=width, label="Dual Generator",
            alpha=0.8, color='#FF6B35')

    # Mayo-Lewis理论 - 改为线条图
    plt.plot(x, theory_dist, linewidth=2, label="Mayo-Lewis Theory",
             color='#4169E1', marker='o', markersize=4, alpha=0.8)

    # 可选：显示残差分布（根据之前的修改，这部分不会执行）
    if show_residual and residual_dist is not None:
        plt.plot(x, residual_dist, 'o--', linewidth=2, markersize=3,
                 color='#8B0000', label='Residual', alpha=0.8)

    plt.xlabel("Block Length", fontsize=12, fontweight='bold')
    plt.ylabel("Probability", fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.ylim(bottom=1e-4, top=1)

    if title:
        plt.title(title, fontsize=12, fontweight='bold')

    # 添加序列信息（根据之前的修改，当sequences为None时不显示）
    if sequences:
        seq_text = f"Sequences: {', '.join(sequences[:3])}"  # 只显示前3个序列
        if len(sequences) > 3:
            seq_text += f" (+{len(sequences) - 3} more)"
        plt.text(0.02, 0.98, seq_text, transform=plt.gca().transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_topk_grid(results: List[Dict], out_dir: Path, args, metric: str = 'kl_fused'):
    """生成最佳样本的网格图"""
    if not results:
        print("警告: 没有结果可生成网格图")
        return None

    # 按指标排序
    scores = [r[metric] for r in results]
    order = np.argsort(scores)
    top_k = min(args.top_k, len(results))
    chosen = [results[int(i)] for i in order[:top_k]]

    # 创建单个图像文件
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    image_paths = []

    for i, result in enumerate(chosen):
        sample_idx = result['sample_idx']
        title = f"Sample {sample_idx} (KL={result['kl_fused']:.4f})"
        img_path = overlay_dir / f"sample_{sample_idx:05d}.png"
        create_overlay_plot(
            result['true_dist'], result['pred_dist'], result['theory_dist'],
            None,  # 不传递sequences
            img_path, title=title,
            show_residual=False  # 不显示residual
        )
        image_paths.append(img_path)

    # 创建网格
    cols = 4
    rows = math.ceil(top_k / cols)
    grid_path = out_dir / f"top{top_k}_grid.pdf"
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), dpi=150)

    if rows == 1:
        axes = np.array([axes])

    for n in range(rows * cols):
        r = n // cols
        c = n % cols
        ax = axes[r, c]
        ax.axis("off")

        if n < top_k:
            img = plt.imread(image_paths[n])
            ax.imshow(img)

    # 添加图例（移除了sequence和residual相关的元素）
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#2E8B57', alpha=0.8, label='Ground Truth'),
        Rectangle((0, 0), 1, 1, facecolor='#FF6B35', alpha=0.8, label='Dual Generator'),
        plt.Line2D([0], [0], color='#4169E1', linewidth=2, label='Mayo-Lewis Theory')  # 改为线条
    ]

    figlegend = fig.legend(handles=legend_elements, loc='upper center',
                           bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=12)  # ncol改为3
    for text in figlegend.get_texts():
        text.set_weight('bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()

    return grid_path, chosen


def save_results_summary(results: List[Dict], out_dir: Path):
    """保存评估结果摘要"""

    # 计算统计信息
    kl_fused = [r['kl_fused'] for r in results]
    emd_fused = [r['emd_fused'] for r in results]
    js_fused = [r['js_fused'] for r in results]
    kl_theory = [r['kl_theory'] for r in results]
    emd_theory = [r['emd_theory'] for r in results]
    confidence = [r['confidence'] for r in results]
    quality = [r['quality_score'] for r in results]

    summary = {
        'metric': ['KL_fused', 'EMD_fused', 'JS_fused', 'KL_theory', 'EMD_theory',
                   'Confidence', 'Quality'],
        'mean': [
            np.mean(kl_fused), np.mean(emd_fused), np.mean(js_fused),
            np.mean(kl_theory), np.mean(emd_theory),
            np.mean(confidence), np.mean(quality)
        ],
        'std': [
            np.std(kl_fused), np.std(emd_fused), np.std(js_fused),
            np.std(kl_theory), np.std(emd_theory),
            np.std(confidence), np.std(quality)
        ],
        'median': [
            np.median(kl_fused), np.median(emd_fused), np.median(js_fused),
            np.median(kl_theory), np.median(emd_theory),
            np.median(confidence), np.median(quality)
        ]
    }

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(out_dir / 'evaluation_summary.csv', index=False)

    # 详细结果
    detailed_results = []
    for r in results:
        detailed_results.append({
            'sample_idx': r['sample_idx'],
            'kl_fused': r['kl_fused'],
            'emd_fused': r['emd_fused'],
            'js_fused': r['js_fused'],
            'kl_theory': r['kl_theory'],
            'emd_theory': r['emd_theory'],
            'confidence': r['confidence'],
            'quality_score': r['quality_score'],
            'num_sequences': len(r['sequences']),
            'improvement_over_theory': r['kl_theory'] - r['kl_fused']  # 正值表示改进
        })

    df_detailed = pd.DataFrame(detailed_results)
    df_detailed = df_detailed.sort_values('kl_fused')  # 按KL散度排序
    df_detailed.to_csv(out_dir / 'detailed_results.csv', index=False)

    print(f"\n评估结果摘要:")
    print(f"KL散度 (融合): {np.mean(kl_fused):.6f} ± {np.std(kl_fused):.6f}")
    print(f"KL散度 (理论): {np.mean(kl_theory):.6f} ± {np.std(kl_theory):.6f}")
    print(f"EMD (融合): {np.mean(emd_fused):.6f} ± {np.std(emd_fused):.6f}")
    print(f"EMD (理论): {np.mean(emd_theory):.6f} ± {np.std(emd_theory):.6f}")
    print(f"平均置信度: {np.mean(confidence):.4f}")
    print(f"平均质量评分: {np.mean(quality):.4f}")

    improvement = np.mean([r['improvement_over_theory'] for r in detailed_results])
    print(f"相对理论的改进: {improvement:.6f} (正值表示改进)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="双生成器模型评估")

    # 数据参数
    parser.add_argument('--data_path', type=str, default='../data/copolymer.csv',
                        help='测试数据CSV路径')
    parser.add_argument('--model_path', type=str, default='../dual_generator_checkpoints/best_model.pt',
                        help='双生成器模型权重路径')
    parser.add_argument('--out_dir', type=str, default='dual_gen_evaluation',
                        help='输出目录')

    # 模型参数
    parser.add_argument('--bins', type=int, default=50,
                        help='分布bins数量')
    parser.add_argument('--condition_dim', type=int, default=17,
                        help='条件特征维度')
    parser.add_argument('--cond_dim', type=int, default=128,
                        help='条件编码器维度')
    parser.add_argument('--residual_dim', type=int, default=256,
                        help='残差生成器维度')
    parser.add_argument('--residual_layers', type=int, default=8,
                        help='残差生成器层数')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='扩散步数')

    # 推理参数
    parser.add_argument('--sample_steps', type=int, default=50,
                        help='采样步数')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='引导强度')
    parser.add_argument('--inference_temp', type=float, default=1.0,
                        help='推理温度')
    parser.add_argument('--fusion_strategy', type=str, default='adaptive',
                        choices=['adaptive', 'weighted', 'theory_only', 'residual_only'],
                        help='融合策略')

    # 评估参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大评估样本数')
    parser.add_argument('--test_split', type=str, default='test',
                        help='测试集划分')
    parser.add_argument('--top_k', type=int, default=12,
                        help='选择最佳K个样本')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cuda/cpu/auto)')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"使用设备: {device}")

    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    print(f"加载测试数据: {args.data_path}")
    test_dataset = PolyDataset(
        csv_path=args.data_path,
        max_length=args.bins,
        max_samples=args.max_samples,
        split=args.test_split,
        test_ratio=0.2,
        val_ratio=0.1
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fixed_poly,
        num_workers=0
    )

    print(f"测试样本数: {len(test_dataset)}")

    # 加载模型
    model = load_dual_generator_model(args.model_path, device, args)

    # 评估模型
    print(f"开始模型评估...")
    results = evaluate_dual_generator(model, test_loader, device, args)

    # 保存结果
    save_results_summary(results, out_dir)

    # 生成可视化
    print(f"生成最佳样本可视化...")
    grid_path, chosen = generate_topk_grid(results, out_dir, args, metric='kl_fused')

    if grid_path:
        print(f"网格图保存至: {grid_path}")

    print(f"评估完成! 结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
