#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from collections import Counter


def extract_target_block_distribution(polymer_chains, max_length=50):
    """Extract the TARGET block distribution that we want to predict"""
    all_block_lengths = []

    for chain in polymer_chains:
        if chain:
            current_length = 1
            current_monomer = chain[0]

            for i in range(1, len(chain)):
                if chain[i] == current_monomer:
                    current_length += 1
                else:
                    all_block_lengths.append(current_length)
                    current_monomer = chain[i]
                    current_length = 1

            all_block_lengths.append(current_length)

    counter = Counter(all_block_lengths)
    lengths = np.arange(1, max_length + 1)
    probs = np.array([counter.get(length, 0) for length in lengths], dtype=np.float32)

    return probs / probs.sum()


def extract_input_features(polymer_chains):
    total_monomers = sum(len(chain) for chain in polymer_chains)
    all_monomers = ''.join(polymer_chains)

    all_block_lengths = []

    for chain in polymer_chains:
        if chain:
            current_length = 1
            current_monomer = chain[0]

            for i in range(1, len(chain)):
                if chain[i] == current_monomer:
                    current_length += 1
                else:
                    all_block_lengths.append(current_length)
                    current_monomer = chain[i]
                    current_length = 1

            all_block_lengths.append(current_length)

    features = []

    f_A = all_monomers.count('A') / len(all_monomers) if all_monomers else 0
    f_B = 1 - f_A
    features.extend([f_A, f_B])  # 聚合物序列中单体 A 和 B 的摩尔分数

    all_pairs = []
    for chain in polymer_chains:
        if len(chain) >= 2:
            chain_pairs = [chain[i:i + 2] for i in range(len(chain) - 1)]
            all_pairs.extend(chain_pairs)

    pair_counts = Counter(all_pairs)
    total_pairs = len(all_pairs)

    p_AA = pair_counts['AA'] / total_pairs if total_pairs > 0 else 0
    p_BB = pair_counts['BB'] / total_pairs if total_pairs > 0 else 0
    p_AB = pair_counts['AB'] / total_pairs if total_pairs > 0 else 0
    p_BA = pair_counts['BA'] / total_pairs if total_pairs > 0 else 0

    features.extend([p_AA, p_BB, p_AB, p_BA])  # 相邻二元组转移概率

    block_array = np.array(all_block_lengths)
    features.extend([
        np.mean(block_array),
        np.std(block_array),
        np.min(block_array),
        np.max(block_array),
        np.median(block_array),
    ])

    features.extend([
        total_monomers,
        len(polymer_chains),
        total_monomers / len(polymer_chains) if len(polymer_chains) > 0 else 0,  # 平均序列长度
    ])

    return np.array(features[:17], dtype=np.float32)


class PolyDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 max_length: int = 50,
                 max_samples: Optional[int] = None,
                 split: str = 'train',
                 test_ratio: float = 0.2,
                 val_ratio: float = 0.1):

        self.csv_path = Path(csv_path)
        self.max_length = max_length
        self.split = split

        self.data = self._load_and_process_data(max_samples)
        self.data = self._split_data(test_ratio, val_ratio)

        print(f"Dataset length: {len(self.data)}")

    def _load_and_process_data(self, max_samples: Optional[int]) -> List[Dict]:
        df = pd.read_csv(self.csv_path)
        df = df.dropna(how='all')
        print(f"{df.shape}")

        if max_samples and max_samples < len(df):
            df = df.head(max_samples)

        processed_data = []
        for idx, row in df.iterrows():
            seq_str = str(row['seq']).strip()
            sequences = ast.literal_eval(seq_str)
            sequences = [s.replace('c', '') for s in sequences]

            input_features = extract_input_features(sequences)
            target_block_dist = extract_target_block_distribution(sequences, self.max_length)

            condition_features = self._extract_condition_features(row)

            # 计算简化的理论分布（Mayo-Lewis） TODO：检查
            theoretical_dist = self._compute_mayo_lewis_distribution(input_features)

            # 计算残差  TODO：这里可以在这直接计算残差吗？
            residual = target_block_dist - theoretical_dist

            # import matplotlib.pyplot as plt
            # import numpy as np
            #
            # # 假设你已有这三个变量（50维）
            # # residual = target_block_dist - theoretical_dist
            #
            # # 创建 x 轴：块长度 1 到 50
            # x = np.arange(1, len(target_block_dist) + 1)  # [1, 2, ..., 50]
            #
            # plt.figure(figsize=(12, 6))
            #
            # # 画目标分布
            # plt.bar(x - 0.2, target_block_dist, width=0.4, label='Target Distribution', color='blue', alpha=0.7)
            #
            # # 画理论分布
            # plt.bar(x + 0.2, theoretical_dist, width=0.4, label='Theoretical Distribution', color='orange', alpha=0.7)
            #
            # # 画残差（用折线图或带标记的点更清晰）
            # plt.plot(x, residual, label='Residual (Target - Theoretical)', color='red', marker='o', linestyle='-',
            #          linewidth=2, markersize=4)
            #
            # # 添加细节
            # plt.xlabel('Block Length', fontsize=12)
            # plt.ylabel('Probability / Residual', fontsize=12)
            # plt.title('Comparison of Target, Theoretical and Residual Distributions', fontsize=14)
            # plt.legend()
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            # plt.xticks(x[::2])  # 每隔一个显示 x 轴标签，避免拥挤
            # plt.tight_layout()
            #
            # plt.show()

            sample = {
                'idx': len(processed_data),
                'sequences': sequences,
                'block_distribution': target_block_dist,
                'theoretical_distribution': theoretical_dist,
                'residual_target': residual,
                'condition_features': condition_features,
                'input_features': input_features,
                'metadata': {
                    'original_idx': idx,
                    'name': str(row.get('name', f'sample_{idx}')),
                    'temp': float(row.get('Temp', 0)),
                    'activation': float(row.get('activation', 0))
                }
            }

            processed_data.append(sample)

        if len(processed_data) == 0:
            raise ValueError("没有成功处理的样本，请检查数据格式")

        return processed_data

    def _extract_condition_features(self, row: pd.Series) -> np.ndarray:
        multi_val_features = ['size', 'epsAA', 'epsAB', 'epsBB', 'damp', 'angleKA', 'angleKB',
                              'activationAA', 'activationBB', 'activationAB']
        single_val_features = ['Nmono', 'angleA', 'angleB', 'shiftA', 'shiftB', 'Temp', 'epshard']

        features = []
        for feat in multi_val_features + single_val_features:
            val = row.get(feat, 0)
            if pd.isna(val):
                val = 0
            features.append(float(val))

        return np.array(features, dtype=np.float32)

    def _compute_mayo_lewis_distribution(self, input_features: np.ndarray) -> np.ndarray:
        """计算简化的Mayo-Lewis理论分布"""
        # 从输入特征中提取关键参数
        f_A = input_features[0] if len(input_features) > 0 else 0.5
        p_AA = input_features[2] if len(input_features) > 2 else 0.5
        p_BB = input_features[3] if len(input_features) > 3 else 0.5

        # 简化的Mayo-Lewis计算
        # 这里使用一个简化的指数分布作为理论基础
        lengths = np.arange(1, self.max_length + 1)

        # 基于p_AA和p_BB计算平均块长度
        mean_A = 1 / (1 - p_AA + 1e-8)
        mean_B = 1 / (1 - p_BB + 1e-8)
        overall_mean = f_A * mean_A + (1 - f_A) * mean_B

        # 指数分布
        lambda_param = 1 / max(overall_mean, 1e-8)
        theoretical = np.exp(-lambda_param * (lengths - 1))

        # 归一化
        if np.sum(theoretical) > 0:
            theoretical = theoretical / np.sum(theoretical)
        else:
            theoretical = np.ones(self.max_length) / self.max_length

        return theoretical.astype(np.float32)

    def _split_data(self, test_ratio: float, val_ratio: float) -> List[Dict]:
        """数据分割"""
        np.random.seed(42)
        n_total = len(self.data)
        indices = np.random.permutation(n_total)

        n_test = int(test_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_test - n_val

        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"未知的split: {self.split}")

        return [self.data[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]

        return {
            'idx': sample['idx'],
            'sequences': sample['sequences'],
            'block_distributions': torch.tensor(sample['block_distribution']),
            'theoretical_distributions': torch.tensor(sample['theoretical_distribution']),
            'residual_targets': torch.tensor(sample['residual_target']),
            'condition_features': torch.tensor(sample['condition_features']),
            'input_features': torch.tensor(sample['input_features']),
            'metadata': sample['metadata']
        }


def collate_fixed_poly(batch: List[Dict]) -> Dict:
    """修复的collate函数"""
    return {
        'indices': [sample['idx'] for sample in batch],
        'sequences': [sample['sequences'] for sample in batch],
        'block_distributions': torch.stack([sample['block_distributions'] for sample in batch]),
        'theoretical_distributions': torch.stack([sample['theoretical_distributions'] for sample in batch]),
        'residual_targets': torch.stack([sample['residual_targets'] for sample in batch]),
        'condition_features': torch.stack([sample['condition_features'] for sample in batch]),
        'input_features': torch.stack([sample['input_features'] for sample in batch]),
        'metadata': [sample['metadata'] for sample in batch]
    }
