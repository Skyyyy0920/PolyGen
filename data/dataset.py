#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双生成器聚合物数据集

支持原始PolyGen-F06C数据格式，并扩展支持双生成器训练
"""

import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

from .mayo_lewis import MayoLewisCalculator


class DualPolyDataset(Dataset):
    """
    双生成器聚合物数据集
    
    支持：
    1. 加载PolyGen-F06C格式的copolymer.csv数据
    2. 提取Mayo-Lewis理论分布
    3. 计算残差目标（实际分布 - 理论分布）
    4. 条件特征提取和编码
    """
    
    def __init__(self, 
                 csv_path: str,
                 max_length: int = 50,
                 max_samples: Optional[int] = None,
                 split: str = 'train',
                 test_ratio: float = 0.2,
                 val_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        初始化数据集
        
        Args:
            csv_path: copolymer.csv文件路径
            max_length: 最大块长度
            max_samples: 最大样本数（用于调试）
            split: 数据分割 ('train', 'val', 'test')
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            random_seed: 随机种子
        """
        self.csv_path = Path(csv_path)
        self.max_length = max_length
        self.split = split
        
        # 初始化Mayo-Lewis计算器
        self.mayo_lewis_calc = MayoLewisCalculator(max_length=max_length)
        
        # 加载和预处理数据
        self.data = self._load_data(max_samples)
        
        # 数据分割
        self.data = self._split_data(test_ratio, val_ratio, random_seed)
        
        print(f"📊 DualPolyDataset初始化完成:")
        print(f"  数据文件: {self.csv_path}")
        print(f"  分割: {self.split}")
        print(f"  样本数: {len(self.data)}")
        print(f"  最大块长度: {self.max_length}")
    
    def _load_data(self, max_samples: Optional[int]) -> List[Dict]:
        """加载和预处理数据"""
        print(f"📂 加载数据: {self.csv_path}")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.csv_path}")
        
        # 读取CSV
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(self.csv_path, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(self.csv_path, encoding='latin-1')
        
        # 删除完全为空的行
        df = df.dropna(how='all')
        
        if max_samples:
            df = df.head(max_samples)
            print(f"  限制样本数: {max_samples}")
        
        print(f"  原始样本数: {len(df)}")
        print(f"  数据列: {list(df.columns)}")
        
        if len(df) == 0:
            raise ValueError("数据文件中没有有效数据行")
        
        # 预处理数据
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # 跳过空行或无效行
                if pd.isna(row.get('seq')) or pd.isna(row.get('block_dist')):
                    continue
                
                # 解析序列数据
                sequences = self._parse_sequences(row['seq'])
                if not sequences:
                    continue
                
                # 解析块长度分布
                block_dist = self._parse_block_distribution(row['block_dist'])
                if block_dist is None or len(block_dist) == 0:
                    continue
                
                # 确保分布长度匹配
                if len(block_dist) > self.max_length:
                    block_dist = block_dist[:self.max_length]
                elif len(block_dist) < self.max_length:
                    # 用零填充
                    padding = np.zeros(self.max_length - len(block_dist))
                    block_dist = np.concatenate([block_dist, padding])
                
                # 归一化分布
                if np.sum(block_dist) > 0:
                    block_dist = block_dist / np.sum(block_dist)
                
                # 提取条件特征
                condition_features = self._extract_condition_features(row, sequences)
                
                # 计算Mayo-Lewis理论分布
                theoretical_dist = self.mayo_lewis_calc.calculate_theoretical_distribution(sequences)
                
                # 计算残差目标
                residual_target = block_dist - theoretical_dist
                
                # 构造样本
                sample = {
                    'idx': idx,
                    'sequences': sequences,
                    'block_distribution': block_dist.astype(np.float32),
                    'theoretical_distribution': theoretical_dist.astype(np.float32),
                    'residual_target': residual_target.astype(np.float32),
                    'condition_features': condition_features.astype(np.float32),
                    'mayo_lewis_params': self.mayo_lewis_calc.extract_sequence_statistics(sequences),
                    'metadata': {
                        'name': row.get('name', f'sample_{idx}'),
                        'activation': row.get('activation', 0),
                        'temp': row.get('Temp', 0),
                        'prob_AA': row.get('probAA', 0),
                        'prob_BB': row.get('probBB', 0),
                    }
                }
                
                processed_data.append(sample)
                
            except Exception as e:
                print(f"  警告: 样本{idx}处理失败: {e}")
                continue
        
        print(f"  有效样本数: {len(processed_data)}")
        return processed_data
    
    def _parse_sequences(self, seq_str: str) -> List[str]:
        """解析序列字符串"""
        try:
            if isinstance(seq_str, str):
                # 尝试解析为Python列表
                sequences = ast.literal_eval(seq_str)
                if isinstance(sequences, list):
                    return [str(seq) for seq in sequences if seq and isinstance(seq, str)]
            return []
        except:
            return []
    
    def _parse_block_distribution(self, block_dist_str: str) -> Optional[np.ndarray]:
        """解析块长度分布"""
        try:
            if isinstance(block_dist_str, str):
                # 解析为numpy数组
                parsed = ast.literal_eval(block_dist_str)
                if isinstance(parsed, list) and len(parsed) == 2:
                    # 格式: [lengths_array, probs_array]
                    lengths, probs = parsed
                    if isinstance(probs, (list, np.ndarray)):
                        return np.array(probs, dtype=np.float32)
            return None
        except:
            return None
    
    def _extract_condition_features(self, row: pd.Series, sequences: List[str]) -> np.ndarray:
        """提取条件特征"""
        # 基础实验条件
        features = [
            float(row.get('activation', 0)),  # 活化能
            float(row.get('Temp', 0)),        # 温度
            float(row.get('probAA', 0)),      # AA转换概率
            float(row.get('probBB', 0)),      # BB转换概率
            float(row.get('probAABB', 0)),    # AABB转换概率
            float(row.get('probAB', 0)),      # AB转换概率
        ]
        
        # 序列统计特征
        if sequences:
            all_seq = ''.join(sequences)
            seq_length = len(all_seq)
            f_A = all_seq.count('A') / seq_length if seq_length > 0 else 0.5
            
            # 转换频率（复杂度指标）
            transitions = 0
            for seq in sequences:
                for i in range(len(seq) - 1):
                    if seq[i] != seq[i + 1]:
                        transitions += 1
            
            transition_freq = transitions / max(1, seq_length - len(sequences))
            
            features.extend([
                f_A,               # 单体A摩尔分数
                1 - f_A,          # 单体B摩尔分数  
                seq_length,       # 总序列长度
                len(sequences),   # 序列数量
                transition_freq,  # 转换频率
            ])
        else:
            features.extend([0.5, 0.5, 0, 0, 0])
        
        # 物理化学参数
        features.extend([
            float(row.get('epsAA', 0)),      # AA相互作用能
            float(row.get('epsAB', 0)),      # AB相互作用能
            float(row.get('epsBB', 0)),      # BB相互作用能
            float(row.get('damp', 0)),       # 阻尼系数
            float(row.get('angleA', 0)),     # A角度
            float(row.get('angleB', 0)),     # B角度
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _split_data(self, test_ratio: float, val_ratio: float, random_seed: int) -> List[Dict]:
        """数据分割"""
        np.random.seed(random_seed)
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
        """获取单个样本"""
        sample = self.data[idx]
        
        return {
            'idx': sample['idx'],
            'sequences': sample['sequences'],
            'block_distribution': torch.tensor(sample['block_distribution']),
            'theoretical_distribution': torch.tensor(sample['theoretical_distribution']),
            'residual_target': torch.tensor(sample['residual_target']),
            'condition_features': torch.tensor(sample['condition_features']),
            'mayo_lewis_params': sample['mayo_lewis_params'],
            'metadata': sample['metadata']
        }
    
    def get_feature_dim(self) -> int:
        """获取条件特征维度"""
        if len(self.data) > 0:
            return len(self.data[0]['condition_features'])
        return 17  # 默认特征维度
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if not self.data:
            return {}
        
        block_dists = np.stack([sample['block_distribution'] for sample in self.data])
        theoretical_dists = np.stack([sample['theoretical_distribution'] for sample in self.data])
        residuals = np.stack([sample['residual_target'] for sample in self.data])
        
        return {
            'n_samples': len(self.data),
            'max_length': self.max_length,
            'feature_dim': self.get_feature_dim(),
            'block_dist_stats': {
                'mean': float(np.mean(block_dists)),
                'std': float(np.std(block_dists)),
                'min': float(np.min(block_dists)),
                'max': float(np.max(block_dists))
            },
            'theoretical_dist_stats': {
                'mean': float(np.mean(theoretical_dists)),
                'std': float(np.std(theoretical_dists)),
                'min': float(np.min(theoretical_dists)),
                'max': float(np.max(theoretical_dists))
            },
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            }
        }


def collate_dual_poly(batch: List[Dict]) -> Dict:
    """
    双生成器数据集的collate函数
    
    Args:
        batch: 批量样本列表
        
    Returns:
        批量处理后的数据字典
    """
    # 提取各个字段
    indices = [sample['idx'] for sample in batch]
    sequences_batch = [sample['sequences'] for sample in batch]
    
    # 堆叠张量数据
    block_distributions = torch.stack([sample['block_distribution'] for sample in batch])
    theoretical_distributions = torch.stack([sample['theoretical_distribution'] for sample in batch])
    residual_targets = torch.stack([sample['residual_target'] for sample in batch])
    condition_features = torch.stack([sample['condition_features'] for sample in batch])
    
    # 收集元数据
    mayo_lewis_params = [sample['mayo_lewis_params'] for sample in batch]
    metadata = [sample['metadata'] for sample in batch]
    
    return {
        'indices': indices,
        'sequences': sequences_batch,
        'block_distributions': block_distributions,
        'theoretical_distributions': theoretical_distributions,
        'residual_targets': residual_targets,
        'condition_features': condition_features,
        'mayo_lewis_params': mayo_lewis_params,
        'metadata': metadata
    }


def test_dataset():
    """测试数据集功能"""
    print("🧪 测试DualPolyDataset...")
    
    # 使用真实数据路径
    csv_path = "PolyGen-F06C/data/copolymer.csv"
    
    try:
        # 创建数据集
        dataset = DualPolyDataset(
            csv_path=csv_path,
            max_length=30,
            max_samples=100,  # 限制样本数用于测试
            split='train'
        )
        
        print(f"数据集统计:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试数据加载
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=collate_dual_poly
        )
        
        # 测试一个批次
        batch = next(iter(dataloader))
        
        print(f"\n批次测试:")
        print(f"  批次大小: {len(batch['indices'])}")
        print(f"  块分布形状: {batch['block_distributions'].shape}")
        print(f"  理论分布形状: {batch['theoretical_distributions'].shape}")
        print(f"  残差目标形状: {batch['residual_targets'].shape}")
        print(f"  条件特征形状: {batch['condition_features'].shape}")
        
        # 检查数据质量
        print(f"\n数据质量检查:")
        print(f"  块分布求和: {torch.sum(batch['block_distributions'], dim=1)[:3]}")
        print(f"  理论分布求和: {torch.sum(batch['theoretical_distributions'], dim=1)[:3]}")
        print(f"  残差范围: [{torch.min(batch['residual_targets']):.4f}, {torch.max(batch['residual_targets']):.4f}]")
        
        print("✅ 数据集测试通过!")
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()
