#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据集实现，专门处理PolyGen-F06C数据格式
"""

import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


class SimplePolyDataset(Dataset):
    """简化的聚合物数据集，专门处理CSV格式问题"""
    
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
        
        # 加载和处理数据
        self.data = self._load_and_process_data(max_samples)
        
        # 数据分割
        self.data = self._split_data(test_ratio, val_ratio)
        
        print(f"SimplePolyDataset初始化完成: {self.split}, 样本数: {len(self.data)}")
    
    def _load_and_process_data(self, max_samples: Optional[int]) -> List[Dict]:
        """加载和处理数据"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.csv_path}")
        
        # 读取CSV，处理编码问题
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(self.csv_path, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(self.csv_path, encoding='latin-1')
        
        print(f"原始数据形状: {df.shape}")
        
        # 删除完全空的行
        df = df.dropna(how='all')
        print(f"清理后数据形状: {df.shape}")
        
        if len(df) == 0:
            raise ValueError("数据文件中没有有效数据")
        
        # 限制样本数
        if max_samples and max_samples < len(df):
            df = df.head(max_samples)
        
        # 处理数据
        processed_data = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 检查必需字段
                if pd.isna(row.get('seq')) or pd.isna(row.get('block_dist')):
                    failed_count += 1
                    continue
                
                # 解析序列
                seq_str = str(row['seq']).strip()
                if not seq_str or seq_str == 'nan':
                    failed_count += 1
                    continue
                
                sequences = ast.literal_eval(seq_str)
                if not isinstance(sequences, list) or len(sequences) == 0:
                    failed_count += 1
                    continue
                
                # 过滤无效序列
                valid_sequences = []
                for seq in sequences:
                    if isinstance(seq, str) and seq.strip() and all(c in 'AB' for c in seq.strip()):
                        valid_sequences.append(seq.strip())
                
                if len(valid_sequences) == 0:
                    failed_count += 1
                    continue
                
                # 解析块长度分布
                block_dist_str = str(row['block_dist']).strip()
                if not block_dist_str or block_dist_str == 'nan':
                    failed_count += 1
                    continue
                
                try:
                    # 处理包含numpy数组的字符串
                    # 替换numpy array表示为标准列表
                    block_dist_str = block_dist_str.replace('array(', '[').replace(')', ']')
                    # 处理可能的多行格式
                    block_dist_str = ' '.join(block_dist_str.split())
                    
                    parsed_dist = ast.literal_eval(block_dist_str)
                    if not isinstance(parsed_dist, list) or len(parsed_dist) != 2:
                        failed_count += 1
                        continue
                    
                    lengths, probs = parsed_dist
                    
                    # 转换为数值列表
                    if isinstance(lengths, list):
                        lengths = [float(x) for x in lengths if isinstance(x, (int, float))]
                    if isinstance(probs, list):
                        probs = [float(x) for x in probs if isinstance(x, (int, float))]
                    
                    if not probs or len(probs) == 0:
                        failed_count += 1
                        continue
                        
                except Exception as parse_error:
                    failed_count += 1
                    continue
                
                # 转换为numpy数组
                probs = np.array(probs, dtype=np.float32)
                
                # 调整长度
                if len(probs) > self.max_length:
                    probs = probs[:self.max_length]
                elif len(probs) < self.max_length:
                    padding = np.zeros(self.max_length - len(probs), dtype=np.float32)
                    probs = np.concatenate([probs, padding])
                
                # 归一化
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                
                # 提取基本特征
                features = self._extract_basic_features(row, valid_sequences)
                
                # 计算理论分布（简化版）
                theoretical_dist = self._compute_simple_theoretical(valid_sequences)
                
                # 计算残差
                residual = probs - theoretical_dist
                
                # 构造样本
                sample = {
                    'idx': len(processed_data),
                    'sequences': valid_sequences,
                    'block_distribution': probs,
                    'theoretical_distribution': theoretical_dist,
                    'residual_target': residual,
                    'condition_features': features,
                    'metadata': {
                        'original_idx': idx,
                        'name': str(row.get('name', f'sample_{idx}')),
                        'temp': float(row.get('Temp', 0)),
                        'activation': float(row.get('activation', 0))
                    }
                }
                
                processed_data.append(sample)
                
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"成功处理样本: {len(processed_data)}, 失败: {failed_count}")
        
        if len(processed_data) == 0:
            raise ValueError("没有成功处理的样本，请检查数据格式")
        
        return processed_data
    
    def _extract_basic_features(self, row: pd.Series, sequences: List[str]) -> np.ndarray:
        """提取基本特征"""
        # 基础实验条件
        features = [
            float(row.get('activation', 0)),
            float(row.get('Temp', 0)),
            float(row.get('probAA', 0)),
            float(row.get('probBB', 0)),
            float(row.get('probAABB', 0)),
            float(row.get('probAB', 0)),
        ]
        
        # 序列统计
        if sequences:
            all_seq = ''.join(sequences)
            seq_len = len(all_seq)
            if seq_len > 0:
                f_A = all_seq.count('A') / seq_len
                features.extend([
                    f_A,
                    1 - f_A,
                    seq_len,
                    len(sequences),
                    seq_len / len(sequences) if len(sequences) > 0 else 0  # 平均序列长度
                ])
            else:
                features.extend([0.5, 0.5, 0, 0, 0])
        else:
            features.extend([0.5, 0.5, 0, 0, 0])
        
        # 物理化学参数
        features.extend([
            float(row.get('epsAA', 0)),
            float(row.get('epsAB', 0)),
            float(row.get('epsBB', 0)),
            float(row.get('damp', 0)),
            float(row.get('angleA', 0)),
            float(row.get('angleB', 0)),
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _compute_simple_theoretical(self, sequences: List[str]) -> np.ndarray:
        """计算简化的理论分布"""
        if not sequences:
            # 返回均匀分布
            dist = np.ones(self.max_length, dtype=np.float32)
            return dist / np.sum(dist)
        
        # 统计实际块长度
        all_blocks = []
        for seq in sequences:
            if not seq:
                continue
            
            current_block = seq[0]
            block_length = 1
            
            for i in range(1, len(seq)):
                if seq[i] == current_block:
                    block_length += 1
                else:
                    all_blocks.append(block_length)
                    current_block = seq[i]
                    block_length = 1
            
            all_blocks.append(block_length)
        
        # 构建分布
        dist = np.zeros(self.max_length, dtype=np.float32)
        
        if all_blocks:
            for length in all_blocks:
                if 1 <= length <= self.max_length:
                    dist[length - 1] += 1
        
        # 归一化
        if np.sum(dist) > 0:
            dist = dist / np.sum(dist)
        else:
            dist = np.ones(self.max_length, dtype=np.float32) / self.max_length
        
        return dist
    
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
            'metadata': sample['metadata']
        }


def collate_simple_poly(batch: List[Dict]) -> Dict:
    """简化的collate函数"""
    return {
        'indices': [sample['idx'] for sample in batch],
        'sequences': [sample['sequences'] for sample in batch],
        'block_distributions': torch.stack([sample['block_distributions'] for sample in batch]),
        'theoretical_distributions': torch.stack([sample['theoretical_distributions'] for sample in batch]),
        'residual_targets': torch.stack([sample['residual_targets'] for sample in batch]),
        'condition_features': torch.stack([sample['condition_features'] for sample in batch]),
        'metadata': [sample['metadata'] for sample in batch]
    }
