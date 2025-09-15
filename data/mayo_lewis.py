#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Dict, List, Union
from collections import Counter


class MayoLewisCalculator:
    """
    Mayo-Lewis理论直接计算器
    
    基于经典Mayo-Lewis共聚理论，从聚合物序列统计量
    直接计算理论块长度分布
    """
    
    def __init__(self, max_length: int = 50):
        self.max_length = max_length
    
    def extract_sequence_statistics(self, sequences: List[str]) -> Dict[str, float]:
        """
        从聚合物序列中直接提取统计量
        
        Args:
            sequences: 聚合物序列列表，如 ['AAABBB', 'BBAABB']
            
        Returns:
            统计参数字典，包含f_A, p_AA, p_BB等
        """
        if not sequences:
            return self._get_default_params()
        
        # 合并所有序列
        all_monomers = ''.join(sequences)
        if not all_monomers:
            return self._get_default_params()
        
        # 计算单体组成
        count_A = all_monomers.count('A')
        count_B = all_monomers.count('B')
        total_monomers = len(all_monomers)
        
        f_A = count_A / total_monomers if total_monomers > 0 else 0.5
        f_B = 1 - f_A
        
        # 计算转换概率
        pair_counts = Counter()
        for seq in sequences:
            if len(seq) >= 2:
                pairs = [seq[i:i+2] for i in range(len(seq)-1)]
                pair_counts.update(pairs)
        
        total_pairs = sum(pair_counts.values())
        if total_pairs == 0:
            return self._get_default_params()
        
        p_AA = pair_counts.get('AA', 0) / total_pairs
        p_BB = pair_counts.get('BB', 0) / total_pairs
        p_AB = pair_counts.get('AB', 0) / total_pairs
        p_BA = pair_counts.get('BA', 0) / total_pairs
        
        return {
            'f_A': f_A,
            'f_B': f_B,
            'p_AA': p_AA,
            'p_BB': p_BB,
            'p_AB': p_AB,
            'p_BA': p_BA,
            'total_monomers': total_monomers,
            'total_pairs': total_pairs
        }
    
    def _get_default_params(self) -> Dict[str, float]:
        """返回默认参数"""
        return {
            'f_A': 0.5, 'f_B': 0.5,
            'p_AA': 0.3, 'p_BB': 0.3,
            'p_AB': 0.2, 'p_BA': 0.2,
            'total_monomers': 0, 'total_pairs': 0
        }
    
    def calculate_reactivity_ratios(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        根据Mayo-Lewis理论估计反应性比
        
        Args:
            params: 序列统计参数
            
        Returns:
            反应性比 r_A, r_B
        """
        f_A = params['f_A']
        f_B = params['f_B']
        p_AA = params['p_AA']
        p_BB = params['p_BB']
        
        # 计算交叉转换概率
        p_AB = 1 - p_AA - p_BB
        p_AB = max(0.01, min(0.98, p_AB))  # 数值稳定性
        
        # Mayo-Lewis反应性比估计
        if p_AB > 0 and f_A > 0 and f_B > 0:
            r_A = (p_AA / (p_AB / 2)) * (f_B / f_A)
            r_B = (p_BB / (p_AB / 2)) * (f_A / f_B)
        else:
            r_A = r_B = 1.0
        
        # 确保反应性比在合理范围内
        r_A = max(0.01, min(10.0, r_A))
        r_B = max(0.01, min(10.0, r_B))
        
        return {'r_A': r_A, 'r_B': r_B, 'p_AB_calc': p_AB}
    
    def calculate_continuation_probabilities(self, params: Dict[str, float], 
                                           reactivity_ratios: Dict[str, float]) -> Dict[str, float]:
        """
        计算块连续概率
        
        Args:
            params: 序列统计参数
            reactivity_ratios: 反应性比
            
        Returns:
            连续概率 p_A_continue, p_B_continue
        """
        f_A = params['f_A']
        f_B = params['f_B']
        r_A = reactivity_ratios['r_A']
        r_B = reactivity_ratios['r_B']
        
        # Mayo-Lewis连续概率公式
        denominator_A = r_A * f_A + f_B
        denominator_B = r_B * f_B + f_A
        
        p_A_continue = (r_A * f_A / denominator_A) if denominator_A > 0 else 0
        p_B_continue = (r_B * f_B / denominator_B) if denominator_B > 0 else 0
        
        # 数值稳定性
        p_A_continue = max(0, min(0.99, p_A_continue))
        p_B_continue = max(0, min(0.99, p_B_continue))
        
        return {
            'p_A_continue': p_A_continue,
            'p_B_continue': p_B_continue
        }
    
    def calculate_geometric_distribution(self, p_continue: float, max_length: int) -> np.ndarray:
        """
        计算几何分布
        
        Args:
            p_continue: 连续概率
            max_length: 最大块长度
            
        Returns:
            几何分布概率数组
        """
        lengths = np.arange(1, max_length + 1)
        
        if p_continue > 0:
            # 几何分布：P(X=k) = (1-p) * p^(k-1)
            probs = (1 - p_continue) * (p_continue ** (lengths - 1))
        else:
            # 如果连续概率为0，所有块长度为1
            probs = np.zeros(max_length)
            probs[0] = 1.0
        
        return probs
    
    def calculate_theoretical_distribution(self, sequences: List[str]) -> np.ndarray:
        """
        计算Mayo-Lewis理论分布
        
        Args:
            sequences: 聚合物序列列表
            
        Returns:
            理论块长度分布 [max_length,]
        """
        # 1. 提取序列统计
        params = self.extract_sequence_statistics(sequences)
        
        # 2. 计算反应性比
        reactivity_ratios = self.calculate_reactivity_ratios(params)
        
        # 3. 计算连续概率
        continuation_probs = self.calculate_continuation_probabilities(params, reactivity_ratios)
        
        # 4. 计算A和B的几何分布
        p_A_dist = self.calculate_geometric_distribution(
            continuation_probs['p_A_continue'], self.max_length
        )
        p_B_dist = self.calculate_geometric_distribution(
            continuation_probs['p_B_continue'], self.max_length
        )
        
        # 5. 按组成加权组合
        f_A = params['f_A']
        f_B = params['f_B']
        combined_dist = f_A * p_A_dist + f_B * p_B_dist
        
        # 6. 归一化
        total = np.sum(combined_dist)
        if total > 0:
            combined_dist = combined_dist / total
        
        return combined_dist.astype(np.float32)
    
    def batch_calculate_distributions(self, sequences_batch: List[List[str]]) -> torch.Tensor:
        """
        批量计算理论分布
        
        Args:
            sequences_batch: 批量序列，每个元素是一个序列列表
            
        Returns:
            批量理论分布 [batch_size, max_length]
        """
        distributions = []
        
        for sequences in sequences_batch:
            dist = self.calculate_theoretical_distribution(sequences)
            distributions.append(dist)
        
        return torch.tensor(np.stack(distributions), dtype=torch.float32)
    
    def get_detailed_analysis(self, sequences: List[str]) -> Dict:
        """
        获取详细的Mayo-Lewis分析结果
        
        Args:
            sequences: 聚合物序列列表
            
        Returns:
            详细分析结果
        """
        params = self.extract_sequence_statistics(sequences)
        reactivity_ratios = self.calculate_reactivity_ratios(params)
        continuation_probs = self.calculate_continuation_probabilities(params, reactivity_ratios)
        theoretical_dist = self.calculate_theoretical_distribution(sequences)
        
        return {
            'sequence_statistics': params,
            'reactivity_ratios': reactivity_ratios,
            'continuation_probabilities': continuation_probs,
            'theoretical_distribution': theoretical_dist,
            'distribution_stats': {
                'peak_position': int(np.argmax(theoretical_dist)) + 1,
                'peak_value': float(np.max(theoretical_dist)),
                'mean_length': float(np.sum(theoretical_dist * np.arange(1, len(theoretical_dist) + 1))),
                'entropy': -float(np.sum(theoretical_dist * np.log(theoretical_dist + 1e-8)))
            }
        }


def validate_mayo_lewis_calculation():
    """验证Mayo-Lewis计算的正确性"""
    print("🧪 验证Mayo-Lewis计算器...")
    
    calc = MayoLewisCalculator(max_length=20)
    
    # 测试用例1：交替共聚物
    alternating_seq = ['ABABABABAB', 'BABABABABA']
    result1 = calc.get_detailed_analysis(alternating_seq)
    
    print(f"交替共聚物分析:")
    print(f"  f_A: {result1['sequence_statistics']['f_A']:.3f}")
    print(f"  p_AA: {result1['sequence_statistics']['p_AA']:.3f}")
    print(f"  r_A: {result1['reactivity_ratios']['r_A']:.3f}")
    print(f"  峰值位置: {result1['distribution_stats']['peak_position']}")
    print(f"  平均块长度: {result1['distribution_stats']['mean_length']:.2f}")
    
    # 测试用例2：块状共聚物
    block_seq = ['AAAAABBBBBB', 'BBBBBAAAAAA']
    result2 = calc.get_detailed_analysis(block_seq)
    
    print(f"\n块状共聚物分析:")
    print(f"  f_A: {result2['sequence_statistics']['f_A']:.3f}")
    print(f"  p_AA: {result2['sequence_statistics']['p_AA']:.3f}")
    print(f"  r_A: {result2['reactivity_ratios']['r_A']:.3f}")
    print(f"  峰值位置: {result2['distribution_stats']['peak_position']}")
    print(f"  平均块长度: {result2['distribution_stats']['mean_length']:.2f}")
    
    print("✅ Mayo-Lewis计算器验证完成!")


if __name__ == "__main__":
    validate_mayo_lewis_calculation()
