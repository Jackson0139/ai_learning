import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10, epsilon=1e-10):
    """
    计算PSI值（Population Stability Index）
    
    参数:
        expected: 基准分布（训练集或第一周数据）
        actual: 实际分布（待比较的数据）
        bins: 分箱数量或自定义分箱边界列表，默认10
        epsilon: 防止除零的小值
    
    返回:
        psi值
    """
    expected = np.array(expected)
    actual = np.array(actual)
    
    # 判断bins是整数（分箱数量）还是列表（自定义分箱边界）
    if isinstance(bins, (list, np.ndarray)):
        # 使用自定义分箱边界
        bin_edges = np.array(bins)
    else:
        # 创建分箱边界（基于expected数据）
        min_val = np.min(expected)
        max_val = np.max(expected)
        
        # 等频分箱
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        
        # 确保边界唯一性
        bin_edges = np.unique(bin_edges)
        
        # 如果等频分箱失败（去重后分箱数不够），降级为等距分箱
        if len(bin_edges) < bins + 1:
            bin_edges = np.linspace(min_val, max_val, bins + 1)
        else:
            bin_edges[0] = min_val - epsilon
            bin_edges[-1] = max_val + epsilon
    
    # 计算每个箱的占比
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # 转换为比例
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # 防止除零
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
    
    # 计算PSI: sum((actual% - expected%) * ln(actual% / expected%))
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)
    
    return round(psi, 4)


def calculate_psi_by_group(df, col, group_col, base_group=None, bins=10):
    """
    按分组计算PSI值，与基准组比较
    
    参数:
        df: DataFrame
        col: 要计算PSI的列名
        group_col: 分组列名（如周数）
        base_group: 基准组值，默认为第一组
        bins: 分箱数量或自定义分箱边界列表
    
    返回:
        DataFrame，包含每组的PSI值
    """
    groups = sorted(df[group_col].unique())
    
    if base_group is None:
        base_group = groups[0]
    
    # 基准数据
    base_data = df[df[group_col] == base_group][col]
    
    results = []
    for group_val in groups:
        group_data = df[df[group_col] == group_val][col]
        if len(group_data) > 0:
            psi = calculate_psi(base_data, group_data, bins=bins)
            results.append({
                group_col: group_val,
                'psi': psi,
                'sample_count': len(group_data),
                'is_base': group_val == base_group
            })
    
    return pd.DataFrame(results)
