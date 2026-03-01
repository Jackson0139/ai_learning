import numpy as np
import pandas as pd


def calculate_ks(score, label):
    """
    计算KS值（Kolmogorov-Smirnov统计量）
    
    参数:
        score: 模型分数，numpy数组或pandas Series
        label: 标签（0/1），numpy数组或pandas Series
    
    返回:
        ks值（0-1之间）
    """
    score = np.array(score)
    label = np.array(label)
    
    # 按分数排序
    sorted_indices = np.argsort(score)
    sorted_score = score[sorted_indices]
    sorted_label = label[sorted_indices]
    
    # 计算累计分布
    total_good = np.sum(label == 0)
    total_bad = np.sum(label == 1)
    
    if total_good == 0 or total_bad == 0:
        return 0.0
    
    cum_good = np.cumsum(sorted_label == 0) / total_good
    cum_bad = np.cumsum(sorted_label == 1) / total_bad
    
    # KS = max(|cum_good - cum_bad|)
    ks = np.max(np.abs(cum_good - cum_bad))
    
    return round(ks, 4)


def calculate_ks_by_group(df, score_col, label_col, group_col):
    """
    按分组计算KS值
    
    参数:
        df: DataFrame
        score_col: 分数列名
        label_col: 标签列名
        group_col: 分组列名（如周数）
    
    返回:
        DataFrame，包含每组的KS值、bad_cnt、bad_rate
    """
    results = []
    
    for group_val, group_df in df.groupby(group_col):
        if len(group_df) > 0:
            ks = calculate_ks(group_df[score_col], group_df[label_col])
            bad_cnt = int(group_df[label_col].sum())  # target求和
            bad_rate = round(group_df[label_col].mean(), 4)  # target取平均
            results.append({
                group_col: group_val,
                'ks': ks,
                'sample_count': len(group_df),
                'bad_cnt': bad_cnt,
                'bad_rate': bad_rate
            })
    
    return pd.DataFrame(results)
