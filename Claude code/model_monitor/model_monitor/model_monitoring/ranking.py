import pandas as pd
import numpy as np
from ..binning.equal_freq import EqualFreqBinner


class ModelRankingAnalyzer:
    """
    模型排序性分析
    统计每周每个分箱的数量、占比
    """
    
    def __init__(self, score_col='score', week_col='week', n_bins=10, 
                 bin_edges=None, binning_strategy='auto'):
        """
        参数:
            score_col: 分数列名
            week_col: 周数列名
            n_bins: 分箱数量
            bin_edges: 自定义分箱边界，None则使用n_bins自动分箱
            binning_strategy: 分箱策略，'auto'自动选择，'equal_freq'等频，'equal_width'等距
        """
        self.score_col = score_col
        self.week_col = week_col
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.binning_strategy = binning_strategy
        self.binner = EqualFreqBinner(n_bins=n_bins, bin_edges=bin_edges, 
                                     binning_strategy=binning_strategy)
        self.ranking_results = None
        self.base_week = None
        self.data_distribution = {}  # 存储数据分布统计信息
    
    def analyze(self, df, base_week=None):
        """
        分析模型排序性
        
        参数:
            df: DataFrame
            base_week: 基准周，用于学习分箱边界
        
        返回:
            DataFrame，包含每周每个分箱的统计
        """
        weeks = sorted(df[self.week_col].unique())
        
        if base_week is None:
            base_week = weeks[0]
        
        self.base_week = base_week
        
        # 收集基准周的数据分布统计信息
        base_data = df[df[self.week_col] == base_week][self.score_col]
        self.data_distribution = {
            'min': float(base_data.min()),
            'max': float(base_data.max()),
            'mean': float(base_data.mean()),
            'std': float(base_data.std()),
            'unique_count': int(base_data.nunique()),
            'sample_count': len(base_data),
            'is_uniform': base_data.nunique() >= self.n_bins
        }
        
        # 如果没有自定义分箱边界，用基准周数据学习分箱边界
        if self.bin_edges is None:
            self.binner.fit(base_data)
        
        # 统计每周每个分箱的数据
        all_stats = []
        for week in weeks:
            week_data = df[df[self.week_col] == week][self.score_col]
            if len(week_data) > 0:
                stats = self.binner.get_binning_stats(week_data, group_name=week)
                all_stats.append(stats)
        
        self.ranking_results = pd.concat(all_stats, ignore_index=True)
        
        return self.ranking_results
    
    def get_summary(self):
        """
        获取汇总信息
        """
        if self.ranking_results is None:
            return None
        
        # 按周汇总
        weekly_summary = self.ranking_results.groupby('group').agg({
            'count': 'sum',
            'percentage': 'sum'
        }).reset_index()
        
        # 获取分箱边界
        bin_edges = self.binner.get_bin_edges()
        if isinstance(bin_edges, np.ndarray):
            bin_edges = bin_edges.tolist()
        
        summary = {
            'base_week': self.base_week,
            'n_bins': self.n_bins,
            'bin_edges': bin_edges,
            'weekly_stats': weekly_summary.to_dict('records'),
            'detailed_stats': self.ranking_results.to_dict('records')
        }
        
        return summary
    
    def get_binning_stats_by_week(self, week):
        """
        获取指定周的分箱统计
        """
        if self.ranking_results is None:
            return None
        
        return self.ranking_results[self.ranking_results['group'] == week]
