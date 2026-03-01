import pandas as pd
import numpy as np
from ..metrics.psi import calculate_psi, calculate_psi_by_group


class ModelStabilityMonitor:
    """
    模型稳定性监控
    计算每周模型分数的PSI值，以第一周为基准
    """
    
    def __init__(self, score_col='score', week_col='week', n_bins=10, 
                 bin_edges=None, binning_strategy='auto'):
        """
        参数:
            score_col: 分数列名
            week_col: 周数列名
            n_bins: PSI计算分箱数
            bin_edges: 自定义分箱边界，None则使用n_bins自动分箱
            binning_strategy: 分箱策略，'auto'自动选择，'equal_freq'等频，'equal_width'等距
        """
        self.score_col = score_col
        self.week_col = week_col
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.binning_strategy = binning_strategy
        self.psi_results = None
        self.base_week = None
    
    def analyze(self, df, base_week=None):
        """
        分析模型稳定性
        
        参数:
            df: DataFrame，包含分数、周数
            base_week: 基准周，默认为第一周
        
        返回:
            DataFrame，每周PSI值
        """
        weeks = sorted(df[self.week_col].unique())
        
        if base_week is None:
            base_week = weeks[0]
        
        self.base_week = base_week
        base_data = df[df[self.week_col] == base_week][self.score_col]
        
        # 如果有自定义分箱边界，使用自定义分箱计算PSI
        if self.bin_edges is not None:
            bins = self.bin_edges
        else:
            # 根据策略选择分箱方式
            if self.binning_strategy == 'auto':
                # 自动选择：先尝试等频分箱
                bins = self.n_bins  # 先尝试等频
                # 如果等频分箱失败，会在calculate_psi中处理
            elif self.binning_strategy == 'equal_freq':
                bins = self.n_bins
            elif self.binning_strategy == 'equal_width':
                # 等距分箱：使用数据范围
                min_val = np.min(base_data)
                max_val = np.max(base_data)
                bins = np.linspace(min_val, max_val, self.n_bins + 1)
            else:
                bins = self.n_bins
        
        results = []
        for week in weeks:
            week_data = df[df[self.week_col] == week][self.score_col]
            if len(week_data) > 0:
                psi = calculate_psi(base_data, week_data, bins=bins)
                results.append({
                    self.week_col: week,
                    'psi': psi,
                    'sample_count': len(week_data),
                    'is_base': week == base_week
                })
        
        self.psi_results = pd.DataFrame(results)
        return self.psi_results
    
    def get_summary(self):
        """
        获取汇总信息
        """
        if self.psi_results is None:
            return None
        
        non_base = self.psi_results[~self.psi_results['is_base']]
        
        summary = {
            'weekly_psi': self.psi_results.to_dict('records'),
            'base_week': self.base_week,
            'max_psi': non_base['psi'].max() if len(non_base) > 0 else 0,
            'min_psi': non_base['psi'].min() if len(non_base) > 0 else 0,
            'avg_psi': round(non_base['psi'].mean(), 4) if len(non_base) > 0 else 0
        }
        
        return summary
