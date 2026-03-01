import pandas as pd
from ..metrics.ks import calculate_ks, calculate_ks_by_group


class ModelDiscriminationMonitor:
    """
    模型区分度监控
    计算每周模型分数的KS值
    """
    
    def __init__(self, score_col='score', label_col='label', week_col='week'):
        """
        参数:
            score_col: 分数列名
            label_col: 标签列名
            week_col: 周数列名
        """
        self.score_col = score_col
        self.label_col = label_col
        self.week_col = week_col
        self.ks_results = None
    
    def analyze(self, df):
        """
        分析模型区分度
        
        参数:
            df: DataFrame，包含分数、标签、周数
        
        返回:
            DataFrame，每周KS值
        """
        self.ks_results = calculate_ks_by_group(
            df, 
            self.score_col, 
            self.label_col, 
            self.week_col
        )
        
        return self.ks_results
    
    def get_summary(self):
        """
        获取汇总信息
        """
        if self.ks_results is None:
            return None
        
        summary = {
            'weekly_ks': self.ks_results.to_dict('records'),
            'max_ks': self.ks_results['ks'].max(),
            'min_ks': self.ks_results['ks'].min(),
            'avg_ks': round(self.ks_results['ks'].mean(), 4)
        }
        
        return summary
