import pandas as pd
import numpy as np
from ..metrics.psi import calculate_psi
from ..binning.equal_freq import EqualFreqBinner


class FeatureStabilityMonitor:
    """
    特征稳定性监控
    计算每个特征每周的PSI值和分箱统计明细
    """
    
    def __init__(self, feature_cols, week_col='week', n_bins=10, 
                 feature_bins=None, binning_strategy='auto'):
        """
        参数:
            feature_cols: 特征列名列表
            week_col: 周数列名
            n_bins: 分箱数量
            feature_bins: 特征自定义分箱边界字典，如{'fea1': [0, 20, 50, 100], ...}
            binning_strategy: 分箱策略，'auto'自动选择，'equal_freq'等频，'equal_width'等距
        """
        self.feature_cols = feature_cols
        self.week_col = week_col
        self.n_bins = n_bins
        self.feature_bins = feature_bins or {}
        self.binning_strategy = binning_strategy
        self.psi_results = {}
        self.binning_results = {}
        self.binners = {}
        self.base_week = None
        self.data_distribution = {}  # 存储数据分布统计信息
    
    def analyze(self, df, base_week=None):
        """
        分析特征稳定性
        
        参数:
            df: DataFrame
            base_week: 基准周
        
        返回:
            dict，每个特征的PSI和分箱统计
        """
        weeks = sorted(df[self.week_col].unique())
        
        if base_week is None:
            base_week = weeks[0]
        
        self.base_week = base_week
        
        for feature in self.feature_cols:
            # 收集基准周的数据分布统计信息
            base_data = df[df[self.week_col] == base_week][feature]
            self.data_distribution[feature] = {
                'min': float(base_data.min()),
                'max': float(base_data.max()),
                'mean': float(base_data.mean()),
                'std': float(base_data.std()),
                'unique_count': int(base_data.nunique()),
                'sample_count': len(base_data),
                'is_uniform': self._is_uniform(base_data)
            }
            
            # 获取该特征的自定义分箱边界（如果有）
            custom_bins = self.feature_bins.get(feature)
            
            # 计算PSI
            psi_results = self._calculate_feature_psi(df, feature, weeks, base_week, custom_bins)
            self.psi_results[feature] = psi_results
            
            # 分箱统计
            binning_results = self._calculate_feature_binning(df, feature, weeks, base_week, custom_bins)
            self.binning_results[feature] = binning_results
        
        return {
            'psi': self.psi_results,
            'binning': self.binning_results
        }
    
    def _is_uniform(self, data):
        """
        判断数据是否均匀分布
        如果唯一值数量小于分箱数量，认为不均匀
        """
        return data.nunique() >= self.n_bins
    
    def _calculate_feature_psi(self, df, feature, weeks, base_week, custom_bins=None):
        """计算单个特征的PSI"""
        base_data = df[df[self.week_col] == base_week][feature]
        
        # 使用自定义分箱或根据策略选择分箱方式
        if custom_bins is not None:
            bins = custom_bins
        else:
            # 根据策略选择分箱方式
            if self.binning_strategy == 'auto':
                bins = self.n_bins  # 先尝试等频，失败会在calculate_psi中处理
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
            week_data = df[df[self.week_col] == week][feature]
            if len(week_data) > 0:
                psi = calculate_psi(base_data, week_data, bins=bins)
                results.append({
                    self.week_col: week,
                    'psi': psi,
                    'sample_count': len(week_data),
                    'is_base': week == base_week
                })
        
        return pd.DataFrame(results)
    
    def _calculate_feature_binning(self, df, feature, weeks, base_week, custom_bins=None):
        """计算单个特征的分箱统计"""
        # 创建分箱器，传入自定义分箱边界（如果有）和分箱策略
        binner = EqualFreqBinner(n_bins=self.n_bins, bin_edges=custom_bins, 
                               binning_strategy=self.binning_strategy)
        
        # 如果没有自定义分箱边界，用基准周数据学习分箱边界
        if custom_bins is None:
            base_data = df[df[self.week_col] == base_week][feature]
            binner.fit(base_data)
        
        self.binners[feature] = binner
        
        # 统计每周每个分箱的数据
        all_stats = []
        for week in weeks:
            week_data = df[df[self.week_col] == week][feature]
            if len(week_data) > 0:
                stats = binner.get_binning_stats(week_data, group_name=week)
                all_stats.append(stats)
        
        return pd.concat(all_stats, ignore_index=True)
    
    def get_summary(self):
        """
        获取汇总信息
        """
        summary = {
            'base_week': self.base_week,
            'features': {}
        }
        
        for feature in self.feature_cols:
            psi_df = self.psi_results[feature]
            non_base = psi_df[~psi_df['is_base']]
            
            # 获取分箱边界
            bin_edges = self.binners[feature].get_bin_edges()
            if isinstance(bin_edges, np.ndarray):
                bin_edges = bin_edges.tolist()
            
            summary['features'][feature] = {
                'weekly_psi': psi_df.to_dict('records'),
                'max_psi': non_base['psi'].max() if len(non_base) > 0 else 0,
                'min_psi': non_base['psi'].min() if len(non_base) > 0 else 0,
                'avg_psi': round(non_base['psi'].mean(), 4) if len(non_base) > 0 else 0,
                'binning_stats': self.binning_results[feature].to_dict('records'),
                'bin_edges': bin_edges
            }
        
        return summary
