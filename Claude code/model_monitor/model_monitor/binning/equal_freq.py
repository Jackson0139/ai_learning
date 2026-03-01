import numpy as np
import pandas as pd


class EqualFreqBinner:
    """
    智能分箱器
    支持自定义分箱阈值，如果没有提供则基于基准数据自动分箱
    自动选择等频或等距分箱策略
    """
    
    def __init__(self, n_bins=10, bin_edges=None, binning_strategy='auto'):
        """
        参数:
            n_bins: 分箱数量，默认10
            bin_edges: 自定义分箱边界，如果提供则直接使用，否则基于数据自动学习
            binning_strategy: 分箱策略
                - 'auto': 自动选择（推荐），先尝试等频，失败则用等距
                - 'equal_freq': 等频分箱
                - 'equal_width': 等距分箱
        """
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.bin_labels = None
        self.is_custom_bins = bin_edges is not None
        self.binning_strategy = binning_strategy
        self.actual_strategy = 'custom' if bin_edges is not None else None  # 记录实际使用的策略
        
        if self.bin_edges is not None:
            self._setup_labels()
    
    def _setup_labels(self):
        """根据bin_edges设置标签"""
        self.bin_labels = [f'bin_{i+1}' for i in range(len(self.bin_edges) - 1)]
    
    def fit(self, data):
        """
        基于数据学习分箱边界（仅在未提供自定义分箱边界时执行）
        
        参数:
            data: 基准数据（如第一周数据）
        """
        # 如果已提供自定义分箱边界，则跳过学习
        if self.is_custom_bins:
            return self
        
        data = np.array(data)
        
        # 根据策略选择分箱方式
        if self.binning_strategy == 'auto':
            # 自动选择：先尝试等频分箱
            edges = self._equal_freq_binning(data)
            # 如果等频分箱失败（分箱数不足），降级为等距分箱
            if edges is None or len(edges) < self.n_bins + 1:
                edges = self._equal_width_binning(data)
                self.actual_strategy = 'equal_width'
            else:
                self.actual_strategy = 'equal_freq'
        elif self.binning_strategy == 'equal_freq':
            edges = self._equal_freq_binning(data)
            # 如果等频分箱失败，降级为等距分箱
            if edges is None or len(edges) < self.n_bins + 1:
                edges = self._equal_width_binning(data)
                self.actual_strategy = 'equal_width (fallback)'
            else:
                self.actual_strategy = 'equal_freq'
        elif self.binning_strategy == 'equal_width':
            edges = self._equal_width_binning(data)
            self.actual_strategy = 'equal_width'
        else:
            raise ValueError(f"不支持的binning_strategy: {self.binning_strategy}")
        
        self.bin_edges = edges
        self._setup_labels()
        
        return self
    
    def _equal_freq_binning(self, data):
        """
        等频分箱
        尝试将数据分成N个等频的区间
        """
        # 计算等频分箱边界
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(data, percentiles)
        
        # 确保边界唯一性
        bin_edges = np.unique(bin_edges)
        
        # 如果去重后分箱数不够，返回None表示失败
        if len(bin_edges) < self.n_bins + 1:
            return None
        
        # 将第一个边界设为-inf，最后一个边界设为+inf
        # 确保能覆盖所有可能的数据范围
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        return bin_edges
    
    def _equal_width_binning(self, data):
        """
        等距分箱
        将数据范围均匀分成N个区间
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        # 等距分箱边界
        bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
        
        # 将第一个边界设为-inf，最后一个边界设为+inf
        # 确保能覆盖所有可能的数据范围
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        return bin_edges
    
    def transform(self, data):
        """
        将数据转换为分箱标签
        
        参数:
            data: 要分箱的数据
        
        返回:
            分箱标签数组
        """
        if self.bin_edges is None:
            raise ValueError("请先调用fit方法学习分箱边界，或在初始化时提供bin_edges")
        
        data = np.array(data)
        
        # 使用pd.cut进行分箱
        bins = pd.cut(data, bins=self.bin_edges, labels=self.bin_labels, 
                    include_lowest=True, right=False)
        
        # 处理超出边界的数据，分配到最近的分箱
        # 小于最小边界的分配到第一个分箱
        # 大于最大边界的分配到最后一个分箱
        if bins.isna().any():
            min_edge = self.bin_edges[0]
            max_edge = self.bin_edges[-1]
            
            # 小于最小边界的
            bins[(data < min_edge) & bins.isna()] = self.bin_labels[0]
            # 大于最大边界的
            bins[(data >= max_edge) & bins.isna()] = self.bin_labels[-1]
        
        return bins
    
    def fit_transform(self, data):
        """
        学习分箱边界并转换数据
        """
        self.fit(data)
        return self.transform(data)
    
    def get_binning_stats(self, data, group_name=None):
        """
        获取分箱统计信息
        
        参数:
            data: 数据
            group_name: 分组名称（如周数）
        
        返回:
            DataFrame，包含每个分箱的统计信息
        """
        bins = self.transform(data)
        
        stats = pd.DataFrame({
            'bin': self.bin_labels,
            'bin_range': [f'({self.bin_edges[i]:.2f}, {self.bin_edges[i+1]:.2f}]' 
                         for i in range(len(self.bin_edges) - 1)]
        })
        
        # 统计每个箱的数量和占比（只统计成功分箱的数据，排除NaN）
        valid_bins = bins[~bins.isna()]
        value_counts = valid_bins.value_counts().sort_index()
        total = len(valid_bins)  # 使用有效数据的总数
        
        stats['count'] = stats['bin'].map(value_counts).fillna(0).astype(int)
        stats['percentage'] = (stats['count'] / total).round(4)
        
        if group_name is not None:
            stats['group'] = group_name
        
        return stats
    
    def get_bin_edges(self):
        """获取分箱边界"""
        return self.bin_edges
    
    def get_actual_strategy(self):
        """获取实际使用的分箱策略"""
        return self.actual_strategy
    
    def set_bin_edges(self, bin_edges):
        """
        设置自定义分箱边界
        
        参数:
            bin_edges: 分箱边界数组
        """
        self.bin_edges = np.array(bin_edges)
        self.is_custom_bins = True
        self._setup_labels()
