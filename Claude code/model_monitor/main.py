#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型监控工具主入口

使用示例:
    from main import ModelMonitor
    
    # 方式1: 基础使用（自动分箱）
    monitor = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3']
    )
    results = monitor.run('data/model_data.csv')
    
    # 方式2: 传入自定义分箱阈值
    monitor = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3'],
        score_bins=[300, 400, 500, 600, 700, 800, 900, 1000],  # 自定义模型分数分箱
        feature_bins={
            'fea1': [0, 20, 40, 60, 80, 100],
            'fea2': [0, 25, 50, 75, 100]
        }  # 自定义特征分箱，未指定的特征将自动分箱
    )
    results = monitor.run('data/model_data.csv')
"""

import pandas as pd
from model_monitor import (
    ModelDiscriminationMonitor,
    ModelStabilityMonitor,
    ModelRankingAnalyzer,
    FeatureStabilityMonitor,
    DataProcessor
)
from model_monitor.reports import ReportGenerator


class ModelMonitor:
    """
    模型监控主类
    整合所有监控功能，提供统一接口
    """
    
    def __init__(self, score_col, label_col, feature_cols,
                 timestamp_col='verify_time', week_col='week', n_bins=10,
                 score_bins=None, feature_bins=None, binning_strategy='auto'):
        """
        参数:
            score_col: 模型分数列名（必填）
            label_col: 标签列名（必填）
            feature_cols: 入模特征列名列表（必填）
            timestamp_col: 时间戳列名，默认'verify_time'
            week_col: 周数列名，默认'week'
            n_bins: 分箱数量，默认10
            score_bins: 模型分数自定义分箱边界，None则自动分箱
            feature_bins: 特征自定义分箱边界字典，如{'fea1': [0, 20, 50, 100], ...}，未指定的特征将自动分箱
            binning_strategy: 分箱策略，'auto'自动选择，'equal_freq'等频，'equal_width'等距
        """
        self.score_col = score_col
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.timestamp_col = timestamp_col
        self.week_col = week_col
        self.n_bins = n_bins
        self.score_bins = score_bins
        self.feature_bins = feature_bins or {}
        self.binning_strategy = binning_strategy
        
        # 初始化监控器
        self.discrimination_monitor = None
        self.stability_monitor = None
        self.ranking_analyzer = None
        self.feature_monitor = None
        
        # 报告生成器
        self.report_generator = ReportGenerator()
    
    def run(self, data_path, base_week=None, output_excel=None):
        """
        运行完整的模型监控流程
        
        参数:
            data_path: 数据文件路径
            base_week: 基准周，None则使用第一周
            output_excel: Excel输出路径，None则不导出
        
        返回:
            监控结果字典
        """
        # 1. 加载数据
        print("正在加载数据...")
        df = DataProcessor.load_data(data_path)
        print(f"数据加载完成，共 {len(df)} 条记录")
        
        # 2. 验证必填列是否存在
        required_cols = [self.score_col, self.label_col, self.timestamp_col] + self.feature_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件中缺少以下列: {missing_cols}")
        
        print(f"模型分数列: {self.score_col}")
        print(f"标签列: {self.label_col}")
        print(f"入模特征: {self.feature_cols}")
        
        # 显示分箱方式
        if self.score_bins:
            print(f"模型分数分箱: 自定义分箱 {self.score_bins}")
        else:
            print(f"模型分数分箱: 自动分箱（n_bins={self.n_bins}）")
        
        custom_features = [f for f in self.feature_cols if f in self.feature_bins]
        auto_features = [f for f in self.feature_cols if f not in self.feature_bins]
        if custom_features:
            print(f"特征分箱: 自定义分箱 {custom_features}")
        if auto_features:
            print(f"特征分箱: 自动分箱 {auto_features}")
        
        # 3. 添加周数列
        print("\n正在处理时间数据...")
        df = DataProcessor.add_week_column(df, self.timestamp_col, self.week_col)
        weeks = sorted(df[self.week_col].unique())
        print(f"数据时间跨度: 第{weeks[0]}周 到 第{weeks[-1]}周")
        
        # 4. 模型区分度监控（KS）
        print("\n正在计算模型区分度(KS)...")
        self.discrimination_monitor = ModelDiscriminationMonitor(
            score_col=self.score_col,
            label_col=self.label_col,
            week_col=self.week_col
        )
        self.discrimination_monitor.analyze(df)
        print("KS计算完成")
        
        # 5. 模型稳定性监控（PSI）
        print("\n正在计算模型稳定性(PSI)...")
        self.stability_monitor = ModelStabilityMonitor(
            score_col=self.score_col,
            week_col=self.week_col,
            n_bins=self.n_bins,
            bin_edges=self.score_bins,
            binning_strategy=self.binning_strategy
        )
        self.stability_monitor.analyze(df, base_week=base_week)
        bin_type = "自定义分箱" if self.score_bins else "自动分箱"
        print(f"PSI计算完成（{bin_type}），基准周: 第{self.stability_monitor.base_week}周")
        
        # 6. 模型排序性分析
        print("\n正在分析模型排序性...")
        self.ranking_analyzer = ModelRankingAnalyzer(
            score_col=self.score_col,
            week_col=self.week_col,
            n_bins=self.n_bins,
            bin_edges=self.score_bins,
            binning_strategy=self.binning_strategy
        )
        self.ranking_analyzer.analyze(df, base_week=base_week)
        print("排序性分析完成")
        
        # 打印模型分数分箱策略和数据分布信息
        print("\n模型分数分箱策略和数据分布:")
        print("-" * 80)
        dist = self.ranking_analyzer.data_distribution
        strategy = self.ranking_analyzer.binner.get_actual_strategy()
        print(f"分箱策略: {strategy}")
        print(f"唯一值数量: {dist['unique_count']}")
        print(f"最小值: {dist['min']:.2f}")
        print(f"最大值: {dist['max']:.2f}")
        print(f"平均值: {dist['mean']:.2f}")
        print(f"标准差: {dist['std']:.2f}")
        print(f"样本数量: {dist['sample_count']}")
        if not dist['is_uniform']:
            print(f"⚠️  模型分数数据分布不均匀（唯一值{dist['unique_count']} < 分箱数{self.n_bins}），已使用{strategy}分箱")
        print("-" * 80)
        
        # 7. 特征稳定性监控
        print("\n正在分析特征稳定性...")
        self.feature_monitor = FeatureStabilityMonitor(
            feature_cols=self.feature_cols,
            week_col=self.week_col,
            n_bins=self.n_bins,
            feature_bins=self.feature_bins,
            binning_strategy=self.binning_strategy
        )
        self.feature_monitor.analyze(df, base_week=base_week)
        print("特征稳定性分析完成")
        
        # 打印特征分箱策略和数据分布信息
        print("\n特征分箱策略和数据分布:")
        print("-" * 80)
        print(f"{'特征':<10} {'分箱策略':<20} {'唯一值':<10} {'最小值':<10} {'最大值':<10} {'标准差':<10}")
        print("-" * 80)
        for feature in self.feature_cols:
            dist = self.feature_monitor.data_distribution[feature]
            strategy = self.feature_monitor.binners[feature].get_actual_strategy()
            uniform_status = "均匀" if dist['is_uniform'] else "不均匀"
            print(f"{feature:<10} {strategy:<20} {dist['unique_count']:<10} "
                  f"{dist['min']:<10.2f} {dist['max']:<10.2f} {dist['std']:<10.2f}")
            if not dist['is_uniform']:
                print(f"  ⚠️  {feature} 数据分布不均匀（唯一值{dist['unique_count']} < 分箱数{self.n_bins}），已使用{strategy}分箱")
        print("-" * 80)
        
        # 8. 生成报告
        print("\n正在生成报告...")
        report = self.report_generator.generate_full_report(
            model_discrimination=self.discrimination_monitor,
            model_stability=self.stability_monitor,
            model_ranking=self.ranking_analyzer,
            feature_stability=self.feature_monitor
        )
        
        # 9. 打印摘要
        self.report_generator.print_summary(report)
        
        # 10. 导出Excel（可选）
        if output_excel:
            print(f"\n正在导出Excel报告...")
            self.report_generator.export_to_excel(
                model_discrimination=self.discrimination_monitor,
                model_stability=self.stability_monitor,
                model_ranking=self.ranking_analyzer,
                feature_stability=self.feature_monitor,
                output_file=output_excel
            )
        
        return report
    
    def get_model_ks(self):
        """获取模型KS结果"""
        if self.discrimination_monitor:
            return self.discrimination_monitor.ks_results
        return None
    
    def get_model_psi(self):
        """获取模型PSI结果"""
        if self.stability_monitor:
            return self.stability_monitor.psi_results
        return None
    
    def get_model_ranking(self):
        """获取模型排序性结果"""
        if self.ranking_analyzer:
            return self.ranking_analyzer.ranking_results
        return None
    
    def get_feature_psi(self, feature=None):
        """
        获取特征PSI结果
        
        参数:
            feature: 特征名，None则返回所有特征
        """
        if self.feature_monitor:
            if feature:
                return self.feature_monitor.psi_results.get(feature)
            return self.feature_monitor.psi_results
        return None


if __name__ == '__main__':
    print("=" * 80)
    print("模型监控工具 - 批量处理")
    print("=" * 80)
    
    # 场景1: 均匀数据 - 自动分箱
    print("\n【场景1】均匀数据 - 自动分箱")
    print("=" * 80)
    monitor1 = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3'],
        timestamp_col='verify_time',
        week_col='week',
        n_bins=10,
        score_bins=None,
        feature_bins=None,
        binning_strategy='auto'
    )
    
    results1 = monitor1.run(
        data_path='data/model_data.csv',
        output_excel='reports/model_monitor_report_uniform.xlsx'
    )
    
    # 场景2: 不均匀数据 - 自动分箱（智能降级）
    print("\n\n【场景2】不均匀数据 - 自动分箱（智能降级）")
    print("=" * 80)
    monitor2 = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3'],
        timestamp_col='verify_time',
        week_col='week',
        n_bins=10,
        score_bins=None,
        feature_bins=None,
        binning_strategy='auto'
    )
    
    results2 = monitor2.run(
        data_path='data/model_data_uneven.csv',
        output_excel='reports/model_monitor_report_uneven_auto.xlsx'
    )
    
    # 场景3: 不均匀数据 - 自定义分箱阈值
    print("\n\n【场景3】不均匀数据 - 自定义分箱阈值")
    print("=" * 80)
    monitor3 = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3'],
        timestamp_col='verify_time',
        week_col='week',
        n_bins=10,
        # 自定义模型分数分箱
        score_bins=[300, 400, 500, 600, 700, 800, 900, 1000],
        # 自定义特征分箱
        feature_bins={
            'fea1': [0, 20, 40, 60, 80, 100],
            'fea2': [0, 25, 50, 75, 100],
            'fea3': None  # 自动分箱
        },
        binning_strategy='auto'
    )
    
    results3 = monitor3.run(
        data_path='data/model_data_uneven.csv',
        output_excel='reports/model_monitor_report_uneven_custom.xlsx'
    )
    
    # 场景4: 不均匀数据 - 强制等距分箱
    print("\n\n【场景4】不均匀数据 - 强制等距分箱")
    print("=" * 80)
    monitor4 = ModelMonitor(
        score_col='score',
        label_col='label',
        feature_cols=['fea1', 'fea2', 'fea3'],
        timestamp_col='verify_time',
        week_col='week',
        n_bins=10,
        score_bins=None,
        feature_bins=None,
        binning_strategy='equal_width'
    )
    
    results4 = monitor4.run(
        data_path='data/model_data_uneven.csv',
        output_excel='reports/model_monitor_report_uneven_equal_width.xlsx'
    )
    
    print("\n\n" + "=" * 80)
    print("所有场景处理完成！")
    print("=" * 80)
    print("\n生成的报告文件:")
    print("  1. reports/model_monitor_report_uniform.xlsx - 均匀数据自动分箱")
    print("  2. reports/model_monitor_report_uneven_auto.xlsx - 不均匀数据自动分箱")
    print("  3. reports/model_monitor_report_uneven_custom.xlsx - 不均匀数据自定义分箱")
    print("  4. reports/model_monitor_report_uneven_equal_width.xlsx - 不均匀数据等距分箱")
