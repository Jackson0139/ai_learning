import pandas as pd
import numpy as np
from datetime import datetime


class DataProcessor:
    """
    数据处理器
    处理原始数据，添加周数列等
    """
    
    @staticmethod
    def add_week_column(df, timestamp_col='verify_time', week_col='week'):
        """
        根据时间戳添加周数列（日期范围格式）
        
        参数:
            df: DataFrame
            timestamp_col: 时间戳列名
            week_col: 周数列名
        
        返回:
            添加了周数列的DataFrame
        """
        df = df.copy()
        
        # 将时间戳转换为日期
        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
        
        # 按周分组（以数据中最小日期所在周为第1周）
        min_date = df['datetime'].min()
        df['week_num'] = ((df['datetime'] - min_date).dt.days // 7 + 1).astype(int)
        
        # 生成日期范围格式: %y-%m-%d ~ %y-%m-%d
        def get_week_date_range(week_num):
            week_start = min_date + pd.Timedelta(days=(week_num - 1) * 7)
            week_end = week_start + pd.Timedelta(days=6)
            return f"{week_start.strftime('%y-%m-%d')} ~ {week_end.strftime('%y-%m-%d')}"
        
        df[week_col] = df['week_num'].apply(get_week_date_range)
        
        # 删除临时列
        df = df.drop(columns=['datetime', 'week_num'])
        
        return df
    
    @staticmethod
    def load_data(file_path):
        """
        加载数据
        """
        df = pd.read_csv(file_path)
        return df
    
    @staticmethod
    def get_feature_columns(df, exclude_cols=None):
        """
        获取特征列（排除指定列）
        """
        if exclude_cols is None:
            exclude_cols = ['score', 'label', 'verify_time', 'week']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
