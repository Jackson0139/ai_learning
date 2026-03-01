import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

records_per_week = 5000
num_weeks = 4
total_records = records_per_week * num_weeks

end_date = datetime.now()
start_date = end_date - timedelta(weeks=num_weeks)

base_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

verify_times = np.random.randint(base_timestamp, end_timestamp, total_records)
verify_times.sort()

# 生成不均匀分布的数据
# 第一周：分数集中在600-700之间
week1_scores = np.random.randint(600, 701, records_per_week)
# 其他周：分数分布均匀
other_scores = np.random.randint(300, 1001, records_per_week * (num_weeks - 1))
scores = np.concatenate([week1_scores, other_scores])

# 生成特征，其中fea1有大量重复值（模拟不均匀分布）
# 第一周：fea1大部分是50
week1_fea1 = np.random.choice([50, 51, 49], records_per_week, p=[0.8, 0.1, 0.1])
# 其他周：fea1分布均匀
other_fea1 = np.random.normal(50, 15, records_per_week * (num_weeks - 1))
fea1 = np.concatenate([week1_fea1, other_fea1])

# 其他特征正常分布
fea2 = np.random.uniform(0, 100, total_records)
fea3 = np.random.exponential(10, total_records)

labels = np.random.choice([0, 1], total_records, p=[0.7, 0.3])

df = pd.DataFrame({
    'score': scores,
    'fea1': np.round(fea1, 4),
    'fea2': np.round(fea2, 4),
    'fea3': np.round(fea3, 4),
    'verify_time': verify_times,
    'label': labels
})

output_path = '/Users/pengsmac/Desktop/learning/AI工具学习/Claude code/model_monitor/data/model_data_uneven.csv'
df.to_csv(output_path, index=False)

print(f"不均匀数据生成完成！")
print(f"总记录数: {len(df)}")
print(f"\n第一周分数分布（集中在600-700）:")
week1_data = df[df['verify_time'] < base_timestamp + 7 * 24 * 3600]
print(f"  最小值: {week1_data['score'].min()}")
print(f"  最大值: {week1_data['score'].max()}")
print(f"  唯一值数量: {week1_data['score'].nunique()}")

print(f"\n第一周fea1分布（大量重复值）:")
print(f"  最小值: {week1_data['fea1'].min()}")
print(f"  最大值: {week1_data['fea1'].max()}")
print(f"  唯一值数量: {week1_data['fea1'].nunique()}")
print(f"  值50的占比: {(week1_data['fea1'] == 50).sum() / len(week1_data) * 100:.2f}%")

print(f"\n数据预览:")
print(df.head(10))
