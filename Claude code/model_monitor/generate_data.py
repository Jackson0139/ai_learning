import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

records_per_week = 5000
num_weeks = 10
total_records = records_per_week * num_weeks

end_date = datetime.now()
start_date = end_date - timedelta(weeks=num_weeks)

base_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

verify_times = np.random.randint(base_timestamp, end_timestamp, total_records)
verify_times.sort()

scores = np.random.randint(300, 1001, total_records)

# 生成10个特征
fea1 = np.random.normal(50, 15, total_records)
fea2 = np.random.uniform(0, 100, total_records)
fea3 = np.random.exponential(10, total_records)
fea4 = np.random.normal(100, 25, total_records)
fea5 = np.random.uniform(-50, 50, total_records)
fea6 = np.random.gamma(2, 2, total_records)
fea7 = np.random.beta(2, 5, total_records) * 100
fea8 = np.random.normal(0, 1, total_records)
fea9 = np.random.poisson(5, total_records)
fea10 = np.random.lognormal(0, 1, total_records)

labels = np.random.choice([0, 1], total_records, p=[0.7, 0.3])

df = pd.DataFrame({
    'score': scores,
    'fea1': np.round(fea1, 4),
    'fea2': np.round(fea2, 4),
    'fea3': np.round(fea3, 4),
    'fea4': np.round(fea4, 4),
    'fea5': np.round(fea5, 4),
    'fea6': np.round(fea6, 4),
    'fea7': np.round(fea7, 4),
    'fea8': np.round(fea8, 4),
    'fea9': np.round(fea9, 4),
    'fea10': np.round(fea10, 4),
    'verify_time': verify_times,
    'label': labels
})

output_path = '/Users/pengsmac/Desktop/learning/AI工具学习/Claude code/model_monitor/data/model_data.csv'
df.to_csv(output_path, index=False)

print(f"数据生成完成！")
print(f"总记录数: {len(df)}")
print(f"特征数: 10个")
print(f"时间范围: {datetime.fromtimestamp(df['verify_time'].min())} ~ {datetime.fromtimestamp(df['verify_time'].max())}")
print(f"每周数据量检查:")

for week in range(num_weeks):
    week_start = base_timestamp + week * 7 * 24 * 3600
    week_end = week_start + 7 * 24 * 3600
    week_count = len(df[(df['verify_time'] >= week_start) & (df['verify_time'] < week_end)])
    print(f"  第{week+1}周: {week_count} 条")

print(f"\n数据预览:")
print(df.head(10))
print(f"\n标签分布:")
print(df['label'].value_counts())
