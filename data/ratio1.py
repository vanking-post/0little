#读取traffic_flows_complete1中的数据，与源数据库traffic_flows_west进行对比，主要对比每个车辆ID下的数据行数，计算比例与差值。
#最后将处理后数据数量占比小于0.9的车辆ID从traffic_flows_complete1中全部删除并写入traffic_flows_complete1中。
#计算出的比例也写入文件traffic_flows_ratio1中。
#后来经思虑无益。
import pandas as pd
import numpy as np
import gc
import os

# 设置数据路径
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
west_pkl_processed_path = os.path.join(save_dir, "traffic_flows_complete1.pkl")
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")
ratio_output_path = os.path.join(save_dir, "traffic_flows_ratio1.csv")

# 读取数据
traffic_flows_west = pd.read_pickle(west_pkl_path)
traffic_flows_west_processed = pd.read_pickle(west_pkl_processed_path)

# 计算每个ID的数量
original_counts = traffic_flows_west['ID'].value_counts().reset_index()
original_counts.columns = ['ID', 'original_count']

processed_counts = traffic_flows_west_processed['ID'].value_counts().reset_index()
processed_counts.columns = ['ID', 'processed_count']

# 合并两个计数结果，只保留processed中存在的ID
ratio_df = pd.merge(processed_counts, original_counts, on='ID', how='inner')

# 计算比例（处理后数量 / 原始数量）和差值
ratio_df['ratio'] = (ratio_df['processed_count'] / ratio_df['original_count']).round(3)
ratio_df['count_diff'] = (- ratio_df['processed_count'] + ratio_df['original_count'])

# 过滤掉数量相同的ID
ratio_df = ratio_df[ratio_df['processed_count'] != ratio_df['original_count']]

# 重命名列
ratio_df = ratio_df.rename(columns={'processed_count': 'processed_count', 'original_count': 'original_count'})

# 按照ID升序排序
ratio_df = ratio_df.sort_values(by='ID').reset_index(drop=True)

# 选择输出列并排序（ID, processed_count, west_count, count_diff, ratio）
ratio_df = ratio_df[['ID', 'processed_count', 'original_count', 'count_diff', 'ratio']]

# 保存结果
ratio_df.to_csv(ratio_output_path, index=False, encoding='utf-8-sig')

print(f"\nID数量对比结果已保存至：{ratio_output_path}")
print(f"共涉及 {len(ratio_df)} 个不同车辆ID")

# 统计保留比例的分布
if len(ratio_df) > 0:
    total_original = len(traffic_flows_west)
    total_processed = len(traffic_flows_west_processed)
    overall_ratio = total_processed / total_original

    print(f"总体保留比例：{overall_ratio:.4f} ({total_processed}/{total_original})")
    print(f"保留比例统计：")
    print(f"完全保留的ID数量：{len(ratio_df[ratio_df['ratio'] == 1.0])}")
    print(f"部分保留的ID数量：{len(ratio_df[(ratio_df['ratio'] < 1.0) & (ratio_df['ratio'] > 0.0)])}")
    print(f"平均保留比例：{ratio_df['ratio'].mean():.4f}")
    print(f"保留比例中位数：{ratio_df['ratio'].median():.4f}")
    print(f"平均差值：{ratio_df['count_diff'].mean():.2f}")
else:
    print("没有需要记录的车辆ID（所有ID都完全保留或完全删除）")

# 显示差值最大的10个ID（如果存在）
if len(ratio_df) > 0:
    print(f"\n差值最大的10个ID：")
    print(ratio_df.nlargest(10, 'count_diff')[['ID', 'processed_count', 'original_count', 'count_diff', 'ratio']])

# 根据保留比例阈值过滤数据
THRESHOLD = 0.9

# 获取保留比例低于阈值的ID
low_ratio_ids = ratio_df[ratio_df['ratio'] < THRESHOLD]['ID'].tolist()

# 获取保留比例大于等于阈值的ID
high_ratio_ids = ratio_df[ratio_df['ratio'] >= THRESHOLD]['ID'].tolist()

print(f"\n按阈值 {THRESHOLD} 过滤数据：")
print(f"保留比例 < {THRESHOLD} 的ID数量：{len(low_ratio_ids)}")
print(f"保留比例 >= {THRESHOLD} 的ID数量：{len(high_ratio_ids)}")

# 从processed数据中删除保留比例低于阈值的ID
filtered_traffic_flows = traffic_flows_west_processed[~traffic_flows_west_processed['ID'].isin(low_ratio_ids)].copy()

print(f"过滤前数据行数：{len(traffic_flows_west_processed)}")
print(f"过滤后数据行数：{len(filtered_traffic_flows)}")
print(f"删除了 {len(traffic_flows_west_processed) - len(filtered_traffic_flows)} 行数据")

# 保存过滤后的数据到原路径
#filtered_traffic_flows.to_pickle(west_pkl_processed_path) #########慎重写入
#print(f"\n过滤后的数据已保存至：{west_pkl_processed_path}")

# 重新计算过滤后的统计信息
if len(high_ratio_ids) > 0:
    filtered_counts = filtered_traffic_flows['ID'].value_counts().reset_index()
    filtered_counts.columns = ['ID', 'filtered_count']

    # 与原始数据对比
    filtered_ratio_df = pd.merge(filtered_counts, original_counts, on='ID', how='inner')
    filtered_ratio_df['filtered_ratio'] = (
                filtered_ratio_df['filtered_count'] / filtered_ratio_df['original_count']).round(3)

    total_filtered = len(filtered_traffic_flows)
    filtered_overall_ratio = total_filtered / total_original

    print(f"\n过滤后总体保留比例：{filtered_overall_ratio:.4f} ({total_filtered}/{total_original})")
    print(f"过滤后保留比例统计：")
    print(f"平均保留比例：{filtered_ratio_df['filtered_ratio'].mean():.4f}")
    print(f"保留比例中位数：{filtered_ratio_df['filtered_ratio'].median():.4f}")

    # 显示过滤后保留比例最低的10个ID
    if len(filtered_ratio_df) > 0:
        print(f"\n过滤后保留比例最低的10个ID：")
        print(filtered_ratio_df.nsmallest(10, 'filtered_ratio')[
                  ['ID', 'filtered_count', 'original_count', 'filtered_ratio']])

# 显示被删除的ID列表
if len(low_ratio_ids) > 0:
    print(f"\n被删除的ID列表（保留比例 < {THRESHOLD}）：")
    deleted_ids_df = ratio_df[ratio_df['ID'].isin(low_ratio_ids)].copy()
    print(deleted_ids_df[['ID', 'processed_count', 'original_count', 'ratio']].sort_values('ratio'))

# 清理内存
del traffic_flows_west, traffic_flows_west_processed, ratio_df, filtered_traffic_flows
gc.collect()