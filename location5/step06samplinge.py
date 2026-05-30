import pandas as pd
import numpy as np
import random
from collections import defaultdict
import gc
import matplotlib.pyplot as plt  # 正确导入 pyplot


def process_east(df):
    print("\n--- 正在执行step06 east 向样本提取与标签分类 ---")
    sampling_data = []
    source_counts = {
        '左变道': defaultdict(int),
        '右变道': defaultdict(int),
        '跟驰': defaultdict(int)
    }
    ramp_lane = 0

    for vehicle_id, group in df.groupby('ID'):
        group = group.sort_values('time')
        lanes = group['LaneID'].values
        if len(lanes) < 2:
            continue

        change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1

        # 无变道 → 跟驰
        if len(change_idx) == 0:
            if len(group) >= 50:
                s = random.randint(0, len(group) - 50)
                sample = group.iloc[s:s + 50].copy()
                sample['Label'] = '跟驰'
                sampling_data.append(sample)
                source_counts['跟驰']['常规跟驰'] += 1
            continue

        # 有变道：只考虑第一次
        first_idx = change_idx[0]
        prev_lane = lanes[first_idx - 1]
        cur_lane = lanes[first_idx]

        # 排除 1→0 驶离匝道
        if prev_lane == 1 and cur_lane == ramp_lane:
            continue

        behavior = '左变道' if cur_lane > prev_lane else '右变道'
        if first_idx >= 100:
            sample = group.iloc[first_idx - 100:first_idx - 50].copy()
            sample['Label'] = behavior
            sampling_data.append(sample)
            source_counts[behavior]['普通变道'] += 1

    # ---------- 合并样本并输出统计 ----------
    if not sampling_data:
        print("⚠️ east 向未提取到符合条件的样本")
        return pd.DataFrame()

    res_df = pd.concat(sampling_data, ignore_index=True)

    # ---------- 删除不需要的列，返回最终结果 ----------
    to_drop = ['Dist_to_right_edge_marking',
               'Dist_to_left_marking', 'Dist_to_right_marking']
    res_df = res_df.drop(columns=[c for c in to_drop if c in res_df.columns])

    print(f"✅ east 向样本提取完成：提取了 {len(res_df) // 50} 辆车的样本片段")
    for label, sub_src in source_counts.items():
        total = sum(sub_src.values())
        if total > 0:
            print(f"   - {label}: {total} 辆 (来源: {dict(sub_src)})")
    return res_df


# if __name__ == "__main__":
#     # 读取数据（请根据实际文件路径修改）
#     raw_csv_path = r"E:\0little\read\CQSkyEyedata5\location5t\traffic_flows_east_sample.csv"
#     traffic_flows = pd.read_csv(raw_csv_path, index_col=False, low_memory=False, encoding="gbk")
#     df_east = process_east(traffic_flows)