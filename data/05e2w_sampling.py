#读取traffic_flows_guiyi数据，输出traffic_flows_sampling
#从该数据guiyi中截取出可供模型训练使用的样本数据，截取的思路为变道点前100到50帧的行驶数据
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# --- 阈值参数控制 ---
THRESHOLD_LEFT_RIGHT_CHANGE = 100
THRESHOLD_SECONDARY_FIRST_CHANGE = 100
THRESHOLD_SECONDARY_SECOND_CHANGE = 50
THRESHOLD_RAMP_FIRST_CHANGE = 100
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50

# --- 数据路径 ---
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
file_path = os.path.join(save_dir, "traffic_flows_guiyi.pkl")

# 读取数据
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    df = pd.read_pickle(file_path)
    df = df.drop(columns=[col for col in ['Class', 'Length', 'Width'] if col in df.columns])
    df['ID'] = df['ID'].astype(int)
    df['LaneID'] = df['LaneID'].astype(int)

    # 排序
    df_sorted = df.sort_values(['ID', 'time']).reset_index(drop=True)
    all_vehicle_ids = df_sorted['ID'].unique()

    # --- 容器初始化 ---
    lane_changes = {}
    ramp_categories = {k: [] for k in ['从1驶向匝道0', '匝道0从2驶离', '匝道0从3驶离',
                                       '匝道0从2变道至1后驶离', '匝道0从3变道至2后驶离', '匝道0从3变道至1后驶离']}
    maneuver_counts = defaultdict(int)
    maneuver_vehicles = defaultdict(list)

    # 增加采样成功计数器
    source_sample_counts = {
        '左变道': {'普通变道': 0, '二次变道': 0},
        '右变道': {'普通变道': 0, '二次变道': 0, '驶离主路车辆': 0}
    }
    sampling_data = []

    print("=== 车辆分类与采样开始 ===")

    # --- 核心处理循环 ---
    for vehicle_id, group in df_sorted.groupby('ID'):
        lanes = group['LaneID'].values
        lane_changes[vehicle_id] = lanes.tolist()

        if len(lanes) < 2:
            continue

        change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
        unique_seq = [lanes[0]] + lanes[change_idx].tolist()

        # 1. 匝道驶离逻辑
        if lanes[-1] == 0:
            main_lanes = [l for l in unique_seq if l != 0]
            if not main_lanes: continue

            last_main = main_lanes[-1]
            category = None
            if last_main == 1:
                category = '从1驶向匝道0' if len(main_lanes) == 1 else (
                    '匝道0从2变道至1后驶离' if main_lanes[-2] == 2 else '匝道0从3变道至1后驶离')
            elif last_main == 2:
                category = '匝道0从2驶离' if len(main_lanes) == 1 else '匝道0从3变道至2后驶离'
            elif last_main == 3:
                category = '匝道0从3驶离'

            if category:
                ramp_categories[category].append(vehicle_id)
                # 采样：右变道（驶离）
                if category in ['匝道0从2变道至1后驶离', '匝道0从3变道至2后驶离', '匝道0从3变道至1后驶离']:
                    if len(change_idx) > 0 and change_idx[0] >= THRESHOLD_RAMP_FIRST_CHANGE:
                        if lanes[change_idx[0]] < lanes[change_idx[0] - 1] and change_idx[0] >= 100:
                            sample = group.iloc[change_idx[0] - 100: change_idx[0] - 50].copy()
                            sample['Label'] = '右变道'
                            sampling_data.append(sample)
                            source_sample_counts['右变道']['驶离主路车辆'] += 1

        # 2. 主路变道逻辑
        elif any(l in [1, 2, 3] for l in lanes):
            if len(unique_seq) < 2:
                behavior = '跟驰'
            else:
                changes = ['R' if unique_seq[i] < unique_seq[i - 1] else 'L' for i in range(1, len(unique_seq))]
                if len(changes) == 1:
                    behavior = '左变道' if changes[0] == 'L' else '右变道'
                else:
                    mapping = {('L', 'L'): '左变道后左变道', ('L', 'R'): '左变道后右变道',
                               ('R', 'L'): '右变道后左变道', ('R', 'R'): '右变道后右变道'}
                    behavior = mapping.get((changes[0], changes[-1]), '跟驰')

            maneuver_counts[behavior] += 1
            maneuver_vehicles[behavior].append(vehicle_id)

            # 3. 主路变道采样
            first_idx = change_idx[0] if len(change_idx) > 0 else 0
            if behavior == '跟驰' and len(group) >= 50:
                s = random.randint(0, len(group) - 50)
                sample = group.iloc[s:s + 50].copy()
                sample['Label'] = '跟驰'
                sampling_data.append(sample)
            elif behavior in ['左变道', '右变道'] and first_idx >= 100:
                sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                sample['Label'] = behavior
                sampling_data.append(sample)
                source_sample_counts[behavior]['普通变道'] += 1
            elif '后' in behavior and first_idx >= 100:
                label = '左变道' if behavior.startswith('左变道') else '右变道'
                sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                sample['Label'] = label
                sampling_data.append(sample)
                source_sample_counts[label]['二次变道'] += 1

    # --- 最终输出 ---
    print(f"\n=== 样本统计与保存 ===")
    if sampling_data:
        final_df = pd.concat(sampling_data, ignore_index=True)
        to_drop = ['time', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking', 'Dist_to_left_marking',
                   'Dist_to_right_marking']
        final_df = final_df.drop(columns=[c for c in to_drop if c in final_df.columns])

        final_df.to_pickle(os.path.join(save_dir, "traffic_flows_sampling.pkl"))
        final_df.to_csv(os.path.join(save_dir, "traffic_flows_sampling.csv"), index=False)

        print(f"数据处理完成，已保存至 {save_dir}")
        print("-" * 50)

        # 优化后的标签分布打印
        label_counts = final_df['Label'].value_counts()
        print(f"最终标签分布（总行数）:")
        for label, row_count in label_counts.items():
            vehicle_count = row_count // 50
            if label == '跟驰':
                print(f"  * {label}: {row_count} 行 (约 {vehicle_count} 辆车)")
            else:
                print(f"  * {label}: {row_count} 行 ({vehicle_count} 辆车)")
                # 打印来源细分
                for source, s_count in source_sample_counts[label].items():
                    if s_count > 0:
                        print(f"      - 来自{source}: {s_count} 辆")
        print("-" * 50)
    else:
        print("未提取到有效采样数据。")