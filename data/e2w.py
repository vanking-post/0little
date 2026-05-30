import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# --- 阈值参数控制 ---
THRESHOLD_LEFT_RIGHT_CHANGE = 100 # 普通变道（左/右变道）：100（4秒）
THRESHOLD_SECONDARY_FIRST_CHANGE = 100 # 二次变道的第一次变道：100帧（4秒）
THRESHOLD_SECONDARY_SECOND_CHANGE = 50 # 二次变道的第二次变道：100帧（4秒）
THRESHOLD_RAMP_FIRST_CHANGE = 100 # 匝道汇入第1次变道：200帧（8秒）
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50 # 匝道汇入后续变道：100帧（4秒）

# --- 数据路径 ---
save_dir = r"/read/CQSkyEyedata5/location5e"
file_path = os.path.join(save_dir, "traffic_flows_guiyi.pkl")

# 读取数据
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    df = pd.read_pickle(file_path)
    df = df.drop(columns=[col for col in ['Class', 'Length', 'Width'] if col in df.columns])
    df['ID'] = df['ID'].astype(int)
    df['LaneID'] = df['LaneID'].astype(int)

    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 排序
    df_sorted = df.sort_values(['ID', 'time']).reset_index(drop=True)
    all_vehicle_ids = df_sorted['ID'].unique()
    print(f"总共有 {len(all_vehicle_ids)} 辆车")

    # --- 容器初始化 ---
    lane_changes = {}
    ramp_categories = {k: [] for k in ['从1驶向匝道0', '匝道0从2驶离', '匝道0从3驶离',
                                       '匝道0从2变道至1后驶离', '匝道0从3变道至2后驶离', '匝道0从3变道至1后驶离']}
    maneuver_counts = defaultdict(int)
    maneuver_vehicles = defaultdict(list)
    source_tracking = {
        '左变道': {'普通变道': [], '二次变道': [], '驶离主路车辆': []},
        '右变道': {'普通变道': [], '二次变道': [], '驶离主路车辆': []}
    }
    sampling_data = []

    print("=== 车辆分类开始 ===")

    # --- 核心处理循环 ---
    for vehicle_id, group in df_sorted.groupby('ID'):
        lanes = group['LaneID'].values
        lane_changes[vehicle_id] = lanes.tolist()

        if len(lanes) < 2:
            continue

        # 获取去重车道序列及变道索引
        change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
        unique_seq = [lanes[0]] + lanes[change_idx].tolist()

        # 1. 匝道驶离逻辑 (驶入0车道)
        if lanes[-1] == 0:
            main_lanes = [l for l in unique_seq if l != 0]
            if not main_lanes: continue

            last_main = main_lanes[-1]
            category = None

            if last_main == 1:
                if len(main_lanes) == 1:
                    category = '从1驶向匝道0'
                else:
                    prev = main_lanes[-2]
                    if prev == 2:
                        category = '匝道0从2变道至1后驶离'
                    elif prev == 3:
                        category = '匝道0从3变道至1后驶离'
            elif last_main == 2:
                if len(main_lanes) == 1:
                    category = '匝道0从2驶离'
                else:
                    category = '匝道0从3变道至2后驶离' if main_lanes[-2] == 3 else '匝道0从2驶离'
            elif last_main == 3:
                category = '匝道0从3驶离'

            if category:
                ramp_categories[category].append(vehicle_id)
                # 关键过滤与采样：驶离主路车辆（第一次右变道，即车道号减小）
                if category in ['匝道0从2变道至1后驶离', '匝道0从3变道至2后驶离', '匝道0从3变道至1后驶离']:
                    if len(change_idx) > 0 and change_idx[0] >= THRESHOLD_RAMP_FIRST_CHANGE:
                        if lanes[change_idx[0]] < lanes[change_idx[0] - 1]:  # 右变道
                            source_tracking['右变道']['驶离主路车辆'].append(vehicle_id)
                            # 采样截取
                            if change_idx[0] >= 100:
                                sample = group.iloc[change_idx[0] - 100: change_idx[0] - 50].copy()
                                sample['Label'] = '右变道'
                                sampling_data.append(sample)

        # 2. 主路变道逻辑 (非0结尾，且包含1,2,3)
        elif any(l in [1, 2, 3] for l in lanes):
            if len(unique_seq) < 2:
                behavior = '跟驰'
            else:
                # 逻辑匹配：增加=左(L)，减小=右(R)
                changes = ['R' if unique_seq[i] < unique_seq[i - 1] else 'L' for i in range(1, len(unique_seq))]
                if len(changes) == 1:
                    behavior = '左变道' if changes[0] == 'L' else '右变道'
                else:
                    mapping = {('L', 'L'): '左变道后左变道', ('L', 'R'): '左变道后右变道',
                               ('R', 'L'): '右变道后左变道', ('R', 'R'): '右变道后右变道'}
                    behavior = mapping.get((changes[0], changes[-1]), '跟驰')

            maneuver_counts[behavior] += 1
            maneuver_vehicles[behavior].append(vehicle_id)

            # 3. 主路变道采样与过滤
            if behavior in ['左变道', '右变道', '跟驰'] or '后' in behavior:
                first_idx = change_idx[0] if len(change_idx) > 0 else 0

                # 跟驰采样
                if behavior == '跟驰' and len(group) >= 50:
                    s = random.randint(0, len(group) - 50)
                    sample = group.iloc[s:s + 50].copy()
                    sample['Label'] = '跟驰'
                    sampling_data.append(sample)

                # 普通/二次变道采样
                elif behavior in ['左变道', '右变道'] and first_idx >= THRESHOLD_LEFT_RIGHT_CHANGE:
                    source_tracking[behavior]['普通变道'].append(vehicle_id)
                    if first_idx >= 100:
                        sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                        sample['Label'] = behavior
                        sampling_data.append(sample)

                elif '后' in behavior and first_idx >= THRESHOLD_SECONDARY_FIRST_CHANGE:
                    label = '左变道' if behavior.startswith('左变道') else '右变道'
                    source_tracking[label]['二次变道'].append(vehicle_id)
                    if first_idx >= 100:
                        sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                        sample['Label'] = label
                        sampling_data.append(sample)

    # --- 打印输出 (严格保持原格式) ---
    classified_vehicles = set(
        [v for l in ramp_categories.values() for v in l] + [v for l in maneuver_vehicles.values() for v in l])
    unclassified_vehicles = set(all_vehicle_ids) - classified_vehicles

    print(
        f"\n=== 未分类车辆分析 ===\n总车辆数: {len(all_vehicle_ids)}\n已分类车辆数: {len(classified_vehicles)}\n未分类车辆数: {len(unclassified_vehicles)}")

    if unclassified_vehicles:
        print("\n未分类车辆ID及车道变化情况:")
        for i, vid in enumerate(sorted(unclassified_vehicles)[:20]):
            lanes = lane_changes[vid]
            u_seq = [lanes[0]] + [lanes[j] for j in range(1, len(lanes)) if lanes[j] != lanes[j - 1]]
            print(f"车辆ID: {vid}, 车道序列: {lanes[:10]}..., 去重后序列: {u_seq}")
            if (i + 1) % 10 == 0: print()

    print(f"\n=== 匝道驶离车辆细分 ===")
    for cat, vids in ramp_categories.items():
        if vids: print(f"{cat}: {len(vids)} 辆车")

    print(f"\n=== 车道变换行为统计 ===")
    for m_type, count in maneuver_counts.items():
        print(f"{m_type}: {count} 辆车")

    print(f"\n匝道驶离车辆总数: {sum(len(v) for v in ramp_categories.values())}")
    print(f"非匝道车辆总数: {sum(maneuver_counts.values())}")
    print(f"总计已分类车辆: {len(classified_vehicles)}")

    print("\n=== 车辆分类后统一过滤 ===")
    # 模拟过滤打印（由于处理已合并，此处直接输出分类结果以保持控制台反馈一致）
    for m in ['左变道', '右变道']:
        print(f"{m}（普通变道）汇总后: {len(source_tracking[m]['普通变道'])} 辆")
    for m in ['左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
        label = '左变道' if m.startswith('左变道') else '右变道'
        print(f"{m}汇总后: {len([v for v in maneuver_vehicles[m] if v in source_tracking[label]['二次变道']])} 辆")

    print(f"\n=== 样本截取开始 ===")
    print(f"左变道处理完成，右变道（含驶离）处理完成。")

    # --- 保存数据 ---
    if sampling_data:
        final_df = pd.concat(sampling_data, ignore_index=True)
        to_drop = ['time', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking', 'Dist_to_left_marking',
                   'Dist_to_right_marking']
        final_df = final_df.drop(columns=[c for c in to_drop if c in final_df.columns])

        final_df.to_pickle(os.path.join(save_dir, "traffic_flows_sampling.pkl"))
        final_df.to_csv(os.path.join(save_dir, "traffic_flows_sampling.csv"), index=False)

        print(f"\n合并后的采样数据形状: {final_df.shape}")
        print(f"标签分布:\n{final_df['Label'].value_counts()}")
        print(f"\n数据已保存至: {save_dir}")
    else:
        print("没有采样数据可合并")

    print(f"\n数据截取和处理完成！")