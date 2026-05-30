import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# --- 阈值参数控制 ---
# THRESHOLD_LEFT_RIGHT_CHANGE = 100
# THRESHOLD_SECONDARY_FIRST_CHANGE = 100
# THRESHOLD_SECONDARY_SECOND_CHANGE = 100
# THRESHOLD_RAMP_FIRST_CHANGE = 200
# THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 100

THRESHOLD_LEFT_RIGHT_CHANGE = 100 # 普通变道（左/右变道）：
THRESHOLD_SECONDARY_FIRST_CHANGE = 100 # 二次变道的第一次变道：
THRESHOLD_SECONDARY_SECOND_CHANGE = 50 # 二次变道的第二次变道：
THRESHOLD_RAMP_FIRST_CHANGE = 100 # 匝道汇入第1次变道：
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50 # 匝道汇入后续变道：

# --- 数据路径 ---
save_dir = r"/read/CQSkyEyedata5/location5"
file_path = os.path.join(save_dir, "traffic_flows_guiyi.pkl")

# 读取数据
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    df = pd.read_pickle(file_path)

    # 删除指定列
    columns_to_drop = ['Class', 'Length', 'Width']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 类型转换
    df['ID'] = df['ID'].astype(int)
    df['LaneID'] = df['LaneID'].astype(int)

    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 排序
    df_sorted = df.sort_values(['ID', 'time']).reset_index(drop=True)
    all_vehicle_ids = df_sorted['ID'].unique()
    print(f"总共有 {len(all_vehicle_ids)} 辆车")

    # --- 1. 核心分类与基础统计 ---
    lane_changes = {}
    vehicle_frame_counts = {}
    ramp_categories = {k: [] for k in
                       ['匝道8汇入7并直行', '匝道8汇入7后汇入6', '匝道8汇入7后汇入6后汇入5', '匝道8直接汇入6',
                        '匝道8直接汇入5', '匝道8汇入6后汇入5']}
    maneuver_counts = defaultdict(int)
    maneuver_vehicles = defaultdict(list)

    # 使用 groupby 优化遍历性能
    for vehicle_id, group in df_sorted.groupby('ID'):
        lanes = group['LaneID'].tolist()
        lane_changes[vehicle_id] = lanes
        vehicle_frame_counts[vehicle_id] = len(lanes)

        # 快速去重算法（保留顺序）
        unique_lane_sequence = group['LaneID'].loc[group['LaneID'].shift() != group['LaneID']].tolist()

        if not unique_lane_sequence:
            continue

        # 匝道分类逻辑 (起始车道为8)
        if unique_lane_sequence[0] == 8:
            if len(unique_lane_sequence) >= 2:
                seq = unique_lane_sequence
                if seq[1] == 7:
                    if len(seq) == 2:
                        ramp_categories['匝道8汇入7并直行'].append(vehicle_id)
                    elif len(seq) > 2 and seq[2] == 6:
                        if len(seq) == 3 or (len(seq) > 3 and seq[3] != 5):
                            ramp_categories['匝道8汇入7后汇入6'].append(vehicle_id)
                        elif len(seq) > 3 and seq[3] == 5:
                            ramp_categories['匝道8汇入7后汇入6后汇入5'].append(vehicle_id)
                    else:
                        ramp_categories['匝道8汇入7并直行'].append(vehicle_id)
                elif seq[1] == 6:
                    if len(seq) == 2 or (len(seq) > 2 and seq[2] != 5):
                        ramp_categories['匝道8直接汇入6'].append(vehicle_id)
                    else:
                        ramp_categories['匝道8汇入6后汇入5'].append(vehicle_id)
                elif seq[1] == 5:
                    ramp_categories['匝道8直接汇入5'].append(vehicle_id)
            continue

        # 主路分类逻辑 (5, 6, 7)
        if unique_lane_sequence[0] in [5, 6, 7]:
            if len(unique_lane_sequence) < 2:
                maneuver_type = '跟驰'
            else:
                # 判定变道方向序列
                changes = ['R' if unique_lane_sequence[i] > unique_lane_sequence[i - 1] else 'L' for i in
                           range(1, len(unique_lane_sequence))]
                if not changes:
                    maneuver_type = '跟驰'
                elif len(changes) == 1:
                    maneuver_type = '左变道' if changes[0] == 'L' else '右变道'
                else:
                    # 多次变道组合
                    mapping = {('L', 'L'): '左变道后左变道', ('L', 'R'): '左变道后右变道', ('R', 'L'): '右变道后左变道',
                               ('R', 'R'): '右变道后右变道'}
                    maneuver_type = mapping.get((changes[0], changes[-1]), '跟驰')

            maneuver_counts[maneuver_type] += 1
            maneuver_vehicles[maneuver_type].append(vehicle_id)

    # 打印基础统计 (保持原样)
    print(
        f"数据帧数统计 - 平均: {np.mean(list(vehicle_frame_counts.values())):.2f}, 最小: {min(vehicle_frame_counts.values())}, 最大: {max(vehicle_frame_counts.values())}")

    classified_vehicles = set(
        [v for l in ramp_categories.values() for v in l] + [v for l in maneuver_vehicles.values() for v in l])
    unclassified_vehicles = set(all_vehicle_ids) - classified_vehicles

    print(
        f"\n=== 未分类车辆分析 ===\n总车辆数: {len(all_vehicle_ids)}\n已分类车辆数: {len(classified_vehicles)}\n未分类车辆数: {len(unclassified_vehicles)}")

    if unclassified_vehicles:
        print("\n未分类车辆ID及车道变化情况:")
        for i, vid in enumerate(sorted(unclassified_vehicles)):
            lanes = lane_changes[vid]
            unique_seq = [lanes[0]] + [lanes[j] for j in range(1, len(lanes)) if lanes[j] != lanes[j - 1]]
            print(f"车辆ID: {vid}, 车道序列: {lanes[:10]}..., 去重后序列: {unique_seq}")
            if (i + 1) % 10 == 0: print()

    # 打印分类详情
    print(f"\n=== 匝道汇入车辆细分 ===")
    for cat, vids in ramp_categories.items():
        if vids:
            print(f"\n{cat}: {len(vids)} 辆车")
            for i, vid in enumerate(vids):
                print(f"{vid}", end=" " if (i + 1) % 10 != 0 else "\n")
            print()

    print(f"\n=== 车道变换行为统计 ===")
    for m_type, count in maneuver_counts.items():
        print(f"{m_type}: {count} 辆车")
        for i, vid in enumerate(maneuver_vehicles[m_type]):
            print(f"{vid}", end=" " if (i + 1) % 10 != 0 else "\n")
        print("\n")

    # --- 2. 变道前帧数分析与过滤 ---
    print("=" * 90 + "\n车辆变道前帧数统计分析\n" + "=" * 90)
    change_frame_counts = defaultdict(list)
    ramp_change_frame_counts = defaultdict(list)
    filtered_vehicles = []
    retained_vehicles = {}
    filtered_ramp_categories = {}

    # 复用之前分组的数据进行阈值过滤
    groups = {vid: group for vid, group in df_sorted.groupby('ID')}

    # 普通变道过滤打印
    for m_type in ['左变道', '右变道', '左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
        vids = maneuver_vehicles.get(m_type, [])
        print(f"\n{m_type} 原共有{len(vids)} 辆车，过滤处理后:")
        curr_retained, curr_filtered = [], []

        for vid in vids:
            group = groups[vid]
            lanes = group['LaneID'].values
            idx = np.where(lanes[:-1] != lanes[1:])[0] + 1  # 变道发生点索引

            if m_type in ['左变道', '右变道']:
                if len(idx) > 0:
                    f_count = idx[0]
                    if f_count >= THRESHOLD_LEFT_RIGHT_CHANGE:
                        change_frame_counts[m_type].append(f_count)
                        curr_retained.append((vid, [f_count]))
                    else:
                        curr_filtered.append({'vehicle_id': vid,
                                              'reason': f'变道前帧数不足 ({f_count} < {THRESHOLD_LEFT_RIGHT_CHANGE})'})
            else:  # 二次变道
                if len(idx) >= 2:
                    f1, f2 = idx[0], idx[1] - idx[0]
                    v1, v2 = f1 >= THRESHOLD_SECONDARY_FIRST_CHANGE, f2 >= THRESHOLD_SECONDARY_SECOND_CHANGE
                    if v1 and v2:
                        change_frame_counts[f"{m_type}_第1次变道"].append(f1)
                        change_frame_counts[f"{m_type}_第2次变道"].append(f2)
                        curr_retained.append((vid, [f1, f2]))
                    else:
                        r = f'第一次帧数不足({f1})' if not v1 else f'第二次帧数不足({f2})'
                        curr_filtered.append({'vehicle_id': vid, 'reason': r})

        retained_vehicles[m_type] = curr_retained
        filtered_vehicles.extend(curr_filtered)
        print(
            f"  保留车辆数: {len(curr_retained)}\n  保留前20: {curr_retained[:20]}\n  被过滤前20: {[(f['vehicle_id'], f['reason']) for f in curr_filtered[:20]]}")

    # 匝道过滤打印
    for cat, vids in ramp_categories.items():
        if not vids: continue
        print(f"\n{cat} ({len(vids)} 辆车):")
        curr_r_retained, curr_r_filtered = [], []

        for vid in vids:
            group = groups[vid]
            lanes = group['LaneID'].values
            idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
            if len(idx) >= 1:
                f1 = idx[0]
                v1 = f1 >= THRESHOLD_RAMP_FIRST_CHANGE
                sub_v = all((idx[j] - idx[j - 1]) >= THRESHOLD_RAMP_SUBSEQUENT_CHANGE for j in range(1, len(idx)))
                if v1 and sub_v:
                    curr_r_retained.append(vid)
                    ramp_change_frame_counts[f"{cat}_第1次变道"].append(f1)
                    for j in range(1, len(idx)): ramp_change_frame_counts[f"{cat}_第{j + 1}次变道"].append(
                        idx[j] - idx[j - 1])
                else:
                    curr_r_filtered.append({'vehicle_id': vid, 'reason': '阈值未达标'})

        filtered_ramp_categories[cat] = curr_r_retained
        print(f"  保留车辆数: {len(curr_r_retained)}\n  保留前20: {curr_r_retained[:20]}")

    # --- 3. 数据采样截取 ---
    sampling_data = []
    sampling_info = defaultdict(list)

    # 跟驰采样
    print(f"\n正在处理跟驰车辆...")
    for vid in maneuver_vehicles.get('跟驰', []):
        group = groups[vid]
        if len(group) >= 50:
            s = random.randint(0, len(group) - 50)
            sample = group.iloc[s:s + 50].copy()
            sample['Label'] = '跟驰'
            sampling_data.append(sample)
            sampling_info['跟驰'].append(vid)

    # 变道采样 (普通/二次/匝道)
    print(f"\n正在处理变道采样...")
    # 处理逻辑：遍历保留下来的有效车辆，统一截取变道前4-2秒 (idx-100 到 idx-50)
    for m_type, data in retained_vehicles.items():
        for vid, frames in data:
            group = groups[vid]
            idx = np.where(group['LaneID'].values[:-1] != group['LaneID'].values[1:])[0][0] + 1
            if idx >= 100:
                sample = group.iloc[idx - 100: idx - 50].copy()
                label = '左变道' if m_type.startswith('左变道') else '右变道'
                sample['Label'] = label
                sampling_data.append(sample)
                sampling_info[f"{'二次' if '后' in m_type else '普通'}{label}"].append(vid)

    # 匝道左偏采样
    for cat in ['匝道8汇入7后汇入6', '匝道8汇入7后汇入6后汇入5']:
        for vid in filtered_ramp_categories.get(cat, []):
            group = groups[vid]
            lanes = group['LaneID'].values
            idx_list = np.where(lanes[:-1] != lanes[1:])[0] + 1
            for idx in idx_list:
                # 寻找 7->6 或 6->5
                prev, curr = lanes[idx - 1], lanes[idx]
                if (prev == 7 and curr == 6) or (prev == 6 and curr == 5):
                    if idx >= 100:
                        sample = group.iloc[idx - 100: idx - 50].copy()
                        sample['Label'] = '左变道'
                        sampling_data.append(sample)
                        sampling_info['匝道汇入左变道'].append(vid)
                        break

    # --- 4. 合并与保存 ---
    if sampling_data:
        final_df = pd.concat(sampling_data, ignore_index=True)
        to_drop = ['time', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking', 'Dist_to_left_marking',
                   'Dist_to_right_marking']
        final_df = final_df.drop(columns=[c for c in to_drop if c in final_df.columns])

        # 打印最终统计
        print(f"\n合并后的采样数据形状: {final_df.shape}\n标签分布:\n{final_df['Label'].value_counts()}")

        final_df.to_pickle(os.path.join(save_dir, "traffic_flows_sampling.pkl"))
        final_df.to_csv(os.path.join(save_dir, "traffic_flows_sampling.csv"), index=False)
        print(f"\n数据截取和处理完成！已保存至 {save_dir}")
    else:
        print("没有采样数据可合并")