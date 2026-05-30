import pandas as pd
import numpy as np
import os
import time
import random
from collections import defaultdict

# 阈值参数控制
THRESHOLD_LEFT_RIGHT_CHANGE = 100  # 普通变道（左/右变道）：100（4秒）
THRESHOLD_SECONDARY_FIRST_CHANGE = 100  # 二次变道的第一次变道：100帧（4秒）
THRESHOLD_SECONDARY_SECOND_CHANGE = 100  # 二次变道的第二次变道：100帧（4秒）
THRESHOLD_RAMP_FIRST_CHANGE = 200  # 匝道汇入第1次变道：200帧（8秒）
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 100  # 匝道汇入后续变道：100帧（4秒）

# 数据路径
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
files = [
    ("Raw", "traffic_flows_west.pkl"),  # 源数据库
    ("guiyi", "traffic_flows_guiyi.pkl"),  # 补全数据库
    ("sampling", "traffic_flows_sampling.pkl")]  # 样本化数据库

# 读取补全数据库
file_path = os.path.join(save_dir, files[1][1])  # traffic_flows_guiyi.pkl
df = pd.read_pickle(file_path)

# 删除指定列
columns_to_drop = ['Class', 'Length', 'Width']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# 将车辆ID和车道ID转换为整数
df['ID'] = df['ID'].astype(int)
df['LaneID'] = df['LaneID'].astype(int)

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 获取所有车辆ID
all_vehicle_ids = df['ID'].unique()
print(f"总共有 {len(all_vehicle_ids)} 辆车")

# 按照时间排序数据
df_sorted = df.sort_values(['ID', 'time']).reset_index(drop=True)

# 记录每辆车的车道变化和数据帧数
lane_changes = {}
vehicle_frame_counts = {}

for vehicle_id in all_vehicle_ids:
    vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
    lane_sequence = vehicle_data['LaneID'].tolist()
    lane_changes[vehicle_id] = lane_sequence
    vehicle_frame_counts[vehicle_id] = len(vehicle_data)  # 记录每辆车的数据帧数

print(f"数据帧数统计 - 平均: {np.mean(list(vehicle_frame_counts.values())):.2f}, "
      f"最小: {min(vehicle_frame_counts.values())}, "
      f"最大: {max(vehicle_frame_counts.values())}")

# 匝道汇入车辆细分
ramp_categories = {
    '匝道8汇入7并直行': [],
    '匝道8汇入7后汇入6': [],
    '匝道8汇入7后汇入6后汇入5': [],
    '匝道8直接汇入6': [],
    '匝道8直接汇入5': [],
    '匝道8汇入6后汇入5': []
}

for vehicle_id, lanes in lane_changes.items():
    if lanes and lanes[0] == 8:  # 从8车道开始
        # 去除连续重复的车道值
        unique_lane_sequence = [lanes[0]]
        for i in range(1, len(lanes)):
            if lanes[i] != lanes[i - 1]:
                unique_lane_sequence.append(lanes[i])

        # 根据车道变化序列进行分类
        if len(unique_lane_sequence) >= 2:
            if unique_lane_sequence[1] == 7:  # 从8汇入7
                if len(unique_lane_sequence) == 2:  # 只有8->7，然后直行
                    ramp_categories['匝道8汇入7并直行'].append(vehicle_id)
                elif len(unique_lane_sequence) > 2 and unique_lane_sequence[2] == 6:  # 8->7->6
                    if len(unique_lane_sequence) == 3:  # 只有8->7->6
                        ramp_categories['匝道8汇入7后汇入6'].append(vehicle_id)
                    elif len(unique_lane_sequence) > 3 and unique_lane_sequence[3] == 5:  # 8->7->6->5
                        ramp_categories['匝道8汇入7后汇入6后汇入5'].append(vehicle_id)
                    else:  # 8->7->6->...
                        ramp_categories['匝道8汇入7后汇入6'].append(vehicle_id)
                else:  # 8->7->其他
                    ramp_categories['匝道8汇入7并直行'].append(vehicle_id)
            elif unique_lane_sequence[1] == 6:  # 从8直接汇入6
                if len(unique_lane_sequence) == 2:  # 只有8->6
                    ramp_categories['匝道8直接汇入6'].append(vehicle_id)
                elif len(unique_lane_sequence) > 2 and unique_lane_sequence[2] == 5:  # 8->6->5
                    if len(unique_lane_sequence) == 3:  # 只有8->6->5
                        ramp_categories['匝道8汇入6后汇入5'].append(vehicle_id)
                    else:  # 8->6->5->...
                        ramp_categories['匝道8汇入6后汇入5'].append(vehicle_id)
                else:  # 8->6->其他
                    ramp_categories['匝道8直接汇入6'].append(vehicle_id)
            elif unique_lane_sequence[1] == 5:  # 从8直接汇入5
                ramp_categories['匝道8直接汇入5'].append(vehicle_id)

# 车道变换行为分析（非匝道车辆）
maneuver_counts = defaultdict(int)
maneuver_vehicles = defaultdict(list)

for vehicle_id, lanes in lane_changes.items():
    if not lanes or len(lanes) < 2:
        continue

    # 如果是从8车道开始的，跳过匝道汇入车辆
    if lanes[0] == 8:
        continue

    # 只考虑从5、6、7车道开始的车辆
    if lanes[0] not in [5, 6, 7]:
        continue

    # 去除连续重复的车道值
    unique_lane_sequence = [lanes[0]]
    for i in range(1, len(lanes)):
        if lanes[i] != lanes[i - 1]:
            unique_lane_sequence.append(lanes[i])

    # 如果去重后的序列长度小于2，说明没有真正的车道变化
    if len(unique_lane_sequence) < 2:
        maneuver_type = '跟驰'
    else:
        # 分析车道变化模式
        changes = []
        change_indices = []  # 记录变道发生的位置索引
        for i in range(1, len(unique_lane_sequence)):
            if unique_lane_sequence[i] > unique_lane_sequence[i - 1]:
                changes.append('R')  # 向右变道
                # 找到在原始序列中该变道发生的位置
                for j in range(len(lanes)):
                    if lanes[j] == unique_lane_sequence[i - 1]:
                        continue
                    elif lanes[j] == unique_lane_sequence[i]:
                        change_indices.append(j)
                        break
            elif unique_lane_sequence[i] < unique_lane_sequence[i - 1]:
                changes.append('L')  # 向左变道
                # 找到在原始序列中该变道发生的位置
                for j in range(len(lanes)):
                    if lanes[j] == unique_lane_sequence[i - 1]:
                        continue
                    elif lanes[j] == unique_lane_sequence[i]:
                        change_indices.append(j)
                        break

        # 判断驾驶行为类型
        if not changes:
            maneuver_type = '跟驰'
        elif len(changes) == 1:
            if changes[0] == 'L':
                maneuver_type = '左变道'
            else:
                maneuver_type = '右变道'
        else:
            # 多次变道的情况
            if changes[0] == 'L':
                if changes[-1] == 'L':
                    maneuver_type = '左变道后左变道'
                else:
                    maneuver_type = '左变道后右变道'
            else:  # 第一次是右变道
                if changes[-1] == 'L':
                    maneuver_type = '右变道后左变道'
                else:
                    maneuver_type = '右变道后右变道'

    maneuver_counts[maneuver_type] += 1
    maneuver_vehicles[maneuver_type].append(vehicle_id)

# 收集所有已分类的车辆ID
classified_vehicles = set()
for vehicles in ramp_categories.values():
    classified_vehicles.update(vehicles)
for vehicles in maneuver_vehicles.values():
    classified_vehicles.update(vehicles)

# 找出未分类的车辆
unclassified_vehicles = set(all_vehicle_ids) - classified_vehicles
print(f"\n=== 未分类车辆分析 ===")
print(f"总车辆数: {len(all_vehicle_ids)}")
print(f"已分类车辆数: {len(classified_vehicles)}")
print(f"未分类车辆数: {len(unclassified_vehicles)}")

if unclassified_vehicles:
    print("\n未分类车辆ID及车道变化情况:")
    for i, vehicle_id in enumerate(sorted(unclassified_vehicles)):
        lanes = lane_changes[vehicle_id]

        # 去除连续重复的车道值
        unique_lane_sequence = [lanes[0]]
        for j in range(1, len(lanes)):
            if lanes[j] != lanes[j - 1]:
                unique_lane_sequence.append(lanes[j])

        print(
            f"车辆ID: {vehicle_id}, 车道序列: {lanes[:10]}{'...' if len(lanes) > 10 else ''}, 去重后序列: {unique_lane_sequence}")

        # 分析原因
        if not lanes:
            reason = "车道数据为空"
        elif len(lanes) < 2:
            reason = "车道数据长度不足"
        elif lanes[0] == 8 and len(set(lanes)) == 1:
            reason = "从匝道8开始但一直保持在匝道8，未汇入主路"
        elif lanes[0] not in [5, 6, 7, 8]:
            reason = f"起始车道不在预期范围内({lanes[0]})"
        elif lanes[0] in [5, 6, 7] and len(unique_lane_sequence) < 2:
            reason = "主路车辆但车道无变化"
        else:
            reason = "其他原因"

        print(f"  - 原因: {reason}")

        if (i + 1) % 10 == 0:
            print()  # 每10个换行

print(f"\n=== 匝道汇入车辆细分 ===")
for category, vehicles in ramp_categories.items():
    if vehicles:  # 只打印有车辆的类别
        print(f"\n{category}: {len(vehicles)} 辆车")
        for i, vid in enumerate(vehicles):
            if i % 10 == 0 and i != 0:
                print()  # 每10个换行
            print(f"{vid}", end=" ")
        if len(vehicles) % 10 != 0:
            print()  # 如果最后一行不满10个，换行

print(f"\n\n=== 车道变换行为统计 ===")
for maneuver_type, count in maneuver_counts.items():
    print(f"{maneuver_type}: {count} 辆车")
    vehicles_list = maneuver_vehicles[maneuver_type]
    for i, vid in enumerate(vehicles_list):
        if i % 10 == 0 and i != 0:
            print()  # 每10个换行
        print(f"{vid}", end=" ")
    if len(vehicles_list) % 10 != 0:
        print()  # 如果最后一行不满10个，换行
    print()

total_ramp_vehicles = sum(len(vehicles) for vehicles in ramp_categories.values())
print(f"\n匝道汇入车辆总数: {total_ramp_vehicles}")
print(f"非匝道车辆总数: {sum(maneuver_counts.values())}")
print(f"总计已分类车辆: {len(classified_vehicles)}")

# 计算变道前帧数
print("\n" + "=" * 90)
print("车辆变道前帧数统计分析")
print("=" * 90)

# 存储变道前帧数信息
change_frame_counts = defaultdict(list)
ramp_change_frame_counts = defaultdict(list)

# 存储被过滤的车辆信息
filtered_vehicles = []

# 计算普通变道车辆的变道前帧数
retained_vehicles = {}  # 存储保留下来的车辆信息
for maneuver_type in ['左变道', '右变道', '左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
    if maneuver_type in maneuver_vehicles:
        vehicles = maneuver_vehicles[maneuver_type]
        print(f"\n{maneuver_type} 原共有{len(vehicles)} 辆车，过滤处理后:")

        # 存储当前类型的所有变道前帧数
        current_frame_counts = []
        current_filtered = []

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
            lanes = vehicle_data['LaneID'].tolist()

            # 找到所有变道点
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)  # 记录变道发生的索引

            if maneuver_type in ['左变道', '右变道']:
                # 单次变道，只记录第一次变道前的帧数
                if change_indices:
                    frame_count = change_indices[0]
                    # 应用阈值过滤
                    if frame_count >= THRESHOLD_LEFT_RIGHT_CHANGE:
                        change_frame_counts[maneuver_type].append(frame_count)
                        current_frame_counts.append((vehicle_id, [frame_count]))
                    else:
                        current_filtered.append({
                            'vehicle_id': vehicle_id,
                            'frame_count': frame_count,
                            'threshold': THRESHOLD_LEFT_RIGHT_CHANGE,
                            'reason': f'变道前帧数不足 ({frame_count} < {THRESHOLD_LEFT_RIGHT_CHANGE})'
                        })
            else:
                # 二次变道，记录每次变道前的帧数
                if len(change_indices) >= 2:
                    # 第一次变道前的帧数
                    first_change_frame = change_indices[0]
                    # 第二次变道前的帧数（相对于第一次变道后的帧数）
                    second_change_frame = change_indices[1] - change_indices[0]

                    # 应用阈值过滤
                    first_valid = first_change_frame >= THRESHOLD_SECONDARY_FIRST_CHANGE
                    second_valid = second_change_frame >= THRESHOLD_SECONDARY_SECOND_CHANGE

                    if first_valid and second_valid:
                        change_frame_counts[f"{maneuver_type}_第1次变道"].append(first_change_frame)
                        change_frame_counts[f"{maneuver_type}_第2次变道"].append(second_change_frame)
                        current_frame_counts.append((vehicle_id, [first_change_frame, second_change_frame]))
                    else:
                        if not first_valid:
                            current_filtered.append({
                                'vehicle_id': vehicle_id,
                                'frame_count': first_change_frame,
                                'threshold': THRESHOLD_SECONDARY_FIRST_CHANGE,
                                'reason': f'第一次变道前帧数不足 ({first_change_frame} < {THRESHOLD_SECONDARY_FIRST_CHANGE})'
                            })
                        if not second_valid:
                            current_filtered.append({
                                'vehicle_id': vehicle_id,
                                'frame_count': second_change_frame,
                                'threshold': THRESHOLD_SECONDARY_SECOND_CHANGE,
                                'reason': f'第二次变道前帧数不足 ({second_change_frame} < {THRESHOLD_SECONDARY_SECOND_CHANGE})'
                            })
                elif len(change_indices) == 1:
                    # 只有一次变道
                    frame_count = change_indices[0]
                    # 应用阈值过滤
                    if frame_count >= THRESHOLD_LEFT_RIGHT_CHANGE:
                        change_frame_counts[maneuver_type].append(frame_count)
                        current_frame_counts.append((vehicle_id, [frame_count]))
                    else:
                        current_filtered.append({
                            'vehicle_id': vehicle_id,
                            'frame_count': frame_count,
                            'threshold': THRESHOLD_LEFT_RIGHT_CHANGE,
                            'reason': f'变道前帧数不足 ({frame_count} < {THRESHOLD_LEFT_RIGHT_CHANGE})'
                        })

        # 存储保留的车辆信息
        retained_vehicles[maneuver_type] = current_frame_counts
        # 存储过滤的车辆信息
        filtered_vehicles.extend(current_filtered)

        # 打印保留的车辆信息（前20条）
        print(f"  保留车辆数: {len(current_frame_counts)}")
        print("  保留车辆ID和帧数 (前20条):")
        for i, (vid, frame_counts) in enumerate(current_frame_counts[:20]):
            if len(frame_counts) == 1:
                print(f"    车辆 {vid}: {frame_counts[0]} 帧")
            elif len(frame_counts) == 2:
                print(f"    车辆 {vid} (1次: {frame_counts[0]}, 2次: {frame_counts[1]})")

        # 打印被过滤的车辆信息（前20条）
        if current_filtered:
            print(f"  被过滤车辆数: {len(current_filtered)}")
            print("  被过滤车辆ID和原因 (前20条):")
            for i, info in enumerate(current_filtered[:20]):
                print(f"    车辆 {info['vehicle_id']}: {info['reason']}")

# 计算匝道汇入车辆的变道前帧数 - 修正版本
filtered_ramp_categories = {}  # 存储过滤后的匝道车辆
for category, vehicles in ramp_categories.items():
    if vehicles:
        print(f"\n{category} ({len(vehicles)} 辆车):")
        current_ramp_frame_counts = []
        current_ramp_filtered = []
        current_ramp_retained = []  # 存储保留的车辆ID

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
            lanes = vehicle_data['LaneID'].tolist()

            # 找到所有变道点（从8车道汇入其他车道）
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)  # 记录变道发生的索引

            # 记录每次变道前的帧数（相对上一次变道）
            if len(change_indices) >= 1:
                ramp_frame_counts = [change_indices[0]]  # 第一次变道前的帧数
                first_valid = change_indices[0] >= THRESHOLD_RAMP_FIRST_CHANGE

                # 计算后续变道相对于前一次变道的帧数
                subsequent_valid = True
                for j in range(1, len(change_indices)):
                    relative_frame_count = change_indices[j] - change_indices[j - 1]
                    if relative_frame_count >= THRESHOLD_RAMP_SUBSEQUENT_CHANGE:
                        ramp_frame_counts.append(relative_frame_count)
                    else:
                        subsequent_valid = False
                        current_ramp_filtered.append({
                            'vehicle_id': vehicle_id,
                            'frame_count': relative_frame_count,
                            'threshold': THRESHOLD_RAMP_SUBSEQUENT_CHANGE,
                            'reason': f'{j + 1}次变道前帧数不足 ({relative_frame_count} < {THRESHOLD_RAMP_SUBSEQUENT_CHANGE})'
                        })

                # 如果所有条件都满足，则添加到有效数据中
                if first_valid and subsequent_valid:
                    ramp_change_frame_counts[f"{category}_第1次变道"].append(change_indices[0])
                    current_ramp_frame_counts.append((vehicle_id, ramp_frame_counts))
                    current_ramp_retained.append(vehicle_id)  # 记录保留的车辆ID
                    # 添加后续变道数据
                    for j in range(1, len(change_indices)):
                        relative_frame_count = change_indices[j] - change_indices[j - 1]
                        ramp_change_frame_counts[f"{category}_第{j + 1}次变道"].append(relative_frame_count)
                else:
                    if not first_valid:
                        current_ramp_filtered.append({
                            'vehicle_id': vehicle_id,
                            'frame_count': change_indices[0],
                            'threshold': THRESHOLD_RAMP_FIRST_CHANGE,
                            'reason': f'第1次变道前帧数不足 ({change_indices[0]} < {THRESHOLD_RAMP_FIRST_CHANGE})'
                        })

        # 存储保留的匝道车辆信息
        filtered_ramp_categories[category] = current_ramp_retained
        # 存储过滤的匝道车辆信息
        filtered_vehicles.extend(current_ramp_filtered)

        # 打印保留的车辆信息（前20条）
        print(f"  保留车辆数: {len(current_ramp_retained)}")
        print("  保留车辆ID (前20条):")
        for i, vid in enumerate(current_ramp_retained[:20]):
            if i % 10 == 0 and i != 0:
                print()  # 每10个换行
            print(f"{vid}", end=" ")
        if len(current_ramp_retained) % 10 != 0:
            print()  # 如果最后一行不满10个，换行

        # 打印被过滤的车辆信息（前20条）
        if current_ramp_filtered:
            print(f"  被过滤车辆数: {len(current_ramp_filtered)}")
            print("  被过滤车辆ID和原因 (前20条):")
            for i, info in enumerate(current_ramp_filtered[:20]):
                print(f"    车辆 {info['vehicle_id']}: {info['reason']}")

# 开始数据截取 - 优化版本
sampling_data = []
sampling_info = defaultdict(list)  # 记录样本来源信息

# 截取跟驰车辆数据
print(f"\n正在处理跟驰车辆...")
if '跟驰' in maneuver_vehicles:
    following_vehicles = maneuver_vehicles['跟驰']
    print(f"共有 {len(following_vehicles)} 辆跟驰车辆")

    for vehicle_id in following_vehicles:
        vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

        # 确保数据按时间排序
        vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

        total_frames = len(vehicle_data)

        # 如果车辆数据少于50帧，跳过
        if total_frames < 50:
            continue

        # 随机选择一个起始点（确保有足够的后续数据）
        start_idx = random.randint(0, total_frames - 50)
        sampled_data = vehicle_data.iloc[start_idx:start_idx + 50].copy()

        # 添加标签列
        sampled_data['Label'] = '跟驰'
        sampling_info['跟驰'].append(vehicle_id)
        sampling_data.append(sampled_data)

# 截取普通变道车辆数据 - 优化算法
print(f"\n正在处理普通变道车辆...")


def extract_lane_change_data(vehicle_id, maneuver_type, required_frames=100, label=None):
    """提取变道车辆数据的核心函数"""
    vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

    # 确保数据按时间排序
    vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

    # 检查数据帧数是否足够
    total_frames = len(vehicle_data)
    if total_frames < required_frames:
        print(f"  车辆 {vehicle_id} 数据不足 ({total_frames} < {required_frames})，跳过")
        return None

    # 找到变道发生的位置
    lanes = vehicle_data['LaneID'].tolist()
    change_indices = []
    for i in range(1, len(lanes)):
        if lanes[i] != lanes[i - 1]:
            change_indices.append(i)

    if not change_indices:
        print(f"  车辆 {vehicle_id} 未检测到变道，跳过")
        return None

    # 取第一个变道点
    first_change_idx = change_indices[0]

    # 检查变道前是否有足够的数据
    frames_before_change = first_change_idx
    if frames_before_change < required_frames:
        print(f"  车辆 {vehicle_id} 变道前数据不足 ({frames_before_change} < {required_frames})，跳过")
        return None

    # 提取变道前4-2秒的数据（从first_change_idx-100到first_change_idx-50，共50帧）
    start_idx = first_change_idx - 100
    end_idx = first_change_idx - 50

    # 确保索引在有效范围内
    if start_idx < 0:
        start_idx = 0
    if end_idx > len(vehicle_data):
        end_idx = len(vehicle_data)

    # 确保提取的帧数为50帧
    if end_idx - start_idx >= 50:
        sampled_data = vehicle_data.iloc[start_idx:start_idx + 50].copy()
    else:
        # 如果不够50帧，提取可用的最大数据
        available_frames = end_idx - start_idx
        if available_frames > 0:
            sampled_data = vehicle_data.iloc[start_idx:end_idx].copy()
        else:
            print(f"  车辆 {vehicle_id} 无法提取有效数据，跳过")
            return None

    # 添加标签列
    if label:
        sampled_data['Label'] = label
    else:
        if maneuver_type == '左变道':
            sampled_data['Label'] = '左变道'
        else:
            sampled_data['Label'] = '右变道'

    return sampled_data


# 处理左变道车辆
left_change_vehicles = maneuver_vehicles.get('左变道', [])
print(f"左变道车辆: 原共有{len(left_change_vehicles)} 辆")
processed_left_count = 0

for vehicle_id in left_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '左变道', required_frames=100, label='左变道')
    if result is not None:
        sampling_info['普通左变道'].append(vehicle_id)
        sampling_data.append(result)
        processed_left_count += 1

print(f"成功处理左变道车辆: {processed_left_count} 辆")

# 处理右变道车辆
right_change_vehicles = maneuver_vehicles.get('右变道', [])
print(f"右变道车辆: 原共有{len(right_change_vehicles)} 辆")
processed_right_count = 0

for vehicle_id in right_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '右变道', required_frames=100, label='右变道')
    if result is not None:
        sampling_info['普通右变道'].append(vehicle_id)
        sampling_data.append(result)
        processed_right_count += 1

print(f"成功处理右变道车辆: {processed_right_count} 辆")

# 截取二次变道车辆数据（只处理第一次变道）
print(f"\n正在处理二次变道车辆...")
for maneuver_type in ['左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
    if maneuver_type in maneuver_vehicles:
        vehicles = maneuver_vehicles[maneuver_type]
        print(f"{maneuver_type}: {len(vehicles)} 辆车")

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

            # 确保数据按时间排序
            vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

            # 找到变道发生的位置
            lanes = vehicle_data['LaneID'].tolist()
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)

            if len(change_indices) < 2:
                continue  # 不足两次变道，跳过

            # 取第一次变道点
            first_change_idx = change_indices[0]

            # 确保变道前有足够的数据（至少100帧用于提取4-2秒区间）
            if first_change_idx < 100:
                continue

            # 提取变道前4-2秒的数据（从first_change_idx-100到first_change_idx-50，共50帧）
            start_idx = first_change_idx - 100
            end_idx = first_change_idx - 50
            sampled_data = vehicle_data.iloc[start_idx:end_idx].copy()

            # 添加标签列（根据第一次变道方向决定标签）
            if maneuver_type.startswith('左变道'):
                sampled_data['Label'] = '左变道'
                sampling_info['二次变道第一次左变道'].append(vehicle_id)
            else:
                sampled_data['Label'] = '右变道'
                sampling_info['二次变道第一次右变道'].append(vehicle_id)

            sampling_data.append(sampled_data)

# 截取匝道汇入车辆数据（匝道8汇入主路7后，由车道7汇入6以及车道6汇入车道5的这两部分左变道数据）
print(f"\n正在处理匝道汇入车辆数据...")
ramp_categories_for_extraction = ['匝道8汇入7后汇入6', '匝道8汇入7后汇入6后汇入5']
for category in ramp_categories_for_extraction:
    if category in filtered_ramp_categories:  # 修改：使用过滤后的车辆列表
        vehicles = filtered_ramp_categories[category]  # 修改：使用过滤后的车辆列表
        print(f"{category} (过滤后): {len(vehicles)} 辆车")

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

            # 确保数据按时间排序
            vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

            # 找到变道发生的位置
            lanes = vehicle_data['LaneID'].tolist()
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)

            # 找到从8→7，7→6，6→5的变道点
            for idx in change_indices:
                # 检查当前变道是否符合7→6或6→5
                if idx > 0 and idx < len(lanes):
                    prev_lane = lanes[idx - 1]
                    curr_lane = lanes[idx]

                    # 7→6 或 6→5 的变道
                    if (prev_lane == 7 and curr_lane == 6) or (prev_lane == 6 and curr_lane == 5):
                        # 检查变道前是否有足够的数据（至少100帧用于提取4-2秒区间）
                        if idx >= 100:
                            # 提取变道前4-2秒的数据（从idx-100到idx-50，共50帧）
                            start_idx = idx - 100
                            end_idx = idx - 50
                            sampled_data = vehicle_data.iloc[start_idx:end_idx].copy()

                            # 添加标签列
                            sampled_data['Label'] = '左变道'
                            sampling_info['匝道汇入左变道'].append(vehicle_id)

                            sampling_data.append(sampled_data)
                            break  # 每辆车只处理一个符合条件的变道点

# 合并所有采样数据
print(f"\n合并采样数据...")
if sampling_data:
    combined_sampling_df = pd.concat(sampling_data, ignore_index=True)

    print(f"合并后的采样数据形状: {combined_sampling_df.shape}")

    # 删除指定列 - 更新为新的列名
    columns_to_remove = ['time',  'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking',
                         'Dist_to_left_marking', 'Dist_to_right_marking']
    final_sampling_df = combined_sampling_df.drop(
        columns=[col for col in columns_to_remove if col in combined_sampling_df.columns])

    print(f"删除指定列后的数据形状: {final_sampling_df.shape}")

    # 显示标签分布
    print(f"\n标签分布:")
    if 'Label' in final_sampling_df.columns:
        label_counts = final_sampling_df['Label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} 行")

    # 统计各部分车辆数量
    left_change_count = len([data for data in sampling_data if data['Label'].iloc[0] == '左变道'])
    right_change_count = len([data for data in sampling_data if data['Label'].iloc[0] == '右变道'])
    following_count = len([data for data in sampling_data if data['Label'].iloc[0] == '跟驰'])

    print(f"\n合并后数据统计:")
    print(f"  总行数: {len(final_sampling_df)}")
    print(f"  左变道车辆数: {left_change_count}")
    print(f"  右变道车辆数: {right_change_count}")
    print(f"  跟驰车辆数: {following_count}")

    print(f"\n左变道车辆来源分析:")
    for key, vehicles in sampling_info.items():
        if '左变道' in key:
            print(f"  {key}: {len(vehicles)} 辆车 - {vehicles[:5]}{'...' if len(vehicles) > 5 else ''}")

    print(f"\n右变道车辆来源分析:")
    for key, vehicles in sampling_info.items():
        if '右变道' in key:
            print(f"  {key}: {len(vehicles)} 辆车 - {vehicles[:5]}{'...' if len(vehicles) > 5 else ''}")

    # 保存数据
    csv_path = os.path.join(save_dir, "traffic_flows_sampling.csv")
    pkl_path = os.path.join(save_dir, "traffic_flows_sampling.pkl")

    final_sampling_df.to_csv(csv_path, index=False)
    final_sampling_df.to_pickle(pkl_path)

    print(f"\n数据已保存到:")
    print(f"  CSV: {csv_path}")
    print(f"  PKL: {pkl_path}")

else:
    print("没有采样数据可合并")

print(f"\n数据截取和处理完成！")
