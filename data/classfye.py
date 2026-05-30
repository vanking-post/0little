import pandas as pd
import numpy as np
import os
import time
import random
from collections import defaultdict

# 阈值参数控制
THRESHOLD_LEFT_RIGHT_CHANGE = 100  # 普通变道（左/右变道）：建议至少100帧（4秒）
THRESHOLD_SECONDARY_FIRST_CHANGE = 100  # 二次变道的第一次变道：建议至少100帧（4秒）
THRESHOLD_SECONDARY_SECOND_CHANGE = 100  # 二次变道的第二次变道：建议至少100帧（4秒）
THRESHOLD_RAMP_FIRST_CHANGE = 200  # 匝道驶离第1次变道：建议至少200帧（8秒）
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50  # 匝道驶离后续变道：建议至少50帧（2秒）

# 数据路径
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
files = [
    ("guiyi", "traffic_flows_guiyi.pkl"),  # 补全数据库
    ("sampling", "traffic_flows_sampling.pkl")]  # 样本化数据库

# 读取补全数据库
file_path = os.path.join(save_dir, files[0][1])  # traffic_flows_guiyi.pkl
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

# 匝道驶离车辆细分
ramp_categories = {
    '从1驶向匝道0': [],
    '匝道0从2驶离': [],
    '匝道0从3驶离': [],
    '匝道0从2变道至1后驶离': [],
    '匝道0从3变道至2后驶离': [],
    '匝道0从3变道至1后驶离': []
}

for vehicle_id, lanes in lane_changes.items():
    if lanes and lanes[-1] == 0:  # 最终驶入0车道（驶离）
        # 去除连续重复的车道值
        unique_lane_sequence = [lanes[0]]
        for i in range(1, len(lanes)):
            if lanes[i] != lanes[i - 1]:
                unique_lane_sequence.append(lanes[i])

        # 获取驶离前的车道（最后一个非0车道）
        main_lanes_before_exit = [lane for lane in unique_lane_sequence if lane != 0]
        if main_lanes_before_exit:
            last_main_lane = main_lanes_before_exit[-1]  # 驶离前的车道

            # 查找驶离点：从主路车道到匝道0的变道点
            exit_point = -1
            for i in range(len(lanes) - 1, -1, -1):  # 从后往前查找
                if lanes[i] != 0 and lanes[min(i + 1, len(lanes) - 1)] == 0:
                    exit_point = i
                    break

            if last_main_lane == 1:
                # 从1车道驶离
                if len(main_lanes_before_exit) == 1:  # 直接从1到0
                    ramp_categories['从1驶向匝道0'].append(vehicle_id)
                else:  # 经历过变道过程
                    if len(main_lanes_before_exit) >= 2:
                        prev_lane = main_lanes_before_exit[-2]
                        if prev_lane == 2:
                            ramp_categories['匝道0从2变道至1后驶离'].append(vehicle_id)
                        elif prev_lane == 3:
                            ramp_categories['匝道0从3变道至1后驶离'].append(vehicle_id)
                        else:
                            ramp_categories['匝道0从1驶离'].append(vehicle_id)
                    else:
                        ramp_categories['匝道0从1驶离'].append(vehicle_id)

            elif last_main_lane == 2:
                # 从2车道驶离
                if len(main_lanes_before_exit) == 1:  # 直接从2到0
                    ramp_categories['匝道0从2驶离'].append(vehicle_id)
                else:  # 经历过变道过程
                    if len(main_lanes_before_exit) >= 2:
                        prev_lane = main_lanes_before_exit[-2]
                        if prev_lane == 3:
                            ramp_categories['匝道0从3变道至2后驶离'].append(vehicle_id)
                        else:
                            ramp_categories['匝道0从2驶离'].append(vehicle_id)
                    else:
                        ramp_categories['匝道0从2驶离'].append(vehicle_id)

            elif last_main_lane == 3:
                # 从3车道驶离
                if len(main_lanes_before_exit) == 1:  # 直接从3到0
                    ramp_categories['匝道0从3驶离'].append(vehicle_id)
                else:  # 从3驶离（可能经历变道）
                    ramp_categories['匝道0从3驶离'].append(vehicle_id)

# 车道变换行为分析（非匝道车辆）
maneuver_counts = defaultdict(int)
maneuver_vehicles = defaultdict(list)

for vehicle_id, lanes in lane_changes.items():
    if not lanes or len(lanes) < 2:
        continue

    # 如果是驶入0车道的车辆，跳过匝道驶离车辆
    if lanes[-1] == 0:
        continue

    # 只考虑在1、2、3车道行驶的车辆
    if not any(lane in [1, 2, 3] for lane in lanes):
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
        for i in range(1, len(unique_lane_sequence)):
            if unique_lane_sequence[i] > unique_lane_sequence[i - 1]:
                changes.append('R')  # 向右变道（数字增大）
            elif unique_lane_sequence[i] < unique_lane_sequence[i - 1]:
                changes.append('L')  # 向左变道（数字减小）

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
    for i, vehicle_id in enumerate(sorted(unclassified_vehicles)[:20]):  # 只显示前20个
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
        elif lanes[-1] == 0 and len(set(lanes)) == 1:
            reason = "一直在匝道0，未从主线驶入"
        elif not any(lane in [0, 1, 2, 3] for lane in lanes):
            reason = f"车道不在预期范围内(包含{set(lanes) - {{0, 1, 2, 3}} })"
        elif all(lane in [1, 2, 3] for lane in lanes) and len(unique_lane_sequence) < 2:
            reason = "主线车辆但车道无变化"
        else:
            reason = "其他原因"

        print(f"  - 原因: {reason}")
        if (i + 1) % 10 == 0:
            print()  # 每10个换行

print(f"\n=== 匝道驶离车辆细分 ===")
for category, vehicles in ramp_categories.items():
    if vehicles:  # 只打印有车辆的类别
        print(f"\n{category}: {len(vehicles)} 辆车")
        for i, vid in enumerate(vehicles[:20]):  # 只显示前20个
            if i % 10 == 0 and i != 0:
                print()  # 每10个换行
            print(f"{vid}", end=" ")
        if len(vehicles) > 20:
            print(f"... (显示前20个)")
        print()  # 换行

print(f"\n\n=== 车道变换行为统计 ===")
for maneuver_type, count in maneuver_counts.items():
    print(f"{maneuver_type}: {count} 辆车")
    vehicles_list = maneuver_vehicles[maneuver_type]
    for i, vid in enumerate(vehicles_list[:20]):  # 只显示前20个
        if i % 10 == 0 and i != 0:
            print()  # 每10个换行
        print(f"{vid}", end=" ")
    if len(vehicles_list) > 20:
        print(f"... (显示前20个)")
    print()

total_ramp_vehicles = sum(len(vehicles) for vehicles in ramp_categories.values())
print(f"\n匝道驶离车辆总数: {total_ramp_vehicles}")
print(f"非匝道车辆总数: {sum(maneuver_counts.values())}")
print(f"总计已分类车辆: {len(classified_vehicles)}")

# 计算变道前帧数
print("\n" + "=" * 60)
print("车辆变道前帧数统计分析")
print("=" * 60)

# 存储变道前帧数信息
change_frame_counts = defaultdict(list)
ramp_change_frame_counts = defaultdict(list)

# 存储被过滤的车辆信息
filtered_vehicles = []

# 计算普通变道车辆的变道前帧数
for maneuver_type in ['左变道', '右变道', '左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
    if maneuver_type in maneuver_vehicles:
        vehicles = maneuver_vehicles[maneuver_type]
        print(f"\n{maneuver_type} 原共有{len(vehicles)} 辆车，过滤处理后:")

        # 存储当前类型的所有变道前帧数
        current_frame_counts = []

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
                        filtered_vehicles.append({
                            'type': '普通变道',
                            'subtype': maneuver_type,
                            'vehicle_id': vehicle_id,
                            'frame_count': frame_count,
                            'threshold': THRESHOLD_LEFT_RIGHT_CHANGE
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
                            filtered_vehicles.append({
                                'type': '二次变道',
                                'subtype': f"{maneuver_type}_第1次变道",
                                'vehicle_id': vehicle_id,
                                'frame_count': first_change_frame,
                                'threshold': THRESHOLD_SECONDARY_FIRST_CHANGE
                            })
                        if not second_valid:
                            filtered_vehicles.append({
                                'type': '二次变道',
                                'subtype': f"{maneuver_type}_第2次变道",
                                'vehicle_id': vehicle_id,
                                'frame_count': second_change_frame,
                                'threshold': THRESHOLD_SECONDARY_SECOND_CHANGE
                            })
                elif len(change_indices) == 1:
                    # 只有一次变道
                    frame_count = change_indices[0]
                    # 应用阈值过滤
                    if frame_count >= THRESHOLD_LEFT_RIGHT_CHANGE:
                        change_frame_counts[maneuver_type].append(frame_count)
                        current_frame_counts.append((vehicle_id, [frame_count]))
                    else:
                        filtered_vehicles.append({
                            'type': '普通变道',
                            'subtype': maneuver_type,
                            'vehicle_id': vehicle_id,
                            'frame_count': frame_count,
                            'threshold': THRESHOLD_LEFT_RIGHT_CHANGE
                        })

        # 按4个一组打印
        for i in range(0, len(current_frame_counts), 4):
            batch = current_frame_counts[i:i + 4]
            line_parts = []
            for item in batch:
                vehicle_id, frame_counts = item
                if len(frame_counts) == 1:
                    line_parts.append(f"车辆 {vehicle_id}: {frame_counts[0]} 帧")
                elif len(frame_counts) == 2:
                    line_parts.append(f"车辆 {vehicle_id}(1次: {frame_counts[0]}, 2次: {frame_counts[1]})")
            print("  " + ", ".join(line_parts))

# 计算匝道驶离车辆的变道前帧数
for category, vehicles in ramp_categories.items():
    if vehicles:
        print(f"\n{category} ({len(vehicles)} 辆车):")
        current_ramp_frame_counts = []

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
            lanes = vehicle_data['LaneID'].tolist()

            # 找到驶离匝道的变道点（从主路到0）
            exit_change_index = -1
            for i in range(len(lanes) - 1):
                if lanes[i] != 0 and lanes[i + 1] == 0:  # 从非0车道变为0车道
                    exit_change_index = i
                    break

            if exit_change_index != -1:
                # 计算驶入匝道前的帧数
                frames_before_exit = exit_change_index + 1  # +1 包含变道点那一帧
                if frames_before_exit >= THRESHOLD_RAMP_FIRST_CHANGE:
                    ramp_change_frame_counts[f"{category}_驶离前"].append(frames_before_exit)
                    current_ramp_frame_counts.append((vehicle_id, [frames_before_exit]))
                else:
                    filtered_vehicles.append({
                        'type': '匝道驶离',
                        'subtype': f"{category}_驶离前",
                        'vehicle_id': vehicle_id,
                        'frame_count': frames_before_exit,
                        'threshold': THRESHOLD_RAMP_FIRST_CHANGE
                    })
            else:
                # 如果没找到驶离变道点（理论上不应该发生），记录异常
                print(f"  警告：车辆 {vehicle_id} 在 {category} 中未找到驶离变道点")

        # 按4个一组打印
        for i in range(0, len(current_ramp_frame_counts), 4):
            batch = current_ramp_frame_counts[i:i + 4]
            line_parts = []
            for item in batch:
                vehicle_id, frame_counts = item
                if len(frame_counts) == 1:
                    line_parts.append(f"车辆 {vehicle_id}: {frame_counts[0]} 帧")
                elif len(frame_counts) == 2:
                    line_parts.append(f"车辆 {vehicle_id}(1次: {frame_counts[0]}, 2次: {frame_counts[1]})")
                elif len(frame_counts) == 3:
                    line_parts.append(
                        f"车辆 {vehicle_id}(1次: {frame_counts[0]}, 2次: {frame_counts[1]}, 3次: {frame_counts[2]})")
            print("  " + ", ".join(line_parts))

# 开始数据截取 - 优化版本
sampling_data = []

# 存储过滤后的数据以便后续使用
filtered_maneuver_vehicles = {}
filtered_ramp_vehicles = {}

# 按阈值过滤普通变道车辆
for maneuver_type in ['左变道', '右变道', '左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
    if maneuver_type in maneuver_vehicles:
        original_vehicles = maneuver_vehicles[maneuver_type]
        valid_vehicles = []

        for vehicle_id in original_vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
            lanes = vehicle_data['LaneID'].tolist()

            # 找到所有变道点
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)

            if maneuver_type in ['左变道', '右变道']:
                # 单次变道，检查第一次变道前的帧数
                if change_indices and change_indices[0] >= THRESHOLD_LEFT_RIGHT_CHANGE:
                    valid_vehicles.append(vehicle_id)
            else:
                # 二次变道，检查两次变道的帧数
                if len(change_indices) >= 2:
                    first_change_frame = change_indices[0]
                    second_change_frame = change_indices[1] - change_indices[0]

                    if (first_change_frame >= THRESHOLD_SECONDARY_FIRST_CHANGE and
                            second_change_frame >= THRESHOLD_SECONDARY_SECOND_CHANGE):
                        valid_vehicles.append(vehicle_id)
                elif len(change_indices) == 1:
                    # 只有一次变道，按普通变道标准检查
                    if change_indices[0] >= THRESHOLD_LEFT_RIGHT_CHANGE:
                        valid_vehicles.append(vehicle_id)

        filtered_maneuver_vehicles[maneuver_type] = valid_vehicles
        print(f"{maneuver_type} 过滤后剩余: {len(valid_vehicles)} 辆")

# 按阈值过滤匝道驶离车辆
for category, vehicles in ramp_categories.items():
    valid_vehicles = []

    for vehicle_id in vehicles:
        vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
        lanes = vehicle_data['LaneID'].tolist()

        # 找到驶离匝道的变道点（从主路到0）
        exit_change_index = -1
        for i in range(len(lanes) - 1):
            if lanes[i] != 0 and lanes[i + 1] == 0:  # 从非0车道变为0车道
                exit_change_index = i
                break

        if exit_change_index != -1:
            # 检查驶入匝道前的帧数
            frames_before_exit = exit_change_index + 1  # +1 包含变道点那一帧
            if frames_before_exit >= THRESHOLD_RAMP_FIRST_CHANGE:
                valid_vehicles.append(vehicle_id)
        else:
            print(f"  警告：车辆 {vehicle_id} 在 {category} 中未找到驶离变道点")

    filtered_ramp_vehicles[category] = valid_vehicles
    print(f"{category} 过滤后剩余: {len(valid_vehicles)} 辆")

# 特别处理：将从车道2变道至车道1后再驶离的车辆，其第一次变道（2->1）加入右变道车辆
ramp_2_to_1_then_exit = filtered_ramp_vehicles.get('匝道0从2变道至1后驶离', [])
print(f"\n特殊处理：从车道2变道至车道1后再驶离的车辆数量: {len(ramp_2_to_1_then_exit)}")

# 添加这些车辆的第一次变道（2->1）到右变道车辆列表
if '右变道' not in filtered_maneuver_vehicles:
    filtered_maneuver_vehicles['右变道'] = []

for vehicle_id in ramp_2_to_1_then_exit:
    # 将这些车辆加入右变道车辆列表
    if vehicle_id not in filtered_maneuver_vehicles['右变道']:
        filtered_maneuver_vehicles['右变道'].append(vehicle_id)

print(f"添加后右变道车辆总数: {len(filtered_maneuver_vehicles['右变道'])}")

# 截取跟驰车辆数据
print(f"\n正在处理跟驰车辆...")
if '跟驰' in filtered_maneuver_vehicles:
    following_vehicles = filtered_maneuver_vehicles['跟驰']
elif '跟驰' in maneuver_vehicles:
    following_vehicles = maneuver_vehicles['跟驰']
else:
    following_vehicles = []

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

    sampling_data.append(sampled_data)

# 截取普通变道车辆数据 - 优化算法
print(f"\n正在处理普通变道车辆...")


def extract_lane_change_data(vehicle_id, maneuver_type, required_frames=100):
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
    if maneuver_type == '左变道':
        sampled_data['Label'] = '左变道'
    else:
        sampled_data['Label'] = '右变道'

    return sampled_data


# 处理左变道车辆
left_change_vehicles = filtered_maneuver_vehicles.get('左变道', [])
print(f"左变道车辆: 过滤后共有{len(left_change_vehicles)} 辆")
processed_left_count = 0

for vehicle_id in left_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '左变道', required_frames=100)
    if result is not None:
        sampling_data.append(result)
        processed_left_count += 1

print(f"成功处理左变道车辆: {processed_left_count} 辆")

# 处理右变道车辆
right_change_vehicles = filtered_maneuver_vehicles.get('右变道', [])
print(f"右变道车辆: 过滤后共有{len(right_change_vehicles)} 辆")
processed_right_count = 0

for vehicle_id in right_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '右变道', required_frames=100)
    if result is not None:
        sampling_data.append(result)
        processed_right_count += 1

print(f"成功处理右变道车辆: {processed_right_count} 辆")

# 截取二次变道车辆数据（只处理第一次变道）
print(f"\n正在处理二次变道车辆...")
for maneuver_type in ['左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
    if maneuver_type in filtered_maneuver_vehicles:
        vehicles = filtered_maneuver_vehicles[maneuver_type]
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
            else:
                sampled_data['Label'] = '右变道'

            sampling_data.append(sampled_data)

# 截取匝道驶离车辆数据
print(f"\n正在处理匝道驶离车辆...")
for category, vehicles in filtered_ramp_vehicles.items():
    if vehicles:
        print(f"{category}: {len(vehicles)} 辆车")

        for vehicle_id in vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

            # 确保数据按时间排序
            vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

            # 找到驶离匝道的变道点（从主路到0）
            lanes = vehicle_data['LaneID'].tolist()
            exit_change_index = -1
            for i in range(len(lanes) - 1):
                if lanes[i] != 0 and lanes[i + 1] == 0:  # 从非0车道变为0车道
                    exit_change_index = i
                    break

            if exit_change_index != -1:
                # 提取驶入匝道前4-2秒的数据（从exit_change_index-100到exit_change_index-50，共50帧）
                start_idx = max(0, exit_change_index - 100)
                end_idx = min(len(vehicle_data), exit_change_index - 50)

                # 确保提取的帧数为50帧
                if end_idx - start_idx >= 50:
                    sampled_data = vehicle_data.iloc[start_idx:start_idx + 50].copy()
                else:
                    # 如果不够50帧，提取可用的最大数据
                    available_frames = end_idx - start_idx
                    if available_frames > 0:
                        if available_frames >= 50:
                            sampled_data = vehicle_data.iloc[start_idx:start_idx + 50].copy()
                        else:
                            # 如果不足50帧，取全部可用数据
                            sampled_data = vehicle_data.iloc[start_idx:end_idx].copy()
                    else:
                        print(f"  车辆 {vehicle_id} 无法提取有效匝道驶离数据，跳过")
                        continue

                # 根据驶离前的车道确定标签
                if lanes[exit_change_index] in [1, 2]:  # 从1或2驶离
                    sampled_data['Label'] = '右驶离'  # 向右驶离
                elif lanes[exit_change_index] == 3:  # 从3驶离
                    sampled_data['Label'] = '右驶离'  # 向右驶离
                else:
                    sampled_data['Label'] = '驶离'  # 通用驶离标签

                sampling_data.append(sampled_data)
            else:
                print(f"  警告：车辆 {vehicle_id} 在 {category} 中未找到驶离变道点，跳过")

# 合并所有采样数据
print(f"\n合并采样数据...")
if sampling_data:
    combined_sampling_df = pd.concat(sampling_data, ignore_index=True)

    print(f"合并后的采样数据形状: {combined_sampling_df.shape}")

    # 删除指定列 - 更新为新的列名
    columns_to_remove = ['ID', 'time', 'Frame', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking',
                         'Dist_to_left_marking', 'Dist_to_right_marking']
    final_sampling_df = combined_sampling_df.drop(
        columns=[col for col in columns_to_remove if col in combined_sampling_df.columns])

    print(f"删除指定列后的数据形状: {final_sampling_df.shape}")

    # 保存数据
    csv_path = os.path.join(save_dir, "traffic_flows_sampling.csv")
    pkl_path = os.path.join(save_dir, "traffic_flows_sampling.pkl")

    final_sampling_df.to_csv(csv_path, index=False)
    final_sampling_df.to_pickle(pkl_path)

    print(f"数据已保存到:")
    print(f"  CSV: {csv_path}")
    print(f"  PKL: {pkl_path}")

    # 显示标签分布
    print(f"\n标签分布:")
    if 'Label' in final_sampling_df.columns:
        label_counts = final_sampling_df['Label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count} 行")

else:
    print("没有采样数据可合并")

print(f"\n数据截取和处理完成！")

# 额外报告：左变道车辆处理情况
print(f"\n" + "=" * 50)
print("左变道车辆处理情况详细报告")
print("=" * 50)
original_left_count = len(filtered_maneuver_vehicles.get('左变道', []))
final_left_count = processed_left_count
print(f"过滤后左变道车辆数量: {original_left_count}")
print(f"最终截取成功的左变道车辆数量: {final_left_count}")
if original_left_count > 0:
    print(f"成功率: {final_left_count / original_left_count * 100:.2f}%")

if original_left_count > final_left_count:
    print(f"丢失了 {original_left_count - final_left_count} 辆左变道车辆数据")
    all_filtered_left = set(filtered_maneuver_vehicles.get('左变道', []))
    successfully_processed = set(
        [data['ID'].iloc[0] for data in sampling_data if '左变道' in str(data.get('Label', ''))])
    lost_vehicles = all_filtered_left - successfully_processed
    if len(lost_vehicles) < 20:
        print(f"丢失的车辆ID: {sorted(list(lost_vehicles))}")
    else:
        print(f"丢失的车辆ID (前20个): {sorted(list(lost_vehicles))[:20]}")

# 额外报告：特殊处理车辆情况
print(f"\n" + "=" * 50)
print("特殊处理车辆情况报告")
print("=" * 50)
ramp_2_to_1_then_exit = ramp_categories.get('匝道0从2变道至1后驶离', [])
print(f"从车道2变道至车道1后再驶离的车辆数量: {len(ramp_2_to_1_then_exit)}")
print(f"这些车辆的第一次变道（2->1）已被计入右变道车辆")
print(f"原始右变道车辆数量: {len(maneuver_vehicles.get('右变道', []))}")
print(f"过滤后右变道车辆数量: {len(filtered_maneuver_vehicles.get('右变道', []))}")
