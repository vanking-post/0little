import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# 阈值参数控制
THRESHOLD_LEFT_RIGHT_CHANGE = 100  # 普通变道（左/右变道）：建议至少100帧（4秒）
THRESHOLD_SECONDARY_FIRST_CHANGE = 100  # 二次变道的第一次变道：建议至少100帧（4秒）
THRESHOLD_SECONDARY_SECOND_CHANGE = 50  # 二次变道的第二次变道：建议至少50帧（2秒）
THRESHOLD_RAMP_FIRST_CHANGE = 100  # 匝道驶离第1次变道：建议至少100帧（4秒）
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50  # 匝道驶离后续变道：建议至少50帧（2秒）

# 数据路径
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
files = [
    ("guiyi", "traffic_flows_guiyi.pkl")]  # 补全数据库

# 读取补全数据库
file_path = os.path.join(save_dir, files[0][1])  # traffic_flows_guiyi.pkl
df = pd.read_pickle(file_path)

# 删除指定列
columns_to_drop = ['Class', 'Length','Width']
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

# 记录每辆车的车道变化
lane_changes = {}
for vehicle_id in all_vehicle_ids:
    vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
    lane_sequence = vehicle_data['LaneID'].tolist()
    lane_changes[vehicle_id] = lane_sequence

print("=== 车辆分类开始 ===")

# 匝道驶离车辆细分
ramp_categories = {
    '从1驶向匝道0': [],
    '匝道0从2驶离': [],
    '匝道0从3驶离': [],
    '匝道0从2变道至1后驶离': [],
    '匝道0从3变道至2后驶离': [],
    '匝道0从3变道至1后驶离': []
}

# 车道变换行为分析（非匝道车辆）
maneuver_counts = defaultdict(int)
maneuver_vehicles = defaultdict(list)

for vehicle_id, lanes in lane_changes.items():
    if not lanes or len(lanes) < 2:
        continue

    # 如果是驶入0车道的车辆，进行匝道驶离分类
    if lanes[-1] == 0:
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
    else:
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

print(f"\n\n=== 车道变换行为统计 ===")
for maneuver_type, count in maneuver_counts.items():
    print(f"{maneuver_type}: {count} 辆车")

total_ramp_vehicles = sum(len(vehicles) for vehicles in ramp_categories.values())
print(f"\n匝道驶离车辆总数: {total_ramp_vehicles}")
print(f"非匝道车辆总数: {sum(maneuver_counts.values())}")
print(f"总计已分类车辆: {len(classified_vehicles)}")

print("\n=== 车辆分类后统一过滤 ===")

# 创建字典来记录每种变道类型的数据来源
source_tracking = {
    '左变道': {'普通变道': [], '二次变道': [], '驶离主路车辆': []},
    '右变道': {'普通变道': [], '二次变道': [], '驶离主路车辆': []}
}

# 处理普通变道车辆
filtered_maneuver_vehicles = {}
for maneuver_type in ['左变道', '右变道']:
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

            if change_indices and change_indices[0] >= THRESHOLD_LEFT_RIGHT_CHANGE:
                valid_vehicles.append(vehicle_id)
                source_tracking[maneuver_type]['普通变道'].append(vehicle_id)

        filtered_maneuver_vehicles[maneuver_type] = valid_vehicles

        # 输出过滤信息
        print(
            f"{maneuver_type}（普通变道）过滤前: {len(original_vehicles)} 辆, 过滤后: {len(valid_vehicles)} 辆, 减少了: {len(original_vehicles) - len(valid_vehicles)} 辆")

# 处理二次变道车辆（只考虑第一次变道）
for maneuver_type in ['左变道后左变道', '左变道后右变道', '右变道后左变道', '右变道后右变道']:
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

            if len(change_indices) >= 2:
                first_change_idx = change_indices[0]

                # 检查第一次变道前是否有足够的数据
                if first_change_idx >= THRESHOLD_SECONDARY_FIRST_CHANGE:
                    valid_vehicles.append(vehicle_id)

                    # 根据第一次变道方向添加到对应来源
                    if maneuver_type.startswith('左变道'):
                        source_tracking['左变道']['二次变道'].append(vehicle_id)
                    else:
                        source_tracking['右变道']['二次变道'].append(vehicle_id)

        filtered_maneuver_vehicles[maneuver_type] = valid_vehicles

        # 输出过滤信息
        print(
            f"{maneuver_type}过滤前: {len(original_vehicles)} 辆, 过滤后: {len(valid_vehicles)} 辆, 减少了: {len(original_vehicles) - len(valid_vehicles)} 辆")

# ======================== 关键修改部分开始 ========================
# 处理驶离主路车辆（只考虑第一次右变道，排除直接从1驶入0的车辆）
# 定义需要处理的匝道分类（排除"从1驶向匝道0"）
target_ramp_categories = ['匝道0从2变道至1后驶离', '匝道0从3变道至2后驶离', '匝道0从3变道至1后驶离']
for category in target_ramp_categories:
    if category in ramp_categories:
        original_vehicles = ramp_categories[category]
        valid_vehicles = []

        for vehicle_id in original_vehicles:
            vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id]
            lanes = vehicle_data['LaneID'].tolist()

            # 找到所有变道点
            change_indices = []
            for i in range(1, len(lanes)):
                if lanes[i] != lanes[i - 1]:
                    change_indices.append(i)

            # 仅处理有至少一次变道、且第一次变道满足阈值的车辆
            if change_indices and change_indices[0] >= THRESHOLD_RAMP_FIRST_CHANGE:
                # 提取第一次变道的方向（仅关注右变道）
                first_change_idx = change_indices[0]
                first_lane_before = lanes[first_change_idx - 1]
                first_lane_after = lanes[first_change_idx]

                # 仅保留第一次是右变道的车辆（车道号减小，如2→1、3→2、3→1）
                if first_lane_after < first_lane_before:
                    valid_vehicles.append(vehicle_id)
                    # 归类到右变道的"驶离主路车辆"来源
                    source_tracking['右变道']['驶离主路车辆'].append(vehicle_id)

        # 将符合条件的驶离主路车辆合并到右变道中
        if '右变道' in filtered_maneuver_vehicles:
            filtered_maneuver_vehicles['右变道'].extend(valid_vehicles)
        else:
            filtered_maneuver_vehicles['右变道'] = valid_vehicles

        # 输出过滤信息
        print(
            f"{category}过滤前: {len(original_vehicles)} 辆, 过滤后: {len(valid_vehicles)} 辆, 减少了: {len(original_vehicles) - len(valid_vehicles)} 辆")
# ======================== 关键修改部分结束 ========================

# 开始数据截取
sampling_data = []

print(f"\n=== 样本截取开始 ===")


def extract_lane_change_data(vehicle_id, maneuver_type, required_frames=100):
    """提取变道车辆数据的核心函数"""
    vehicle_data = df_sorted[df_sorted['ID'] == vehicle_id].copy()

    # 确保数据按时间排序
    vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

    # 检查数据帧数是否足够
    total_frames = len(vehicle_data)
    if total_frames < required_frames:
        return None

    # 找到变道发生的位置
    lanes = vehicle_data['LaneID'].tolist()
    change_indices = []
    for i in range(1, len(lanes)):
        if lanes[i] != lanes[i - 1]:
            change_indices.append(i)

    if not change_indices:
        return None

    # 取第一个变道点
    first_change_idx = change_indices[0]

    # 检查变道前是否有足够的数据
    frames_before_change = first_change_idx
    if frames_before_change < required_frames:
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
            return None

    # 添加标签列
    if maneuver_type == '左变道':
        sampled_data['Label'] = '左变道'
    else:
        sampled_data['Label'] = '右变道'

    return sampled_data


# 处理左变道车辆
print(f"\n正在处理左变道车辆...")
left_change_vehicles = []
left_sources = {}

for source_name, vehicles in source_tracking['左变道'].items():
    if vehicles:
        left_change_vehicles.extend(vehicles)
        left_sources[source_name] = vehicles

original_left_count = len(left_change_vehicles)
print(f"左变道车辆来源统计:")
for source, vehicles in source_tracking['左变道'].items():
    print(f"  {source}: {len(vehicles)} 辆")

processed_left_count = 0
for vehicle_id in left_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '左变道', required_frames=100)
    if result is not None:
        sampling_data.append(result)
        processed_left_count += 1

print(f"成功处理左变道车辆: {processed_left_count} 辆")
if processed_left_count != original_left_count:
    print(
        f"左变道车辆样本截取过滤: 原始 {original_left_count} 辆, 成功处理 {processed_left_count} 辆, 未能处理 {original_left_count - processed_left_count} 辆")

# 处理右变道车辆（包括驶离主路的车辆）
print(f"\n正在处理右变道车辆（包括驶离主路车辆）...")
right_change_vehicles = []
right_sources = {}

for source_name, vehicles in source_tracking['右变道'].items():
    if vehicles:
        right_change_vehicles.extend(vehicles)
        right_sources[source_name] = vehicles

original_right_count = len(right_change_vehicles)
print(f"右变道车辆来源统计（包括驶离主路车辆）:")
for source, vehicles in source_tracking['右变道'].items():
    print(f"  {source}: {len(vehicles)} 辆")

processed_right_count = 0
for vehicle_id in right_change_vehicles:
    result = extract_lane_change_data(vehicle_id, '右变道', required_frames=100)
    if result is not None:
        sampling_data.append(result)
        processed_right_count += 1

print(f"成功处理右变道车辆: {processed_right_count} 辆")
if processed_right_count != original_right_count:
    print(
        f"右变道车辆样本截取过滤: 原始 {original_right_count} 辆, 成功处理 {processed_right_count} 辆, 未能处理 {original_right_count - processed_right_count} 辆")

# 处理跟驰车辆
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

# 合并所有采样数据
print(f"\n合并采样数据...")
if sampling_data:
    combined_sampling_df = pd.concat(sampling_data, ignore_index=True)

    print(f"合并后的采样数据形状: {combined_sampling_df.shape}")

    # 删除指定列 - 更新为新的列名
    columns_to_remove = [ 'time', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking',
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

    # 显示变道车辆来源统计
    print(f"\n变道车辆来源统计:")
    print("左变道来源:")
    for source, vehicles in source_tracking['左变道'].items():
        print(f"  {source}: {len(vehicles)} 辆")

    print("右变道来源（包括驶离主路车辆）:")
    for source, vehicles in source_tracking['右变道'].items():
        print(f"  {source}: {len(vehicles)} 辆")

else:
    print("没有采样数据可合并")

print(f"\n数据截取和处理完成！")