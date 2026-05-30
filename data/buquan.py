import pandas as pd
import numpy as np
import gc
import os

# ===============================
# 1. 设置数据路径
# ===============================
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
west_pkl_processed_path = os.path.join(save_dir, "traffic_flows_west_processed.pkl")  # 待补全数据库
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")  # 源数据库
traffic_flows_path = os.path.join(save_dir, "traffic_flows_complete0.csv")  # 补全数据汇集
traffic_pkl_flows_path = os.path.join(save_dir, "traffic_flows_complete0.pkl")  # 补全数据汇集

print(f"{'=' * 50}")
print(f"数据处理开始: 交通流数据补全与TTC重新计算（合并版 - 优化补全逻辑）")
print(f"数据路径: {save_dir}")
print(f"待补全数据库路径: {west_pkl_processed_path}")
print(f"源数据库路径: {west_pkl_path}")
print(f"输出CSV路径: {traffic_flows_path}")
print(f"输出Pickle路径: {traffic_pkl_flows_path}")
print(f"{'=' * 50}\n")

# ===============================
# 2. 读取数据
# ===============================
print("正在读取源数据库和待补全数据库...")
traffic_flows_west = pd.read_pickle(west_pkl_path)  # 源数据库
traffic_flows_west_processed = pd.read_pickle(west_pkl_processed_path)  # 待补全数据库

print(f"源数据库数据量: {len(traffic_flows_west):,} 行")
print(f"待补全数据库数据量: {len(traffic_flows_west_processed):,} 行")
print(f"源数据库列数: {len(traffic_flows_west.columns)} 列")
print(f"待补全数据库列数: {len(traffic_flows_west_processed.columns)} 列")

# 获取所有车辆ID
all_vehicle_ids = traffic_flows_west_processed['ID'].unique()
print(f"需要处理的车辆ID数量: {len(all_vehicle_ids):,} 个")
print(f"待处理的ID列表: {list(all_vehicle_ids)}\n")


# ===============================
# 3. 按ID分组处理数据（参考源数据库进行补全 - buquan0逻辑）
# ===============================
def fill_missing_frames():
    """参考源数据库进行数据补全"""
    # 用于存储补全的数据
    filled_data = []

    # 按ID处理每个需要补全的车辆
    for vehicle_id in all_vehicle_ids:
        print(f"\n正在处理车辆ID: {vehicle_id}")

        # 获取当前车辆的原始数据
        vehicle_data = traffic_flows_west_processed[traffic_flows_west_processed['ID'] == vehicle_id].copy()
        vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

        # 查找缺失的时间点
        time_gaps = []
        for i in range(len(vehicle_data) - 1):
            current_time = vehicle_data.iloc[i]['time']
            next_time = vehicle_data.iloc[i + 1]['time']

            # 检查时间间隔是否大于0.04秒（25帧/秒）
            if next_time - current_time > 0.04:
                gap_start = current_time
                gap_end = next_time
                missing_frames = int((gap_end - gap_start) / 0.04) - 1
                time_gaps.append((gap_start, gap_end, missing_frames))

        print(f"  检测到 {len(time_gaps)} 个时间间隔，总共需要补全 {sum([gap[2] for gap in time_gaps]):,} 帧")

        # 对每个时间间隔进行补全
        for gap_start, gap_end, missing_frames in time_gaps:
            # 获取前一帧和后一帧数据
            prev_frame_data = vehicle_data[vehicle_data['time'] == gap_start].iloc[0]
            next_frame_data = vehicle_data[vehicle_data['time'] == gap_end].iloc[0]

            # 计算合理性指标
            velocity_change_ratio = abs(next_frame_data['Velocity'] - prev_frame_data['Velocity']) / prev_frame_data[
                'Velocity'] if prev_frame_data['Velocity'] != 0 else 0
            displacement = np.sqrt(
                (next_frame_data['X'] - prev_frame_data['X']) ** 2 + (next_frame_data['Y'] - prev_frame_data['Y']) ** 2)
            displacement_ratio = displacement / (prev_frame_data['Velocity'] * (gap_end - gap_start)) if \
                prev_frame_data['Velocity'] != 0 else 0

            is_high_reliability = (velocity_change_ratio < 0.3 and displacement_ratio < 0.3)

            # 生成缺失帧的数据
            for k in range(1, missing_frames + 1):
                missing_time = gap_start + k * 0.04
                missing_frame = int(prev_frame_data['Frame'] + k)

                # 计算插值比例
                ratio = (missing_time - gap_start) / (gap_end - gap_start)

                # 构建补全行数据
                filled_row = {}

                # 应查询数据库补全的字段
                # 从源数据库查询这些字段的值
                source_row = traffic_flows_west[(traffic_flows_west['ID'] == vehicle_id) &
                                                (abs(traffic_flows_west['time'] - missing_time) < 0.001)]

                if len(source_row) > 0:
                    source_row = source_row.iloc[0]
                    # 查询数据库字段
                    filled_row['LaneID'] = int(source_row['LaneID'])
                    filled_row['Dist_to_right_edge_marking'] = round(float(source_row['Dist_to_right_edge_marking']), 2)
                    filled_row['Dist_to_left_marking'] = round(float(source_row['Dist_to_left_marking']), 2)
                    filled_row['Dist_to_right_marking'] = round(float(source_row['Dist_to_right_marking']), 2)
                    filled_row['LeftBehindID'] = int(source_row['LeftBehindID'])
                    filled_row['LeftSideID'] = int(source_row['LeftSideID'])
                    filled_row['LeftFrontID'] = int(source_row['LeftFrontID'])
                    filled_row['BehindID'] = int(source_row['BehindID'])
                    filled_row['EgoVehicleID'] = int(source_row['EgoVehicleID'])
                    filled_row['FrontID'] = int(source_row['FrontID'])
                    filled_row['RightBehindID'] = int(source_row['RightBehindID'])
                    filled_row['RightSideID'] = int(source_row['RightSideID'])
                    filled_row['RightFrontID'] = int(source_row['RightFrontID'])
                else:
                    # 如果源数据库中没有该时间点的数据，使用前一帧的值
                    filled_row['LaneID'] = int(prev_frame_data['LaneID'])
                    filled_row['Dist_to_right_edge_marking'] = round(
                        float(prev_frame_data['Dist_to_right_edge_marking']), 2)
                    filled_row['Dist_to_left_marking'] = round(float(prev_frame_data['Dist_to_left_marking']), 2)
                    filled_row['Dist_to_right_marking'] = round(float(prev_frame_data['Dist_to_right_marking']), 2)
                    filled_row['LeftBehindID'] = int(prev_frame_data['LeftBehindID'])
                    filled_row['LeftSideID'] = int(prev_frame_data['LeftSideID'])
                    filled_row['LeftFrontID'] = int(prev_frame_data['LeftFrontID'])
                    filled_row['BehindID'] = int(prev_frame_data['BehindID'])
                    filled_row['EgoVehicleID'] = int(prev_frame_data['EgoVehicleID'])
                    filled_row['FrontID'] = int(prev_frame_data['FrontID'])
                    filled_row['RightBehindID'] = int(prev_frame_data['RightBehindID'])
                    filled_row['RightSideID'] = int(prev_frame_data['RightSideID'])
                    filled_row['RightFrontID'] = int(prev_frame_data['RightFrontID'])

                # 应插值来补全的字段
                filled_row['Frame'] = missing_frame
                filled_row['time'] = round(missing_time, 2)  # 保留2位小数
                filled_row['ID'] = vehicle_id
                filled_row['Class'] = int(prev_frame_data['Class'])
                filled_row['X'] = round(prev_frame_data['X'] + ratio * (next_frame_data['X'] - prev_frame_data['X']), 2)
                filled_row['Y'] = round(prev_frame_data['Y'] + ratio * (next_frame_data['Y'] - prev_frame_data['Y']), 2)
                filled_row['Length'] = round(float(prev_frame_data['Length']), 2)
                filled_row['Width'] = round(float(prev_frame_data['Width']), 2)

                # 根据合理性选择插值方式
                if is_high_reliability:
                    # 高可靠性：线性插值
                    filled_row['Velocity'] = round(prev_frame_data['Velocity'] + ratio * (
                            next_frame_data['Velocity'] - prev_frame_data['Velocity']), 2)
                    filled_row['Acceleration'] = round(prev_frame_data['Acceleration'] + ratio * (
                            next_frame_data['Acceleration'] - prev_frame_data['Acceleration']), 2)
                    filled_row['long_Vel'] = round(prev_frame_data['long_Vel'] + ratio * (
                            next_frame_data['long_Vel'] - prev_frame_data['long_Vel']), 2)
                    filled_row['lat_Vel'] = round(prev_frame_data['lat_Vel'] + ratio * (
                            next_frame_data['lat_Vel'] - prev_frame_data['lat_Vel']), 2)
                    filled_row['long_Acc'] = round(prev_frame_data['long_Acc'] + ratio * (
                            next_frame_data['long_Acc'] - prev_frame_data['long_Acc']), 2)
                    filled_row['lat_Acc'] = round(prev_frame_data['lat_Acc'] + ratio * (
                            next_frame_data['lat_Acc'] - prev_frame_data['lat_Acc']), 2)
                else:
                    # 低可靠性：前后帧平均
                    filled_row['Velocity'] = round((prev_frame_data['Velocity'] + next_frame_data['Velocity']) / 2, 2)
                    filled_row['Acceleration'] = round(
                        (prev_frame_data['Acceleration'] + next_frame_data['Acceleration']) / 2, 2)
                    filled_row['long_Vel'] = round((prev_frame_data['long_Vel'] + next_frame_data['long_Vel']) / 2, 2)
                    filled_row['lat_Vel'] = round((prev_frame_data['lat_Vel'] + next_frame_data['lat_Vel']) / 2, 2)
                    filled_row['long_Acc'] = round((prev_frame_data['long_Acc'] + next_frame_data['long_Acc']) / 2, 2)
                    filled_row['lat_Acc'] = round((prev_frame_data['lat_Acc'] + next_frame_data['lat_Acc']) / 2, 2)

                # 计算依赖字段（Following_dist, Time_Headway, TTC）
                # 获取前车在缺失时间点的位置（从源数据库查询）
                front_id = filled_row['FrontID']
                if front_id != 0:
                    # 从前车ID查询源数据库（在缺失时间点）
                    front_source = traffic_flows_west[(traffic_flows_west['ID'] == front_id) &
                                                      (abs(traffic_flows_west['time'] - missing_time) < 0.001)]
                    if len(front_source) > 0:
                        front_source = front_source.iloc[0]
                        front_x = front_source['X']
                        front_velocity = front_source['Velocity']

                        # 计算跟车距离
                        following_dist = round(front_x - filled_row['X'], 2)

                        # 计算车头时距
                        time_headway = round(
                            following_dist / filled_row['Velocity'] if filled_row['Velocity'] > 0 else 0, 2)

                        # 计算TTC（碰撞时间）
                        velocity_diff = filled_row['Velocity'] - front_velocity  # 修正：当前车速减前车车速
                        if velocity_diff > 0 and following_dist > 0:  # 当前车速度大于前车速度，且距离为正
                            ttc = round(following_dist / velocity_diff, 2)
                        else:
                            ttc = 0.0  # 无碰撞风险
                    else:
                        # 如果源数据库中没有前车数据，设置为0
                        following_dist = 0.0
                        time_headway = 0.0
                        ttc = 0.0
                else:
                    # 无前车
                    following_dist = 0.0
                    time_headway = 0.0
                    ttc = 0.0

                filled_row['Following_dist'] = following_dist
                filled_row['Time_Headway'] = time_headway
                filled_row['TTC'] = round(ttc, 2)

                # 添加到补全数据列表
                filled_data.append(filled_row)

        print(f"  车辆ID {vehicle_id} 补全完成，补全了 {sum([gap[2] for gap in time_gaps]):,} 帧")

    return pd.DataFrame(filled_data)


# 执行buquan0的补全
print("\n开始执行buquan0补全逻辑（参考源数据库）...")
filled_df_0 = fill_missing_frames()
print(f"buquan0补全完成！补全了 {len(filled_df_0):,} 行数据")

# 合并原始数据和buquan0补全数据
print("\n合并原始数据和buquan0补全数据...")
traffic_flows_west_complete = pd.concat([traffic_flows_west_processed, filled_df_0], ignore_index=True)
traffic_flows_west_complete = traffic_flows_west_complete.sort_values(['ID', 'Frame']).reset_index(drop=True)

print(f"合并后总数据量: {len(traffic_flows_west_complete):,} 行")
print(f"补全前数据量: {len(traffic_flows_west_processed):,} 行")
print(f"buquan0补全数据量: {len(filled_df_0):,} 行\n")

# ===============================
# 4. 基于前后帧进行额外补全（buquan1逻辑）
# ===============================
print("开始执行buquan1补全逻辑（基于前后帧插值）...")
all_vehicle_ids_complete = traffic_flows_west_complete['ID'].unique()

# 识别缺失帧
total_missing_frames_1 = 0
missing_frame_info_1 = []

for vehicle_id in all_vehicle_ids_complete:
    vehicle_data = traffic_flows_west_complete[traffic_flows_west_complete['ID'] == vehicle_id].copy()
    vehicle_data = vehicle_data.sort_values('Frame').reset_index(drop=True)

    for i in range(1, len(vehicle_data)):
        prev_frame = vehicle_data.iloc[i - 1]['Frame']
        curr_frame = vehicle_data.iloc[i]['Frame']

        frame_gap = int(curr_frame - prev_frame)
        if frame_gap > 1:
            missing_frame_info_1.append({
                'vehicle_id': vehicle_id,
                'prev_frame': int(prev_frame),
                'curr_frame': int(curr_frame),
                'missing_count': frame_gap - 1
            })
            total_missing_frames_1 += frame_gap - 1

print(f"buquan1检测到 {len(missing_frame_info_1):,} 处缺失，总计 {total_missing_frames_1:,} 个缺失帧")

# 执行buquan1补全
filled_data_1 = []

for info in missing_frame_info_1:
    vehicle_id = info['vehicle_id']
    prev_frame = info['prev_frame']
    curr_frame = info['curr_frame']
    missing_count = info['missing_count']

    # 获取前后帧数据
    prev_data = traffic_flows_west_complete[
        (traffic_flows_west_complete['ID'] == vehicle_id) &
        (traffic_flows_west_complete['Frame'] == prev_frame)
        ].iloc[0]

    curr_data = traffic_flows_west_complete[
        (traffic_flows_west_complete['ID'] == vehicle_id) &
        (traffic_flows_west_complete['Frame'] == curr_frame)
        ].iloc[0]

    # 计算时间差和帧差
    time_diff = curr_data['time'] - prev_data['time']
    frame_diff = curr_frame - prev_frame

    # 计算合理性指标
    velocity_change_ratio = abs(curr_data['Velocity'] - prev_data['Velocity']) / prev_data['Velocity'] if prev_data[
                                                                                                              'Velocity'] != 0 else 0
    displacement = np.sqrt((curr_data['X'] - prev_data['X']) ** 2 + (curr_data['Y'] - prev_data['Y']) ** 2)
    displacement_ratio = displacement / (prev_data['Velocity'] * time_diff) if prev_data['Velocity'] != 0 else 0
    is_high_reliability = (velocity_change_ratio < 0.3 and displacement_ratio < 0.3)

    # 生成缺失帧
    for k in range(1, missing_count + 1):
        missing_frame = prev_frame + k
        missing_time = prev_data['time'] + k * (time_diff / frame_diff)

        # 创建新行
        new_row = prev_data.copy()
        new_row['Frame'] = missing_frame
        new_row['time'] = round(missing_time, 2)

        # 插值比例
        ratio = k / frame_diff

        # 位置插值
        new_row['X'] = round(prev_data['X'] + ratio * (curr_data['X'] - prev_data['X']), 2)
        new_row['Y'] = round(prev_data['Y'] + ratio * (curr_data['Y'] - prev_data['Y']), 2)

        # 速度、加速度等连续变量
        if is_high_reliability:
            # 高可靠性：线性插值
            new_row['Velocity'] = round(prev_data['Velocity'] + ratio * (curr_data['Velocity'] - prev_data['Velocity']),
                                        2)
            new_row['Acceleration'] = round(
                prev_data['Acceleration'] + ratio * (curr_data['Acceleration'] - prev_data['Acceleration']), 2)
            new_row['long_Vel'] = round(prev_data['long_Vel'] + ratio * (curr_data['long_Vel'] - prev_data['long_Vel']),
                                        2)
            new_row['lat_Vel'] = round(prev_data['lat_Vel'] + ratio * (curr_data['lat_Vel'] - prev_data['lat_Vel']), 2)
            new_row['long_Acc'] = round(prev_data['long_Acc'] + ratio * (curr_data['long_Acc'] - prev_data['long_Acc']),
                                        2)
            new_row['lat_Acc'] = round(prev_data['lat_Acc'] + ratio * (curr_data['lat_Acc'] - prev_data['lat_Acc']), 2)
        else:
            # 低可靠性：前后帧平均
            new_row['Velocity'] = round((prev_data['Velocity'] + curr_data['Velocity']) / 2, 2)
            new_row['Acceleration'] = round((prev_data['Acceleration'] + curr_data['Acceleration']) / 2, 2)
            new_row['long_Vel'] = round((prev_data['long_Vel'] + curr_data['long_Vel']) / 2, 2)
            new_row['lat_Vel'] = round((prev_data['lat_Vel'] + curr_data['lat_Vel']) / 2, 2)
            new_row['long_Acc'] = round((prev_data['long_Acc'] + curr_data['long_Acc']) / 2, 2)
            new_row['lat_Acc'] = round((prev_data['lat_Acc'] + curr_data['lat_Acc']) / 2, 2)

        # 离散数据保持前帧数据
        new_row['LaneID'] = int(prev_data['LaneID'])
        new_row['Dist_to_right_edge_marking'] = round(prev_data['Dist_to_right_edge_marking'], 2)
        new_row['Dist_to_left_marking'] = round(prev_data['Dist_to_left_marking'], 2)
        new_row['Dist_to_right_marking'] = round(prev_data['Dist_to_right_marking'], 2)
        new_row['LeftBehindID'] = int(prev_data['LeftBehindID'])
        new_row['LeftSideID'] = int(prev_data['LeftSideID'])
        new_row['LeftFrontID'] = int(prev_data['LeftFrontID'])
        new_row['BehindID'] = int(prev_data['BehindID'])
        new_row['EgoVehicleID'] = int(prev_data['EgoVehicleID'])
        new_row['FrontID'] = int(prev_data['FrontID'])
        new_row['RightBehindID'] = int(prev_data['RightBehindID'])
        new_row['RightSideID'] = int(prev_data['RightSideID'])
        new_row['RightFrontID'] = int(prev_data['RightFrontID'])

        # 初始设置TTC为0，后续会重新计算
        new_row['Following_dist'] = 0.0
        new_row['Time_Headway'] = 0.0
        new_row['TTC'] = 0.0

        filled_data_1.append(new_row)

# 将buquan1补全的数据添加到原数据中
if filled_data_1:
    filled_df_1 = pd.DataFrame(filled_data_1)
    traffic_flows_west_complete = pd.concat([traffic_flows_west_complete, filled_df_1], ignore_index=True)
    traffic_flows_west_complete = traffic_flows_west_complete.sort_values(['ID', 'Frame']).reset_index(drop=True)
    print(f"buquan1补全完成，新增 {len(filled_df_1):,} 行数据")
else:
    print("buquan1没有缺失帧需要补全")

print(f"buquan1补全后总数据量: {len(traffic_flows_west_complete):,} 行\n")

# ===============================
# 5. 重新计算TTC < 2的数据
# ===============================
print("重新计算TTC < 2的数据...")
low_ttc_data = traffic_flows_west_complete[traffic_flows_west_complete['TTC'] < 2].copy()
print(f"发现 {len(low_ttc_data):,} 行TTC < 2的数据")

if len(low_ttc_data) > 0:
    print("正在重新计算TTC...")

    # 为每个TTC < 2的数据重新计算
    for idx, row in low_ttc_data.iterrows():
        current_id = int(row['ID'])
        current_frame = int(row['Frame'])
        current_time = row['time']
        current_x = row['X']
        current_velocity = row['Velocity']
        front_id = int(row['FrontID'])

        if front_id != 0:  # 如果有前车
            # 在同一时间点查找前车数据
            front_data = traffic_flows_west_complete[
                (traffic_flows_west_complete['ID'] == front_id) &
                (traffic_flows_west_complete['Frame'] == current_frame)
                ]

            if len(front_data) > 0:
                front_data = front_data.iloc[0]
                front_x = front_data['X']
                front_velocity = front_data['Velocity']

                # 计算跟车距离
                following_dist = round(front_x - current_x, 2)

                # 计算车头时距
                time_headway = round(following_dist / current_velocity if current_velocity > 0 else 0, 2)

                # 计算TTC（碰撞时间）
                velocity_diff = current_velocity - front_velocity  # 修正：当前车速减前车车速
                if velocity_diff > 0 and following_dist > 0:  # 当前车速度大于前车速度，且距离为正
                    ttc = round(following_dist / velocity_diff, 2)
                else:
                    ttc = 0.0  # 无碰撞风险

                # 更新原数据
                traffic_flows_west_complete.loc[idx, 'Following_dist'] = following_dist
                traffic_flows_west_complete.loc[idx, 'Time_Headway'] = time_headway
                traffic_flows_west_complete.loc[idx, 'TTC'] = ttc
            else:
                # 如果找不到前车数据，设置TTC为0
                traffic_flows_west_complete.loc[idx, 'Following_dist'] = 0.0
                traffic_flows_west_complete.loc[idx, 'Time_Headway'] = 0.0
                traffic_flows_west_complete.loc[idx, 'TTC'] = 0.0
        else:
            # 无前车，设置TTC为0
            traffic_flows_west_complete.loc[idx, 'Following_dist'] = 0.0
            traffic_flows_west_complete.loc[idx, 'Time_Headway'] = 0.0
            traffic_flows_west_complete.loc[idx, 'TTC'] = 0.0

# 再次检查TTC < 2且TTC != 0的数据
print("\n重新计算后，检查TTC < 2且TTC != 0的数据...")
final_low_ttc_data = traffic_flows_west_complete[
    (traffic_flows_west_complete['TTC'] < 2) & (traffic_flows_west_complete['TTC'] != 0)].copy()
print(f"重新计算后仍有 {len(final_low_ttc_data):,} 行TTC < 2且TTC != 0的数据")

if len(final_low_ttc_data) > 0:
    print("\nTTC < 2且TTC != 0的详细信息：")
    for idx, row in final_low_ttc_data.iterrows():
        current_id = int(row['ID'])
        front_id = int(row['FrontID'])
        following_dist = row['Following_dist']
        current_velocity = row['Velocity']
        ttc = row['TTC']

        # 获取前车速度
        front_data = traffic_flows_west_complete[
            (traffic_flows_west_complete['ID'] == front_id) &
            (traffic_flows_west_complete['Frame'] == row['Frame'])
            ]

        if len(front_data) > 0:
            front_velocity = front_data.iloc[0]['Velocity']
        else:
            front_velocity = 0.0

        print(
            f"  车辆ID: {current_id}, 前车ID: {front_id}, 跟车距离: {following_dist:.2f}, "
            f"当前车速: {current_velocity:.2f}, 前车速度: {front_velocity:.2f}, TTC: {ttc:.2f}")
else:
    print("没有TTC < 2且TTC != 0的数据")

# 展示部分TTC修改情况
print("\n展示部分TTC修改情况：")
sample_low_ttc_data = traffic_flows_west_complete[
    (traffic_flows_west_complete['TTC'] < 2) & (traffic_flows_west_complete['TTC'] != 0)].head(10)
if len(sample_low_ttc_data) > 0:
    print("前10行TTC < 2且TTC != 0的数据：")
    for idx, row in sample_low_ttc_data.iterrows():
        current_id = int(row['ID'])
        front_id = int(row['FrontID'])
        following_dist = row['Following_dist']
        current_velocity = row['Velocity']

        # 获取前车速度
        front_data = traffic_flows_west_complete[
            (traffic_flows_west_complete['ID'] == front_id) &
            (traffic_flows_west_complete['Frame'] == row['Frame'])
            ]

        if len(front_data) > 0:
            front_velocity = front_data.iloc[0]['Velocity']
        else:
            front_velocity = 0.0

        ttc = row['TTC']
        print(
            f"  帧: {int(row['Frame'])}, 车辆ID: {current_id}, 前车ID: {front_id}, "
            f"跟车距离: {following_dist:.2f}, 当前车速: {current_velocity:.2f}, "
            f"前车速度: {front_velocity:.2f}, TTC: {ttc:.2f}")
else:
    print("没有TTC < 2且TTC != 0的数据")

# ===============================
# 6. 连续性验证
# ===============================
print("\n开始进行连续性验证...")
verification_results = []
total_remaining_gaps = 0  # 累积缺失帧数
total_remaining_missing_frames = 0  # 累积缺失帧率数

for vehicle_id in all_vehicle_ids_complete:
    # 筛选特定车辆ID的数据
    vehicle_data = traffic_flows_west_complete[traffic_flows_west_complete['ID'] == vehicle_id].copy()

    # 按时间排序（确保时间序列连续）
    vehicle_data = vehicle_data.sort_values('time').reset_index(drop=True)

    # 计算相邻时间点的差值
    time_diffs = []
    gap_details = []
    for i in range(1, len(vehicle_data)):
        current_frame = int(vehicle_data.iloc[i]['Frame'])
        prev_frame = int(vehicle_data.iloc[i - 1]['Frame'])
        frame_diff = current_frame - prev_frame

        if frame_diff > 1:  # 有缺失帧
            missing_count = frame_diff - 1
            gap_details.append((prev_frame, current_frame, missing_count))
            total_remaining_missing_frames += missing_count  # 累积缺失帧数

        time_diffs.append(frame_diff)

    # 转换为Series以便筛选
    time_diffs_series = pd.Series(time_diffs)
    gaps = time_diffs_series[time_diffs_series > 1.0]
    total_remaining_gaps += len(gaps)  # 累积缺失间隔数

    if len(gaps) == 0:
        print(f"  车辆ID {vehicle_id}: 数据连续性验证通过")
        verification_results.append((vehicle_id, "通过", 0, []))
    else:
        print(f"  车辆ID {vehicle_id}: 仍有 {len(gaps)} 处帧率缺失")
        verification_results.append((vehicle_id, "失败", len(gaps), gap_details))

# 汇总验证结果
print("\n验证结果汇总:")
pass_count = sum(1 for result in verification_results if result[1] == "通过")
fail_count = sum(1 for result in verification_results if result[1] == "失败")
print(f"  通过验证的车辆数: {pass_count:,}")
print(f"  验证失败的车辆数: {fail_count:,}")

if fail_count > 0:
    print("\n验证失败的车辆ID及详细缺失信息:")
    for vid, status, gap_count, gap_details in verification_results:
        if status == "失败":
            print(f"    车辆ID {int(vid)}: {gap_count} 处帧率缺失")
            for prev_frame, curr_frame, missing_count in gap_details:
                print(f"      帧 {prev_frame} 到 帧 {curr_frame} 之间缺失 {missing_count} 帧")

# 打印累积统计信息
print(f"\n累积统计信息:")
print(f"  总共仍有 {total_remaining_gaps:,} 处帧率缺失")
print(f"  总共仍有 {total_remaining_missing_frames:,} 帧缺失")

# ===============================
# 7. 保留2位小数并保存数据
# ===============================
print("\n保留2位小数并保存数据...")
numeric_columns = ['time', 'X', 'Y', 'Length', 'Width', 'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc',
                   'lat_Acc', 'Dist_to_right_edge_marking', 'Dist_to_left_marking', 'Dist_to_right_marking',
                   'Following_dist', 'Time_Headway', 'TTC']
for col in numeric_columns:
    if col in traffic_flows_west_complete.columns:
        traffic_flows_west_complete[col] = traffic_flows_west_complete[col].round(2)

# 保存补全后的数据
print(f"\n正在保存补全后的数据到 {traffic_flows_path}...")
traffic_flows_west_complete.to_csv(traffic_flows_path, index=False, encoding='utf-8-sig')
traffic_flows_west_complete.to_pickle(traffic_pkl_flows_path)
print(f"数据已保存到: {traffic_flows_path} ")

# 最终信息
print(f"\n{'=' * 50}")
print(f"数据补全、TTC重新计算和连续性验证完成！")
print(f"最终数据量: {len(traffic_flows_west_complete):,} 行")
print(f"buquan0补全数据量: {len(filled_df_0):,} 行")
print(f"buquan1补全数据量: {len(filled_df_1):,} 行")
print(f"总补全数据量: {len(filled_df_0) + len(filled_df_1):,} 行")
print(f"{'=' * 50}\n")

# 清理内存
del traffic_flows_west, traffic_flows_west_processed, all_vehicle_ids, all_vehicle_ids_complete, filled_df_0, filled_df_1, traffic_flows_west_complete
gc.collect()
print("内存已清理完成")