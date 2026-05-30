import pandas as pd
import numpy as np
import gc
import os

# 设置数据路径
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
west_pkl_processed_path = os.path.join(save_dir, "traffic_flows_complete.pkl")  # 待补全数据库
traffic_flows_path = os.path.join(save_dir, "traffic_flows_complete1.csv")  # 补全数据汇集，将所有生成的补全数据写入该文件中
traffic_pkl_flows_path = os.path.join(save_dir, "traffic_flows_complete1.pkl")  # 补全数据汇集，将所有生成的补全数据写入该文件中

# 读取数据
traffic_flows_west_processed = pd.read_pickle(west_pkl_processed_path)  # 读取待补全数据库

print(f"原始数据量：{len(traffic_flows_west_processed)} 行")
print(f"数据列数：{len(traffic_flows_west_processed.columns)} 列")
print(f"数据列名：{list(traffic_flows_west_processed.columns)}")

# 按车辆ID和帧率排序
print("\n按车辆ID和帧率排序...")
traffic_flows_west_processed = traffic_flows_west_processed.sort_values(['ID', 'Frame']).reset_index(drop=True)

print(f"排序后数据量：{len(traffic_flows_west_processed)} 行")

# 查找缺失帧并进行补全
print("\n查找缺失帧...")
all_vehicle_ids = traffic_flows_west_processed['ID'].unique()
print(f"共有 {len(all_vehicle_ids)} 个车辆ID")

# 识别缺失帧
total_missing_frames = 0
missing_frame_info = []

for vehicle_id in all_vehicle_ids:
    vehicle_data = traffic_flows_west_processed[traffic_flows_west_processed['ID'] == vehicle_id].copy()
    vehicle_data = vehicle_data.sort_values('Frame').reset_index(drop=True)

    for i in range(1, len(vehicle_data)):
        prev_frame = vehicle_data.iloc[i - 1]['Frame']
        curr_frame = vehicle_data.iloc[i]['Frame']

        frame_gap = int(curr_frame - prev_frame)
        if frame_gap > 1:
            missing_frame_info.append({
                'vehicle_id': vehicle_id,
                'prev_frame': int(prev_frame),
                'curr_frame': int(curr_frame),
                'missing_count': frame_gap - 1
            })
            total_missing_frames += frame_gap - 1

print(f"共发现 {len(missing_frame_info)} 处缺失，总计 {total_missing_frames} 个缺失帧")

# 执行补全
print("\n开始补全缺失帧...")
filled_data = []

for info in missing_frame_info:
    vehicle_id = info['vehicle_id']
    prev_frame = info['prev_frame']
    curr_frame = info['curr_frame']
    missing_count = info['missing_count']

    # 获取前后帧数据
    prev_data = traffic_flows_west_processed[
        (traffic_flows_west_processed['ID'] == vehicle_id) &
        (traffic_flows_west_processed['Frame'] == prev_frame)
        ].iloc[0]

    curr_data = traffic_flows_west_processed[
        (traffic_flows_west_processed['ID'] == vehicle_id) &
        (traffic_flows_west_processed['Frame'] == curr_frame)
        ].iloc[0]

    # 计算时间差
    time_diff = curr_data['time'] - prev_data['time']
    frame_diff = curr_frame - prev_frame

    # 生成缺失帧
    for k in range(1, missing_count + 1):
        missing_frame = prev_frame + k
        missing_time = prev_data['time'] + k * (time_diff / frame_diff)

        # 创建新行
        new_row = prev_data.copy()
        new_row['Frame'] = missing_frame
        new_row['time'] = round(missing_time, 2)

        # 连续数据进行线性插值
        new_row['X'] = round(prev_data['X'] + k * (curr_data['X'] - prev_data['X']) / frame_diff, 2)
        new_row['Y'] = round(prev_data['Y'] + k * (curr_data['Y'] - prev_data['Y']) / frame_diff, 2)
        new_row['Velocity'] = round(
            prev_data['Velocity'] + k * (curr_data['Velocity'] - prev_data['Velocity']) / frame_diff, 2)
        new_row['Acceleration'] = round(
            prev_data['Acceleration'] + k * (curr_data['Acceleration'] - prev_data['Acceleration']) / frame_diff, 2)
        new_row['long_Vel'] = round(
            prev_data['long_Vel'] + k * (curr_data['long_Vel'] - prev_data['long_Vel']) / frame_diff, 2)
        new_row['lat_Vel'] = round(
            prev_data['lat_Vel'] + k * (curr_data['lat_Vel'] - prev_data['lat_Vel']) / frame_diff, 2)
        new_row['long_Acc'] = round(
            prev_data['long_Acc'] + k * (curr_data['long_Acc'] - prev_data['long_Acc']) / frame_diff, 2)
        new_row['lat_Acc'] = round(
            prev_data['lat_Acc'] + k * (curr_data['lat_Acc'] - prev_data['lat_Acc']) / frame_diff, 2)

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

        filled_data.append(new_row)

# 将补全的数据添加到原数据中
if filled_data:
    filled_df = pd.DataFrame(filled_data)
    traffic_flows_west_processed = pd.concat([traffic_flows_west_processed, filled_df], ignore_index=True)
    traffic_flows_west_processed = traffic_flows_west_processed.sort_values(['ID', 'Frame']).reset_index(drop=True)
    print(f"补全完成，新增 {len(filled_df)} 行数据")
else:
    print("没有缺失帧需要补全")

print(f"补全后总数据量：{len(traffic_flows_west_processed)} 行")

# 重新计算TTC < 2的数据
print("\n重新计算TTC < 2的数据...")
low_ttc_data = traffic_flows_west_processed[traffic_flows_west_processed['TTC'] < 2].copy()
print(f"发现 {len(low_ttc_data)} 行TTC < 2的数据")

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
            front_data = traffic_flows_west_processed[
                (traffic_flows_west_processed['ID'] == front_id) &
                (traffic_flows_west_processed['Frame'] == current_frame)
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
                traffic_flows_west_processed.loc[idx, 'Following_dist'] = following_dist
                traffic_flows_west_processed.loc[idx, 'Time_Headway'] = time_headway
                traffic_flows_west_processed.loc[idx, 'TTC'] = ttc
            else:
                # 如果找不到前车数据，设置TTC为0
                traffic_flows_west_processed.loc[idx, 'Following_dist'] = 0.0
                traffic_flows_west_processed.loc[idx, 'Time_Headway'] = 0.0
                traffic_flows_west_processed.loc[idx, 'TTC'] = 0.0
        else:
            # 无前车，设置TTC为0
            traffic_flows_west_processed.loc[idx, 'Following_dist'] = 0.0
            traffic_flows_west_processed.loc[idx, 'Time_Headway'] = 0.0
            traffic_flows_west_processed.loc[idx, 'TTC'] = 0.0

# 再次检查TTC < 2且TTC != 0的数据
print("\n重新计算后，检查TTC < 2且TTC != 0的数据...")
final_low_ttc_data = traffic_flows_west_processed[
    (traffic_flows_west_processed['TTC'] < 2) & (traffic_flows_west_processed['TTC'] != 0)].copy()
print(f"重新计算后仍有 {len(final_low_ttc_data)} 行TTC < 2且TTC != 0的数据")

if len(final_low_ttc_data) > 0:
    print("\nTTC < 2且TTC != 0的详细信息：")
    for idx, row in final_low_ttc_data.iterrows():
        current_id = int(row['ID'])
        front_id = int(row['FrontID'])
        following_dist = row['Following_dist']
        current_velocity = row['Velocity']
        ttc = row['TTC']

        # 获取前车速度
        front_data = traffic_flows_west_processed[
            (traffic_flows_west_processed['ID'] == front_id) &
            (traffic_flows_west_processed['Frame'] == row['Frame'])
            ]

        if len(front_data) > 0:
            front_velocity = front_data.iloc[0]['Velocity']
        else:
            front_velocity = 0.0

        print(
            f" 帧：{int(row['Frame'])} 当前车ID: {current_id}, 前车ID: {front_id}, 跟车距离: {following_dist}, 当前车速: {current_velocity}, 前车速度: {front_velocity}, TTC: {ttc}")

# 展示部分TTC修改情况
print("\n展示部分TTC修改情况：")
sample_low_ttc_data = traffic_flows_west_processed[
    (traffic_flows_west_processed['TTC'] < 2) & (traffic_flows_west_processed['TTC'] != 0)].head(10)
if len(sample_low_ttc_data) > 0:
    print("前10行TTC < 2且TTC != 0的数据：")
    for idx, row in sample_low_ttc_data.iterrows():
        current_id = int(row['ID'])
        front_id = int(row['FrontID'])
        following_dist = row['Following_dist']
        current_velocity = row['Velocity']

        # 获取前车速度
        front_data = traffic_flows_west_processed[
            (traffic_flows_west_processed['ID'] == front_id) &
            (traffic_flows_west_processed['Frame'] == row['Frame'])
            ]

        if len(front_data) > 0:
            front_velocity = front_data.iloc[0]['Velocity']
        else:
            front_velocity = 0.0

        ttc = row['TTC']
        print(
            f"    帧: {int(row['Frame'])}, 当前车ID: {current_id}, 前车ID: {front_id}, 跟车距离: {following_dist}, 当前车速: {current_velocity}, 前车速度: {front_velocity}, TTC: {ttc}")
else:
    print("没有TTC < 2且TTC != 0的数据")

# 连续性验证
print("\n开始进行连续性验证...")
verification_results = []
total_remaining_gaps = 0  # 累积缺失帧数
total_remaining_missing_frames = 0  # 累积缺失帧率数

for vehicle_id in all_vehicle_ids:
    # 筛选特定车辆ID的数据
    vehicle_data = traffic_flows_west_processed[traffic_flows_west_processed['ID'] == vehicle_id].copy()

    # 按时间排序（确保时间序列连续）
    vehicle_data = vehicle_data.sort_values('Frame').reset_index(drop=True)

    # 计算相邻帧的差值
    frame_diffs = []
    gap_details = []
    for i in range(1, len(vehicle_data)):
        current_frame = int(vehicle_data.iloc[i]['Frame'])
        prev_frame = int(vehicle_data.iloc[i - 1]['Frame'])
        frame_diff = current_frame - prev_frame

        if frame_diff > 1:  # 有缺失帧
            missing_count = frame_diff - 1
            gap_details.append((prev_frame, current_frame, missing_count))
            total_remaining_missing_frames += missing_count  # 累积缺失帧数

        frame_diffs.append(frame_diff)

    # 转换为Series以便筛选
    frame_diffs_series = pd.Series(frame_diffs)
    gaps = frame_diffs_series[frame_diffs_series > 1]
    total_remaining_gaps += len(gaps)  # 累积缺失间隔数

    if len(gaps) == 0:
        print(f"  车辆ID {vehicle_id}：连续性验证通过")
        verification_results.append((vehicle_id, "通过", 0, []))
    else:
        print(f"  车辆ID {vehicle_id}：仍有 {len(gaps)} 处帧率缺失")
        verification_results.append((vehicle_id, "失败", len(gaps), gap_details))

# 汇总连续性验证结果
print("\n连续性验证结果汇总：")
pass_count = sum(1 for result in verification_results if result[1] == "通过")
fail_count = sum(1 for result in verification_results if result[1] == "失败")
print(f"  通过验证的车辆数：{pass_count}")
print(f"  验证失败的车辆数：{fail_count}")

if fail_count > 0:
    print("  验证失败的车辆ID及详细缺失信息：")
    for vid, status, gap_count, gap_details in verification_results:
        if status == "失败":
            print(f"    车辆ID {int(vid)}: {gap_count} 处帧率缺失")
            for prev_frame, curr_frame, missing_count in gap_details:
                print(f"      帧 {prev_frame} 到 帧 {curr_frame} 之间缺失 {missing_count} 帧")

# 打印累积统计信息
print(f"\n累积统计信息：")
print(f"  总共仍有 {total_remaining_gaps} 处帧率缺失")
print(f"  总共仍有 {total_remaining_missing_frames} 帧缺失")

# 保留2位小数
numeric_columns = ['time', 'X', 'Y', 'Length', 'Width', 'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc',
                   'lat_Acc', 'Dist_to_right_edge_marking', 'Dist_to_left_marking', 'Dist_to_right_marking',
                   'Following_dist', 'Time_Headway', 'TTC']
for col in numeric_columns:
    if col in traffic_flows_west_processed.columns:
        traffic_flows_west_processed[col] = traffic_flows_west_processed[col].round(2)

# 保存补全后的数据到CSV文件
print(f"\n正在保存补全后的数据到 {traffic_flows_path}...")
traffic_flows_west_processed.to_csv(traffic_flows_path, index=False, encoding='utf-8-sig')
traffic_flows_west_processed.to_pickle(traffic_pkl_flows_path)

print(f"数据已保存到：{traffic_flows_path} 和 {traffic_pkl_flows_path}")

print("\n数据补全、TTC重新计算和连续性验证完成！")
print(f"最终数据量：{len(traffic_flows_west_processed)} 行")

# 清理内存
del traffic_flows_west_processed, filled_data, filled_df, low_ttc_data, final_low_ttc_data, sample_low_ttc_data
gc.collect()

print("内存已清理完成")



