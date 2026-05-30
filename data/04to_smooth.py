#实现样本的平滑与归一。平滑旨在将一些飘忽的数据合理平移到适当位置
#归一旨在将离散的、不同量级的数据调整为标准正态分布，实现数据的等权重分析
#读取traffic_flows_sample，输出traffic_flows_smooth和traffic_flow_guiyi
import pandas as pd
import numpy as np
import gc
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取文件路径
# save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
# traffic_pkl_flows_sample_path = os.path.join(save_dir, "traffic_flows_sample.pkl")  # 待平滑数据汇集
# traffic_pkl_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.pkl")  # 待平滑数据汇集
# traffic_csv_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.csv")  # 待平滑数据汇集
# traffic_csv_flows_guiyi = os.path.join(save_dir, "traffic_flows_guiyi.csv")  # 待归一化数据集
# traffic_pkl_flows_guiyi = os.path.join(save_dir, "traffic_flows_guiyi.pkl")  # 待归一化数据集

# 读取文件路径东向
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
traffic_pkl_flows_sample_path = os.path.join(save_dir, "traffic_flows_sample.pkl")  # 待平滑数据汇集
traffic_pkl_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.pkl")  # 待平滑数据汇集
traffic_csv_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.csv")  # 待平滑数据汇集
traffic_csv_flows_guiyi = os.path.join(save_dir, "traffic_flows_guiyi.csv")  # 待归一化数据集
traffic_pkl_flows_guiyi = os.path.join(save_dir, "traffic_flows_guiyi.pkl")  # 待归一化数据集

# SG滤波参数（按类别设置）
SG_WINDOW_XY = 9  # X/Y坐标平滑窗口
SG_POLYORDER_XY = 3  # X/Y坐标平滑阶数

SG_WINDOW_VEL = 5  # 速度相关变量平滑窗口
SG_POLYORDER_VEL = 3  # 速度相关变量平滑阶数

SG_WINDOW_ACC = 9  # 加速度相关变量平滑窗口
SG_POLYORDER_ACC = 3  # 加速度相关变量平滑阶数

SG_WINDOW_FOLLOW = 9  # 跟车距离平滑窗口
SG_POLYORDER_FOLLOW = 3  # 跟车距离平滑阶数

SG_WINDOW_HEADWAY = 9  # 车头时距平滑窗口
SG_POLYORDER_HEADWAY = 3  # 车头时距平滑阶数

SG_WINDOW_TTC = 9  # 碰撞时间平滑窗口
SG_POLYORDER_TTC = 3  # 碰撞时间平滑阶数

# 新增：距离参数平滑参数
SG_WINDOW_DIST = 9  # 距离参数平滑窗口
SG_POLYORDER_DIST = 3  # 距离参数平滑阶数

SG_WINDOW_OTHER = 9  # 其他变量平滑窗口
SG_POLYORDER_OTHER = 3  # 其他变量平滑阶数

size = 10

print(f"{'=' * 50}")
print(f"数据处理开始: 交通流数据SG滤波平滑处理与可视化对比（多变量随机车辆对比版-三图布局）")
print(f"X/Y坐标平滑参数: 窗口={SG_WINDOW_XY}, 阶数={SG_POLYORDER_XY}")
print(f"速度相关变量平滑参数: 窗口={SG_WINDOW_VEL}, 阶数={SG_POLYORDER_VEL}")
print(f"加速度相关变量平滑参数: 窗口={SG_WINDOW_ACC}, 阶数={SG_POLYORDER_ACC}")
print(f"跟车距离平滑参数: 窗口={SG_WINDOW_FOLLOW}, 阶数={SG_POLYORDER_FOLLOW}")
print(f"车头时距平滑参数: 窗口={SG_WINDOW_HEADWAY}, 阶数={SG_POLYORDER_HEADWAY}")
print(f"碰撞时间平滑参数: 窗口={SG_WINDOW_TTC}, 阶数={SG_POLYORDER_TTC}")
print(f"距离参数平滑参数: 窗口={SG_WINDOW_DIST}, 阶数={SG_POLYORDER_DIST}")
print(f"数据路径: {save_dir}")
print(f"输入数据路径: {traffic_pkl_flows_sample_path}")
print(f"输出Pickle路径: {traffic_pkl_flows_smooth_path}")
print(f"输出CSV路径: {traffic_csv_flows_smooth_path}")
print(f"{'=' * 50}\n")

# 读取数据
print("正在读取数据...")
df = pd.read_pickle(traffic_pkl_flows_sample_path)
print(f"数据读取完成，总数据量: {len(df):,} 行")

# 获取所有车辆ID
vehicle_ids = sorted(df['ID'].unique())
print(f"车辆ID数量: {len(vehicle_ids):,} 个")
print(f"车辆ID范围: {min(vehicle_ids)} - {max(vehicle_ids)}")

# 随机挑选10个车辆ID进行对比
print("随机挑选10个车辆进行对比分析...")
selected_vehicle_ids = np.random.choice(vehicle_ids, size=min(10, len(vehicle_ids)), replace=False)
print(f"随机选择的车辆ID: {selected_vehicle_ids}")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 对所有连续性数据应用SG滤波
print("正在应用SG滤波进行平滑处理...")

df_smooth = df.copy()

# 对X/Y坐标应用SG滤波
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_XY:
        if 'X' in vehicle_data.columns and len(vehicle_data['X']) > 0:
            smooth_x = savgol_filter(vehicle_data['X'].values,
                                     window_length=SG_WINDOW_XY,
                                     polyorder=SG_POLYORDER_XY)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'X'] = smooth_x

        if 'Y' in vehicle_data.columns and len(vehicle_data['Y']) > 0:
            smooth_y = savgol_filter(vehicle_data['Y'].values,
                                     window_length=SG_WINDOW_XY,
                                     polyorder=SG_POLYORDER_XY)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Y'] = smooth_y
    else:
        print(f"车辆ID {vehicle_id} X/Y坐标数据长度不足，跳过SG滤波处理")

# 对速度相关变量应用SG滤波
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_VEL:
        if 'Velocity' in vehicle_data.columns and len(vehicle_data['Velocity']) > 0:
            smooth_vel = savgol_filter(vehicle_data['Velocity'].values,
                                       window_length=SG_WINDOW_VEL,
                                       polyorder=SG_POLYORDER_VEL)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Velocity'] = smooth_vel

        if 'long_Vel' in vehicle_data.columns and len(vehicle_data['long_Vel']) > 0:
            smooth_long_vel = savgol_filter(vehicle_data['long_Vel'].values,
                                            window_length=SG_WINDOW_VEL,
                                            polyorder=SG_POLYORDER_VEL)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'long_Vel'] = smooth_long_vel

        if 'lat_Vel' in vehicle_data.columns and len(vehicle_data['lat_Vel']) > 0:
            smooth_lat_vel = savgol_filter(vehicle_data['lat_Vel'].values,
                                           window_length=SG_WINDOW_VEL,
                                           polyorder=SG_POLYORDER_VEL)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'lat_Vel'] = smooth_lat_vel
    else:
        print(f"车辆ID {vehicle_id} 速度相关数据长度不足，跳过SG滤波处理")

# 对加速度相关变量应用SG滤波
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_ACC:
        if 'Acceleration' in vehicle_data.columns and len(vehicle_data['Acceleration']) > 0:
            smooth_acc = savgol_filter(vehicle_data['Acceleration'].values,
                                       window_length=SG_WINDOW_ACC,
                                       polyorder=SG_POLYORDER_ACC)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Acceleration'] = smooth_acc

        if 'long_Acc' in vehicle_data.columns and len(vehicle_data['long_Acc']) > 0:
            smooth_long_acc = savgol_filter(vehicle_data['long_Acc'].values,
                                            window_length=SG_WINDOW_ACC,
                                            polyorder=SG_POLYORDER_ACC)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'long_Acc'] = smooth_long_acc

        if 'lat_Acc' in vehicle_data.columns and len(vehicle_data['lat_Acc']) > 0:
            smooth_lat_acc = savgol_filter(vehicle_data['lat_Acc'].values,
                                           window_length=SG_WINDOW_ACC,
                                           polyorder=SG_POLYORDER_ACC)
            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'lat_Acc'] = smooth_lat_acc
    else:
        print(f"车辆ID {vehicle_id} 加速度相关数据长度不足，跳过SG滤波处理")

# 对7个距离参数应用SG滤波（保持非负值）
distance_columns = ['LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RS_Dist', 'RF_Dist']

for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_DIST:
        for dist_col in distance_columns:
            if dist_col in vehicle_data.columns and len(vehicle_data[dist_col]) > 0:
                # 获取原始数据
                raw_dist = vehicle_data[dist_col].values.copy()

                # 处理0或负值（避免SG滤波产生负值）
                raw_dist_clean = raw_dist.copy()
                raw_dist_clean[raw_dist_clean <= 0] = 0.001  # 将0或负值替换为很小的正数

                # 在对数域进行SG滤波（防止负值）
                log_dist = np.log(raw_dist_clean)

                # 在对数域进行SG滤波
                smooth_log_dist = savgol_filter(log_dist,
                                                window_length=SG_WINDOW_DIST,
                                                polyorder=SG_POLYORDER_DIST)

                # 还原到原始域
                smooth_dist = np.exp(smooth_log_dist)

                # 确保非负
                smooth_dist[smooth_dist < 0] = 0.001

                df_smooth.loc[df_smooth['ID'] == vehicle_id, dist_col] = smooth_dist
    else:
        print(f"车辆ID {vehicle_id} 距离参数数据长度不足，跳过SG滤波处理")

# 对跟车距离应用对数域SG滤波（避免负值）
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_FOLLOW:
        if 'Following_dist' in vehicle_data.columns and len(vehicle_data['Following_dist']) > 0:
            # 获取原始数据
            raw_follow_dist = vehicle_data['Following_dist'].values.copy()

            # 处理0或负值（避免取对数时出错）
            raw_follow_dist_clean = raw_follow_dist.copy()
            raw_follow_dist_clean[raw_follow_dist_clean <= 0] = 0.001  # 将0或负值替换为很小的正数

            # 对数变换
            log_follow_dist = np.log(raw_follow_dist_clean)

            # 在对数域进行SG滤波
            smooth_log_follow = savgol_filter(log_follow_dist,
                                              window_length=SG_WINDOW_FOLLOW,
                                              polyorder=SG_POLYORDER_FOLLOW)

            # 还原到原始域
            smooth_follow_dist = np.exp(smooth_log_follow)

            # 确保非负
            smooth_follow_dist[smooth_follow_dist < 0] = 0.001

            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Following_dist'] = smooth_follow_dist
    else:
        print(f"车辆ID {vehicle_id} 跟车距离数据长度不足，跳过SG滤波处理")

# 对车头时距应用对数域SG滤波（避免负值）
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_HEADWAY:
        if 'Time_Headway' in vehicle_data.columns and len(vehicle_data['Time_Headway']) > 0:
            # 获取原始数据
            raw_headway = vehicle_data['Time_Headway'].values.copy()

            # 处理0或负值（避免取对数时出错）
            raw_headway_clean = raw_headway.copy()
            raw_headway_clean[raw_headway_clean <= 0] = 0.001  # 将0或负值替换为很小的正数

            # 对数变换
            log_headway = np.log(raw_headway_clean)

            # 在对数域进行SG滤波
            smooth_log_headway = savgol_filter(log_headway,
                                               window_length=SG_WINDOW_HEADWAY,
                                               polyorder=SG_POLYORDER_HEADWAY)

            # 还原到原始域
            smooth_headway = np.exp(smooth_log_headway)

            # 确保非负
            smooth_headway[smooth_headway < 0] = 0.001

            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Time_Headway'] = smooth_headway
    else:
        print(f"车辆ID {vehicle_id} 车头时距数据长度不足，跳过SG滤波处理")

# 对碰撞时间应用对数域SG滤波（避免负值）
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_TTC:
        if 'TTC' in vehicle_data.columns and len(vehicle_data['TTC']) > 0:
            # 获取原始数据
            raw_ttc = vehicle_data['TTC'].values.copy()

            # 处理0或负值（避免取对数时出错）
            raw_ttc_clean = raw_ttc.copy()
            raw_ttc_clean[raw_ttc_clean <= 0] = 0.001  # 将0或负值替换为很小的正数

            # 对数变换
            log_ttc = np.log(raw_ttc_clean)

            # 在对数域进行SG滤波
            smooth_log_ttc = savgol_filter(log_ttc,
                                           window_length=SG_WINDOW_TTC,
                                           polyorder=SG_POLYORDER_TTC)

            # 还原到原始域
            smooth_ttc = np.exp(smooth_log_ttc)

            # 确保非负
            smooth_ttc[smooth_ttc < 0] = 0.001

            df_smooth.loc[df_smooth['ID'] == vehicle_id, 'TTC'] = smooth_ttc
    else:
        print(f"车辆ID {vehicle_id} 碰撞时间数据长度不足，跳过SG滤波处理")

# 对其他变量应用SG滤波
other_columns = ['Dist_to_right_edge_marking', 'Dist_to_right_marking', 'Dist_to_left_marking']

for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_OTHER:
        for other_col in other_columns:
            if other_col in vehicle_data.columns and len(vehicle_data[other_col]) > 0:
                # 获取原始数据
                raw_other = vehicle_data[other_col].values.copy()

                # 对原始数据进行SG滤波，保留负值
                smooth_other = savgol_filter(raw_other,
                                             window_length=SG_WINDOW_OTHER,
                                             polyorder=SG_POLYORDER_OTHER)

                df_smooth.loc[df_smooth['ID'] == vehicle_id, other_col] = smooth_other
    else:
        print(f"车辆ID {vehicle_id} 其他变量数据长度不足，跳过SG滤波处理")
print("SG滤波平滑处理完成")

# 将连续数据保留2位有效数字
print("正在保留2位有效数字...")
continuous_cols = ['X', 'Y', 'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc', 'lat_Acc', 'Following_dist',
                   'Time_Headway', 'TTC', 'LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RS_Dist', 'RF_Dist',
                   'Dist_to_right_edge_marking', 'Dist_to_right_marking', 'Dist_to_left_marking']
for col in continuous_cols:
    if col in df_smooth.columns:
        df_smooth[col] = df_smooth[col].round(2)

print("数据保留2位有效数字处理完成")

# 创建三个大图，每个包含十个车辆的对比图

# 1. X/Y坐标对比图 (5行2列布局)
fig1, axes1 = plt.subplots(5, 2, figsize=(19.2, 10.8))
axes1 = axes1.flatten()

for i, vehicle_id in enumerate(selected_vehicle_ids):
    if i < len(axes1):  # 确保不超过子图数量
        orig_data = df[df['ID'] == vehicle_id].sort_values('time')
        smooth_data = df_smooth[df_smooth['ID'] == vehicle_id].sort_values('time')

        if len(orig_data) > 0 and len(smooth_data) > 0:
            # 计算当前子图的数据范围
            all_x = np.concatenate([orig_data['X'].values, smooth_data['X'].values])
            all_y = np.concatenate([orig_data['Y'].values, smooth_data['Y'].values])

            x_min, x_max = np.min(all_x), np.max(all_x)
            y_min, y_max = np.min(all_y), np.max(all_y)

            # 添加边距
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range == 0: x_range = 1
            if y_range == 0: y_range = 1

            margin_x = x_range * 0.05
            margin_y = y_range * 0.05

            # 绘制对比
            axes1[i].plot(orig_data['X'], orig_data['Y'],
                          linestyle='', marker='o', markersize=4, markerfacecolor='none',
                          markeredgecolor='blue', alpha=0.7, label='原始坐标')
            axes1[i].plot(smooth_data['X'], smooth_data['Y'],
                          linestyle='', marker='o', markersize=4, markerfacecolor='none',
                          markeredgecolor='red', alpha=0.7, label='平滑后坐标')

            axes1[i].set_xlim(x_min - margin_x, x_max + margin_x)
            axes1[i].set_ylim(y_min - margin_y, y_max + margin_y)
            axes1[i].set_title(f'车辆ID {int(vehicle_id)} X/Y坐标', fontsize=10)
            axes1[i].set_xlabel('X坐标', fontsize=8)
            axes1[i].set_ylabel('Y坐标', fontsize=8)
            axes1[i].grid(True, alpha=0.3)
            axes1[i].legend()

# 隐藏多余的子图
for j in range(len(selected_vehicle_ids), len(axes1)):
    axes1[j].set_visible(False)

plt.suptitle('X/Y坐标对比图 (随机选择的10辆车)', fontsize=16)
plt.tight_layout()
plt.show()

print("X/Y坐标对比图绘制完成")

# 2. 速度-时间对比图 (5行2列布局)
fig2, axes2 = plt.subplots(5, 2, figsize=(19.2, 10.8))
axes2 = axes2.flatten()

for i, vehicle_id in enumerate(selected_vehicle_ids):
    if i < len(axes2):  # 确保不超过子图数量
        orig_data = df[df['ID'] == vehicle_id].sort_values('time')
        smooth_data = df_smooth[df_smooth['ID'] == vehicle_id].sort_values('time')

        if len(orig_data) > 0 and len(smooth_data) > 0:
            # 计算当前子图的数据范围
            all_times = np.concatenate([orig_data['time'].values, smooth_data['time'].values])
            all_velocities = np.concatenate([orig_data['Velocity'].values, smooth_data['Velocity'].values])

            time_min, time_max = np.min(all_times), np.max(all_times)
            vel_min, vel_max = np.min(all_velocities), np.max(all_velocities)

            # 添加边距
            time_range = time_max - time_min
            vel_range = vel_max - vel_min
            if time_range == 0: time_range = 1
            if vel_range == 0: vel_range = 1

            margin_time = time_range * 0.05
            margin_vel = vel_range * 0.05

            # 绘制对比
            axes2[i].scatter(orig_data['time'], orig_data['Velocity'],
                             c='none', s=size, alpha=0.7, marker='o', edgecolors='blue', label='原始速度')
            axes2[i].scatter(smooth_data['time'], smooth_data['Velocity'],
                             c='none', s=size, alpha=0.7, marker='o', edgecolors='red', label='平滑后速度')

            axes2[i].set_xlim(time_min - margin_time, time_max + margin_time)
            axes2[i].set_ylim(vel_min - margin_vel, vel_max + margin_vel)
            axes2[i].set_title(f'车辆ID {int(vehicle_id)} 速度-时间', fontsize=10)
            axes2[i].set_xlabel('时间 (s)', fontsize=8)
            axes2[i].set_ylabel('速度 (m/s)', fontsize=8)
            axes2[i].grid(True, alpha=0.3)
            axes2[i].legend()

# 隐藏多余的子图
for j in range(len(selected_vehicle_ids), len(axes2)):
    axes2[j].set_visible(False)

plt.suptitle('速度-时间对比图 (随机选择的10辆车)', fontsize=16)
plt.tight_layout()
plt.show()

print("速度-时间对比图绘制完成")

# 3. 加速度-时间对比图 (5行2列布局)
fig3, axes3 = plt.subplots(5, 2, figsize=(19.2, 10.8))
axes3 = axes3.flatten()

for i, vehicle_id in enumerate(selected_vehicle_ids):
    if i < len(axes3):  # 确保不超过子图数量
        orig_data = df[df['ID'] == vehicle_id].sort_values('time')
        smooth_data = df_smooth[df_smooth['ID'] == vehicle_id].sort_values('time')

        if len(orig_data) > 0 and len(smooth_data) > 0:
            # 计算当前子图的数据范围
            all_times = np.concatenate([orig_data['time'].values, smooth_data['time'].values])
            all_accelerations = np.concatenate([orig_data['Acceleration'].values, smooth_data['Acceleration'].values])

            time_min, time_max = np.min(all_times), np.max(all_times)
            acc_min, acc_max = np.min(all_accelerations), np.max(all_accelerations)

            # 添加边距
            time_range = time_max - time_min
            acc_range = acc_max - acc_min
            if time_range == 0: time_range = 1
            if acc_range == 0: acc_range = 1

            margin_time = time_range * 0.05
            margin_acc = acc_range * 0.05

            # 绘制对比
            axes3[i].scatter(orig_data['time'], orig_data['Acceleration'],
                             c='none', s=size, alpha=0.7, marker='o', edgecolors='blue', label='原始加速度')
            axes3[i].scatter(smooth_data['time'], smooth_data['Acceleration'],
                             c='none', s=size, alpha=0.7, marker='o', edgecolors='red', label='平滑后加速度')

            axes3[i].set_xlim(time_min - margin_time, time_max + margin_time)
            axes3[i].set_ylim(acc_min - margin_acc, acc_max + margin_acc)
            axes3[i].set_title(f'车辆ID {int(vehicle_id)} 加速度-时间', fontsize=10)
            axes3[i].set_xlabel('时间 (s)', fontsize=8)
            axes3[i].set_ylabel('加速度 (m/s²)', fontsize=8)
            axes3[i].grid(True, alpha=0.3)
            axes3[i].legend()

# 隐藏多余的子图
for j in range(len(selected_vehicle_ids), len(axes3)):
    axes3[j].set_visible(False)

plt.suptitle('加速度-时间对比图 (随机选择的10辆车)', fontsize=16)
plt.tight_layout()
plt.show()

print("加速度-时间对比图绘制完成")

# 检查是否有负值
print("\n检查平滑后的数据是否有负值...")
negative_follow_dist_count = len(df_smooth[df_smooth['Following_dist'] < 0])
negative_headway_count = len(df_smooth[df_smooth['Time_Headway'] < 0])
negative_ttc_count = len(df_smooth[df_smooth['TTC'] < 0])
negative_distance_count = {}
for dist_col in distance_columns:
    if dist_col in df_smooth.columns:
        neg_count = len(df_smooth[df_smooth[dist_col] < 0])
        negative_distance_count[dist_col] = neg_count

print(f"跟车距离负值数量: {negative_follow_dist_count}")
print(f"车头时距负值数量: {negative_headway_count}")
print(f"碰撞时间负值数量: {negative_ttc_count}")
for dist_col, count in negative_distance_count.items():
    print(f"{dist_col}负值数量: {count}")

if negative_follow_dist_count > 0 or negative_headway_count > 0 or negative_ttc_count > 0 or any(
        count > 0 for count in negative_distance_count.values()):
    print("警告：仍有负值存在，需要进一步处理！")
else:
    print("成功：所有非负变量均无负值！")

# 保存平滑后的数据
print("正在保存平滑后的数据...")
df_smooth.to_pickle(traffic_pkl_flows_smooth_path)
df_smooth.to_csv(traffic_csv_flows_smooth_path, index=False, encoding='utf-8-sig')
print(f"平滑数据已保存到: {traffic_pkl_flows_smooth_path} ")

# 开始Z-Score标准化处理
print("\n开始Z-Score标准化处理...")

# 需要标准化的列（排除7个距离参数）
normalization_cols = [
    'X', 'Y',
    'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc', 'lat_Acc'
]

# 创建df_normalized作为df_smooth的副本
df_normalized = df_smooth.copy()

# 检查哪些列存在于数据中
available_cols = [col for col in normalization_cols if col in df_normalized.columns]
print(f"需要标准化的列: {available_cols}")

# 使用StandardScaler进行Z-Score标准化
scaler = StandardScaler()
df_normalized[available_cols] = scaler.fit_transform(df_normalized[available_cols])

print("Z-Score标准化处理完成")

# 保存归一化后的数据
print("正在保存归一化后的数据...")
df_normalized.to_pickle(traffic_pkl_flows_guiyi)
df_normalized.to_csv(traffic_csv_flows_guiyi, index=False, encoding='utf-8-sig')
print(f"归一化数据已保存到: {traffic_pkl_flows_guiyi} 和 {traffic_csv_flows_guiyi}")

# 输出标准化后的数据统计信息
print("\n标准化后的数据统计信息:")
for col in available_cols:
    mean_val = df_normalized[col].mean()
    std_val = df_normalized[col].std()
    print(f"{col}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")

# 清理内存
del df, df_smooth, df_normalized, vehicle_ids
gc.collect()
print("内存已清理完成")

print(f"\n{'=' * 50}")
print("数据平滑处理、可视化对比与Z-Score归一化完成！")
print(f"{'=' * 50}\n")