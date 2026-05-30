import pandas as pd
import numpy as np
import gc
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches

# 读取文件路径等预设信息保持不变，请根据实际情况调整
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
traffic_pkl_flows_path = os.path.join(save_dir, "traffic_flows_complete0.pkl")  # 待平滑数据汇集
traffic_pkl_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.pkl")  # 待平滑数据汇集
traffic_csv_flows_smooth_path = os.path.join(save_dir, "traffic_flows_smooth.csv")  # 待平滑数据汇集

# SG滤波参数
SG_WINDOW_LENGTH = 9  # 窗口长度
SG_POLYORDER = 3  # 多项式阶数

print(f"{'=' * 50}")
print(f"数据处理开始: 坐标数据SG滤波平滑处理与可视化对比（Y轴反转版+差值混合模式对比）")
print(f"SG滤波参数: 窗口长度={SG_WINDOW_LENGTH}, 阶数={SG_POLYORDER}")
print(f"显示范围: 不指定")
print(f"图像大小: 1920x1080")
print(f"Y轴方向: 从下到上数值从大到小")
print(f"{'=' * 50}\n")

# 读取数据
df = pd.read_pickle(traffic_pkl_flows_path)
print(f"数据读取完成，总数据量: {len(df):,} 行")

# 获取所有车辆ID
vehicle_ids = sorted(df['ID'].unique())
print(f"车辆ID数量: {len(vehicle_ids):,} 个")
print(f"车辆ID范围: {min(vehicle_ids)} - {max(vehicle_ids)}")

# 为每辆车分配颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(vehicle_ids)))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 绘制原始坐标图
fig1, ax1 = plt.subplots(figsize=(19.2, 10.8))
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df[df['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax1.plot(vehicle_data['X'], vehicle_data['Y'],
                 c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax1.set_title('原始坐标图 (Y轴反转)', fontsize=16)
ax1.set_xlabel('X坐标', fontsize=12)
ax1.set_ylabel('Y坐标', fontsize=12)
ax1.invert_yaxis()  # 反转Y轴
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("原始坐标图绘制完成 (Y轴已反转)")

# 对X和Y坐标应用SG滤波
df_smooth = df.copy()
for vehicle_id in vehicle_ids:
    vehicle_data = df[df['ID'] == vehicle_id].copy()
    if len(vehicle_data) >= SG_WINDOW_LENGTH:
        smooth_x = savgol_filter(vehicle_data['X'], window_length=SG_WINDOW_LENGTH, polyorder=SG_POLYORDER)
        smooth_y = savgol_filter(vehicle_data['Y'], window_length=SG_WINDOW_LENGTH, polyorder=SG_POLYORDER)
        df_smooth.loc[df_smooth['ID'] == vehicle_id, 'X'] = smooth_x
        df_smooth.loc[df_smooth['ID'] == vehicle_id, 'Y'] = smooth_y
    else:
        print(f"车辆ID {vehicle_id} 数据长度不足，跳过SG滤波处理")

print("SG滤波平滑处理完成")

# 绘制平滑后坐标图
fig2, ax2 = plt.subplots(figsize=(19.2, 10.8))
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df_smooth[df_smooth['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax2.plot(vehicle_data['X'], vehicle_data['Y'],
                 c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax2.set_title('SG滤波平滑后坐标图 (Y轴反转)', fontsize=16)
ax2.set_xlabel('X坐标', fontsize=12)
ax2.set_ylabel('Y坐标', fontsize=12)
ax2.invert_yaxis()  # 反转Y轴
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("SG滤波平滑后坐标图绘制完成 (Y轴已反转)")

# 差值对比
print("正在进行差值对比分析...")

# 计算差值数据
df_diff = df.copy()
df_diff['X_diff'] = df['X'] - df_smooth['X']
df_diff['Y_diff'] = df['Y'] - df_smooth['Y']
df_diff['Distance_diff'] = np.sqrt(df_diff['X_diff'] ** 2 + df_diff['Y_diff'] ** 2)

# 绘制差值图
fig3, ax3 = plt.subplots(figsize=(19.2, 10.8))
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df_diff[df_diff['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax3.plot(vehicle_data['X_diff'], vehicle_data['Y_diff'],
                 c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax3.set_title('原始数据与平滑后数据差值图', fontsize=16)
ax3.set_xlabel('X坐标差值', fontsize=12)
ax3.set_ylabel('Y坐标差值', fontsize=12)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("差值图绘制完成")

# 绘制距离差值图
fig4, ax4 = plt.subplots(figsize=(19.2, 10.8))
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df_diff[df_diff['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax4.plot(vehicle_data.index, vehicle_data['Distance_diff'],
                 c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax4.set_title('原始数据与平滑后数据距离差值图', fontsize=16)
ax4.set_xlabel('数据点索引', fontsize=12)
ax4.set_ylabel('距离差值', fontsize=12)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("距离差值图绘制完成")

# 使用matplotlib的混合模式进行图像差值对比
print("正在进行图像差值混合模式对比...")

# 创建原始图像和平滑后图像的灰度版本用于对比
fig_original = plt.figure(figsize=(19.2, 10.8))
ax_orig = fig_original.add_subplot(111)
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df[df['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax_orig.plot(vehicle_data['X'], vehicle_data['Y'],
                     c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax_orig.set_title('原始坐标图 (Y轴反转)', fontsize=16)
ax_orig.set_xlabel('X坐标', fontsize=12)
ax_orig.set_ylabel('Y坐标', fontsize=12)
ax_orig.invert_yaxis()
ax_orig.grid(True, alpha=0.3)
plt.tight_layout()

fig_smoothed = plt.figure(figsize=(19.2, 10.8))
ax_smooth = fig_smoothed.add_subplot(111)
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df_smooth[df_smooth['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax_smooth.plot(vehicle_data['X'], vehicle_data['Y'],
                       c=colors[i], label=f'ID {vehicle_id}', alpha=0.6)

ax_smooth.set_title('SG滤波平滑后坐标图 (Y轴反转)', fontsize=16)
ax_smooth.set_xlabel('X坐标', fontsize=12)
ax_smooth.set_ylabel('Y坐标', fontsize=12)
ax_smooth.invert_yaxis()
ax_smooth.grid(True, alpha=0.3)
plt.tight_layout()

# 显示原始图和平滑后图
plt.figure(fig_original.number)
plt.show()

plt.figure(fig_smoothed.number)
plt.show()

# 计算并显示差值图像
fig_diff = plt.figure(figsize=(19.2, 10.8))
ax_diff = fig_diff.add_subplot(111)

# 绘制原始数据（灰色）
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df[df['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax_diff.plot(vehicle_data['X'], vehicle_data['Y'],
                     c='gray', alpha=0.4, linewidth=1)

# 绘制平滑后数据（红色）
for i, vehicle_id in enumerate(vehicle_ids):
    vehicle_data = df_smooth[df_smooth['ID'] == vehicle_id]
    if len(vehicle_data) > 0:
        ax_diff.plot(vehicle_data['X'], vehicle_data['Y'],
                     c='red', alpha=0.6, linewidth=1)

ax_diff.set_title('原始数据(灰色)与平滑后数据(红色)对比图', fontsize=16)
ax_diff.set_xlabel('X坐标', fontsize=12)
ax_diff.set_ylabel('Y坐标', fontsize=12)
ax_diff.invert_yaxis()
ax_diff.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("图像差值对比完成")

# 统计差值信息
print("\n差值统计信息:")
print(f"X坐标差值范围: [{df_diff['X_diff'].min():.4f}, {df_diff['X_diff'].max():.4f}]")
print(f"Y坐标差值范围: [{df_diff['Y_diff'].min():.4f}, {df_diff['Y_diff'].max():.4f}]")
print(f"距离差值范围: [{df_diff['Distance_diff'].min():.4f}, {df_diff['Distance_diff'].max():.4f}]")
print(f"平均距离差值: {df_diff['Distance_diff'].mean():.4f}")
print(f"距离差值标准差: {df_diff['Distance_diff'].std():.4f}")

# 计算图像差异度（通过路径长度和位置变化来评估）
original_path_lengths = []
smoothed_path_lengths = []

for vehicle_id in vehicle_ids:
    orig_data = df[df['ID'] == vehicle_id].sort_values('time')
    smooth_data = df_smooth[df_smooth['ID'] == vehicle_id].sort_values('time')

    if len(orig_data) > 1:
        orig_distances = np.sqrt(np.diff(orig_data['X']) ** 2 + np.diff(orig_data['Y']) ** 2)
        original_path_lengths.append(np.sum(orig_distances))

    if len(smooth_data) > 1:
        smooth_distances = np.sqrt(np.diff(smooth_data['X']) ** 2 + np.diff(smooth_data['Y']) ** 2)
        smoothed_path_lengths.append(np.sum(smooth_distances))

avg_original_length = np.mean(original_path_lengths) if original_path_lengths else 0
avg_smoothed_length = np.mean(smoothed_path_lengths) if smoothed_path_lengths else 0

print(f"\n路径长度对比:")
print(f"原始数据平均路径长度: {avg_original_length:.4f}")
print(f"平滑后数据平均路径长度: {avg_smoothed_length:.4f}")
print(f"路径长度变化率: {((avg_smoothed_length - avg_original_length) / avg_original_length * 100):.2f}%")

# 保存平滑后的数据
print("\n正在保存平滑后的数据...")
df_smooth.to_pickle(traffic_pkl_flows_smooth_path)
df_smooth.to_csv(traffic_csv_flows_smooth_path, index=False, encoding='utf-8-sig')
print(f"平滑数据已保存到: {traffic_pkl_flows_smooth_path} 和 {traffic_csv_flows_smooth_path}")

# 清理内存
del df, df_smooth, df_diff, vehicle_ids, colors
gc.collect()
print("内存已清理完成")

print(f"\n{'=' * 50}")
print("数据平滑处理与可视化对比完成！")
print(f"{'=' * 50}\n")