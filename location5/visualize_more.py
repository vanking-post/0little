import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
input_file = r"E:\0little\read\CQSkyEyedata5\location5t\traffic_left_change.csv"
output_dir = r"E:\0little\read\CQSkyEyedata5\location5t\left_trajectory_vis"
os.makedirs(output_dir, exist_ok=True)

# ==================== 车道线配置（来自 visualizeXY.py） ====================
LANE_CONFIG = {
    'east': {
        'y_range': (20, 0),          # 东向西，Y倒置
        'x_range': (0, 340),
        'lane_lines': [6.5, 10, 13.75],
        'label': '东向 (Y 0~20)',
        'direction_arrow': (0.15, 0.1, 0.5, 0.1, '从东向西')
    },
    'west': {
        'y_range': (40, 20),         # 西向东，Y倒置
        'x_range': (0, 340),
        'lane_lines': [23.5, 27.25, 31],
        'label': '西向 (Y 20~40)',
        'direction_arrow': (0.85, 0.1, 0.5, 0.1, '从西向东')
    }
}


def add_lane_lines(ax, y_center):
    """根据Y坐标中心值自动选择车道线方向并画线"""
    if 0 <= y_center <= 20:
        cfg = LANE_CONFIG['east']
        # 添加行驶方向箭头标注
        ax.annotate(cfg['direction_arrow'][4],
                    xy=(cfg['direction_arrow'][0], cfg['direction_arrow'][1]),
                    xytext=(cfg['direction_arrow'][2], cfg['direction_arrow'][3]),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='green', edgecolor='gray',
                                    width=1.5, headwidth=7, shrink=0.05, alpha=0.8),
                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)
    else:
        cfg = LANE_CONFIG['west']
        ax.annotate(cfg['direction_arrow'][4],
                    xy=(cfg['direction_arrow'][0], cfg['direction_arrow'][1]),
                    xytext=(cfg['direction_arrow'][2], cfg['direction_arrow'][3]),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='green', edgecolor='gray',
                                    width=1.5, headwidth=7, shrink=0.05, alpha=0.8),
                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)

    for ly in cfg['lane_lines']:
        ax.plot(cfg['x_range'], [ly, ly], color='black', linewidth=0.8,
                linestyle='--', alpha=0.6)

    ax.set_xlim(cfg['x_range'])
    ax.set_ylim(cfg['y_range'])


print("加载数据...")
df = pd.read_csv(input_file)
df = df.sort_values(['ID', 'Frame'])
df['FrameIdx'] = df.groupby('ID').cumcount()  # 0~49

# 速度单位（已在 step04 转为 m/s）
df['Velocity_ms'] = df['Velocity']
df['Velocity_kmh'] = (df['Velocity'] * 3.6).round(1)

n_vehicles = df['ID'].nunique()
print(f"总车辆数: {n_vehicles}, 总行数: {len(df)}")

# 按起始车道分组（后续用）
lane_groups = df.groupby('ID')['LaneID'].first().reset_index()
lane_groups.columns = ['ID', 'StartLane']
df = df.merge(lane_groups, on='ID')

# ==================== 图1. X-Y 轨迹总览 ====================
# 使用 subplot2grid 布局：第一行大图XY轨迹，第二行速度/TTC/加速度
fig = plt.figure(figsize=(24, 16))

# 第一行：XY轨迹图（占满整行，跨2行3列）
ax_xy = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)

rng = np.random.default_rng(42)
sample_ids = rng.choice(df['ID'].unique(), min(30, n_vehicles), replace=False)
print(sample_ids)
colors = plt.cm.tab20(np.linspace(0, 1, len(sample_ids)))

y_center = df['Y'].median()
for idx, vid in enumerate(sample_ids):
    veh = df[df['ID'] == vid]
    x = veh['X'].values
    y = veh['Y'].values
    ax_xy.plot(x, y, color=colors[idx], linewidth=1.5, alpha=0.8)
    ax_xy.scatter(x[-1], y[-1], color=colors[idx], s=60, marker='D',
                  edgecolors='black', linewidths=0.5)
    ax_xy.scatter(x[0], y[0], color=colors[idx], s=40, marker='o',
                  edgecolors='black', linewidths=0.5)

# ★ 添加车道线
add_lane_lines(ax_xy, y_center)

ax_xy.set_xlabel('X 坐标 (m)', fontsize=14)
ax_xy.set_ylabel('Y 坐标 (m)', fontsize=14)
ax_xy.set_title(f'左变道车辆 X-Y 轨迹 (随机30/{n_vehicles}辆)', fontsize=16, fontweight='bold')
ax_xy.legend(['轨迹', '终点(变道前)', '起点'], loc='upper right', fontsize=10)
ax_xy.grid(True, alpha=0.3)
ax_xy.set_aspect('equal')

# 第二行左：速度时序图
ax_vel = plt.subplot2grid((3, 3), (2, 0))
for idx, vid in enumerate(sample_ids[:12]):
    veh = df[df['ID'] == vid]
    ax_vel.plot(veh['FrameIdx'], veh['Velocity_ms'], color=colors[idx],
                linewidth=1.5, alpha=0.8)
ax_vel.set_xlabel('变道前帧序号 (0→49)', fontsize=12)
ax_vel.set_ylabel('速度 (m/s)', fontsize=12)
ax_vel.set_title(f'左变道车速度变化 (随机12辆)', fontsize=13)
ax_vel.grid(True, alpha=0.3)

# 第二行中：TTC 时序图
ax_ttc = plt.subplot2grid((3, 3), (2, 1))
for idx, vid in enumerate(sample_ids[:12]):
    veh = df[df['ID'] == vid]
    if 'TTC' in veh.columns:
        ax_ttc.plot(veh['FrameIdx'], veh['TTC'], color=colors[idx],
                    linewidth=1.5, alpha=0.8)
ax_ttc.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
               label='危险阈值 2s')
ax_ttc.set_xlabel('变道前帧序号 (0→49)', fontsize=12)
ax_ttc.set_ylabel('TTC (s)', fontsize=12)
ax_ttc.set_title(f'左变道车 TTC 变化 (随机12辆)', fontsize=13)
ax_ttc.legend(fontsize=8)
ax_ttc.grid(True, alpha=0.3)

# 第二行右：加速度时序图
ax_acc = plt.subplot2grid((3, 3), (2, 2))
for idx, vid in enumerate(sample_ids[:12]):
    veh = df[df['ID'] == vid]
    ax_acc.plot(veh['FrameIdx'], veh['Acceleration'], color=colors[idx],
                linewidth=1.5, alpha=0.8)
ax_acc.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax_acc.set_xlabel('变道前帧序号 (0→49)', fontsize=12)
ax_acc.set_ylabel('加速度 (m/s²)', fontsize=12)
ax_acc.set_title(f'左变道车加速度变化 (随机12辆)', fontsize=13)
ax_acc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_trajectory_overview.png'), dpi=150)
plt.close()
print("✓ 图1: X-Y轨迹总览（含车道线）")

# ==================== 图2. 单辆车的完整可视化（随机3辆） ====================
sample_vids = rng.choice(df['ID'].unique(), min(3, n_vehicles), replace=False)

for vid in sample_vids:
    veh = df[df['ID'] == vid]
    pet_val = veh['PET'].iloc[0]
    lane_ids = veh['LaneID'].unique()

    fig = plt.figure(figsize=(20, 12))
    # 手动搭建 2×2 布局，左上角为 3D
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')      # 3D: X-Y-Velocity
    ax_safe = fig.add_subplot(2, 2, 2)                    # 安全指标
    ax_dist = fig.add_subplot(2, 2, 3)                    # 距离指标
    ax_kinem = fig.add_subplot(2, 2, 4)                   # 运动学

    indicator_groups_ordered = [
        ('位置与速度', [('X', 'X坐标 (m)'), ('Y', 'Y坐标 (m)'), ('Velocity_ms', '速度 (m/s)')]),
        ('安全指标',    [('TTC', 'TTC (s)'), ('mTTC', 'mTTC (s)'),
                         ('Time_Headway', 'THW (s)'), ('Following_dist', '跟驰距离 (m)')]),
        ('距离指标',    [('LB_Dist', '左后距 (m)'), ('LF_Dist', '左前距 (m)'),
                         ('B_Dist', '后距 (m)'), ('RF_Dist', '右前距 (m)')]),
        ('运动学',      [('Acceleration', '加速度 (m/s²)'), ('lat_Acc', '横向加速度 (m/s²)'),
                         ('Lateral_Jerk', '横向急动度')]),
    ]

    for group_name, cols in indicator_groups_ordered:
        if group_name == '位置与速度':
            # ★★★ 3D 轨迹图：X-Y 平面 + 速度 Z 轴 ★★★
            x_vals = veh['X'].values if 'X' in veh.columns else None
            y_vals = veh['Y'].values if 'Y' in veh.columns else None
            v_vals = veh['Velocity_ms'].values if 'Velocity_ms' in veh.columns else None
            time_idx = veh['FrameIdx'].values

            # 用速度大小映射颜色
            if x_vals is not None and y_vals is not None and v_vals is not None:
                sc = ax_3d.scatter(x_vals, y_vals, v_vals,
                                   c=time_idx, cmap='viridis', s=30,
                                   edgecolors='black', linewidths=0.3, alpha=0.9)
                # 连线（投影到底面 + 空间线）
                ax_3d.plot(x_vals, y_vals, v_vals,
                           color='gray', linewidth=1.0, alpha=0.5)
                # 起点/终点高亮
                ax_3d.scatter([x_vals[0]], [y_vals[0]], [v_vals[0]],
                              color='green', s=80, marker='o',
                              edgecolors='black', linewidths=1.0, label='起点')
                ax_3d.scatter([x_vals[-1]], [y_vals[-1]], [v_vals[-1]],
                              color='red', s=80, marker='D',
                              edgecolors='black', linewidths=1.0, label='变道前终点')
                cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.5, aspect=20, pad=0.05)
                cbar.set_label('帧序号 (0→49)', fontsize=9)

            ax_3d.set_xlabel('X (m)', fontsize=10)
            ax_3d.set_ylabel('Y (m)', fontsize=10)
            ax_3d.set_zlabel('速度 (m/s)', fontsize=10)
            ax_3d.set_title('三维轨迹 (X-Y-速度)', fontsize=12, fontweight='bold')
            ax_3d.legend(fontsize=9, loc='upper left')

        elif group_name == '安全指标':
            # 跟驰距离放在右边 Y 轴，范围根据该车跟驰距离决定
            ax_left = ax_safe
            ax_right = ax_safe.twinx()
            lines_left, labels_left = [], []
            lines_right, labels_right = [], []
            for col, label in cols:
                if col not in veh.columns:
                    continue
                if col == 'Following_dist':
                    line = ax_right.plot(veh['FrameIdx'], veh[col],
                                         color='tab:olive', marker='.', markersize=3,
                                         linewidth=1.5, label=label)
                    lines_right.append(line[0])
                    labels_right.append(label)
                    # 根据该车跟驰距离确定右轴范围
                    v = veh[col].dropna()
                    if len(v) > 0:
                        f_min, f_max = v.min(), v.max()
                        f_range = f_max - f_min if f_max > f_min else 1
                        ax_right.set_ylim(f_min - 0.1*f_range - 1, f_max + 0.1*f_range + 1)
                else:
                    line = ax_left.plot(veh['FrameIdx'], veh[col],
                                        marker='.', markersize=3,
                                        linewidth=1.5, label=label)
                    lines_left.append(line[0])
                    labels_left.append(label)
            ax_left.set_xlabel('帧序号', fontsize=12)
            ax_left.set_ylabel('TTC / THW (s)', fontsize=12)
            ax_left.set_title(group_name, fontsize=13)
            ax_left.grid(True, alpha=0.3)
            ax_right.set_ylabel('跟驰距离 (m)', fontsize=12, color='tab:olive')
            ax_right.tick_params(axis='y', labelcolor='tab:olive')
            all_lines = lines_left + lines_right
            all_labels = labels_left + labels_right
            ax_left.legend(all_lines, all_labels, fontsize=9, loc='best')

        elif group_name == '距离指标':
            for col, label in cols:
                if col in veh.columns:
                    ax_dist.plot(veh['FrameIdx'], veh[col], marker='.', markersize=3,
                                 linewidth=1.5, label=label)
            ax_dist.set_xlabel('帧序号', fontsize=12)
            ax_dist.set_ylabel('距离 (m)', fontsize=12)
            ax_dist.set_title(group_name, fontsize=13)
            ax_dist.legend(fontsize=9, loc='best')
            ax_dist.grid(True, alpha=0.3)

        elif group_name == '运动学':
            for col, label in cols:
                if col in veh.columns:
                    ax_kinem.plot(veh['FrameIdx'], veh[col], marker='.', markersize=3,
                                  linewidth=1.5, label=label)
            ax_kinem.set_xlabel('帧序号', fontsize=12)
            ax_kinem.set_ylabel('值', fontsize=12)
            ax_kinem.set_title(group_name, fontsize=13)
            ax_kinem.legend(fontsize=9, loc='best')
            ax_kinem.grid(True, alpha=0.3)

    fig.suptitle(f'左变道车辆 {int(vid)} | 速度={veh["Velocity_kmh"].iloc[0]:.1f}km/h '
                 f'| PET={pet_val:.2f}s | 车道={list(lane_ids)}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'02_single_vehicle_{int(vid)}.png'), dpi=150)
    plt.close()
    print(f"✓ 图2: 车辆 {int(vid)} 详细可视化")

# ==================== 图3. 车道分组平均轨迹 + 标准差 ====================
fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# ★ 3a) X-Y 每个车道的平均轨迹（增加车道线）
ax = axes[0, 0]
y_center_all = df['Y'].median()
for lane in sorted(df['StartLane'].unique()):
    lane_data = df[df['StartLane'] == lane]
    pivot_x = lane_data.pivot_table(index='FrameIdx', columns='ID',
                                     values='X', aggfunc='first')
    pivot_y = lane_data.pivot_table(index='FrameIdx', columns='ID',
                                     values='Y', aggfunc='first')
    mean_x = pivot_x.mean(axis=1)
    mean_y = pivot_y.mean(axis=1)
    ax.plot(mean_x, mean_y, linewidth=2,
            label=f'起始车道 {int(lane)} (n={pivot_x.shape[1]})')
    ax.scatter(mean_x.iloc[0], mean_y.iloc[0], s=60, marker='o',
               edgecolors='black', linewidths=0.8)
    ax.scatter(mean_x.iloc[-1], mean_y.iloc[-1], s=80, marker='D',
               edgecolors='black', linewidths=0.8)

add_lane_lines(ax, y_center_all)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('各起始车道平均 X-Y 轨迹', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 3b) 速度平均曲线
ax = axes[0, 1]
for lane in sorted(df['StartLane'].unique()):
    lane_data = df[df['StartLane'] == lane]
    pivot_v = lane_data.pivot_table(index='FrameIdx', columns='ID',
                                     values='Velocity_ms', aggfunc='first')
    mean_v = pivot_v.mean(axis=1)
    ax.plot(range(50), mean_v, linewidth=2, label=f'起始车道 {int(lane)}')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('速度 (m/s)', fontsize=12)
ax.set_title('各起始车道平均速度', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3c) TTC 平均曲线
ax = axes[0, 2]
for lane in sorted(df['StartLane'].unique()):
    lane_data = df[df['StartLane'] == lane]
    pivot_ttc = lane_data.pivot_table(index='FrameIdx', columns='ID',
                                       values='TTC', aggfunc='first')
    mean_ttc = pivot_ttc.mean(axis=1)
    ax.plot(range(50), mean_ttc, linewidth=2, label=f'起始车道 {int(lane)}')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('TTC (s)', fontsize=12)
ax.set_title('各起始车道平均 TTC', fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3d) Velocity 带标准差阴影
ax = axes[1, 0]
all_pivot_v = df.pivot_table(index='FrameIdx', columns='ID',
                               values='Velocity_ms', aggfunc='first')
mean_all_v = all_pivot_v.mean(axis=1)
std_all_v = all_pivot_v.std(axis=1)
ax.plot(range(50), mean_all_v, color='blue', linewidth=2.5, label='均值')
ax.fill_between(range(50), mean_all_v - std_all_v, mean_all_v + std_all_v,
                alpha=0.2, color='blue')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('速度 (m/s)', fontsize=12)
ax.set_title(f'全部 {n_vehicles} 辆平均速度 (±1σ)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3e) TTC 带标准差阴影
ax = axes[1, 1]
all_pivot_ttc = df.pivot_table(index='FrameIdx', columns='ID',
                                 values='TTC', aggfunc='first')
mean_all_ttc = all_pivot_ttc.mean(axis=1)
std_all_ttc = all_pivot_ttc.std(axis=1)
ax.plot(range(50), mean_all_ttc, color='red', linewidth=2.5, label='均值')
ax.fill_between(range(50), mean_all_ttc - std_all_ttc, mean_all_ttc + std_all_ttc,
                alpha=0.2, color='red')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('TTC (s)', fontsize=12)
ax.set_title(f'全部 {n_vehicles} 辆平均 TTC (±1σ)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 3f) mTTC 带标准差阴影
ax = axes[1, 2]
all_pivot_mttc = df.pivot_table(index='FrameIdx', columns='ID',
                                  values='mTTC', aggfunc='first')
mean_all_mttc = all_pivot_mttc.mean(axis=1)
std_all_mttc = all_pivot_mttc.std(axis=1)
ax.plot(range(50), mean_all_mttc, color='orange', linewidth=2.5, label='均值')
ax.fill_between(range(50), mean_all_mttc - std_all_mttc, mean_all_mttc + std_all_mttc,
                alpha=0.2, color='orange')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('mTTC (s)', fontsize=12)
ax.set_title(f'全部 {n_vehicles} 辆平均 mTTC (±1σ)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('左变道车辆轨迹组分析', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_group_analysis.png'), dpi=150)
plt.close()
print("✓ 图3: 车道分组分析（含车道线）")

# ==================== 图4. 速度与TTC分布带 ====================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# 速度分布带
ax = axes[0]
vel_matrix = df.pivot_table(index='FrameIdx', columns='ID',
                              values='Velocity_ms', aggfunc='first')
p10 = vel_matrix.quantile(0.1, axis=1)
p25 = vel_matrix.quantile(0.25, axis=1)
p50 = vel_matrix.quantile(0.5, axis=1)
p75 = vel_matrix.quantile(0.75, axis=1)
p90 = vel_matrix.quantile(0.9, axis=1)

ax.fill_between(range(50), p10, p90, alpha=0.15, color='blue', label='P10-P90')
ax.fill_between(range(50), p25, p75, alpha=0.25, color='blue', label='P25-P75')
ax.plot(range(50), p50, color='darkblue', linewidth=2.5, label='中位数(P50)')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('速度 (m/s)', fontsize=12)
ax.set_title(f'速度分布 ({n_vehicles}辆)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# TTC 分布带
ax = axes[1]
ttc_matrix = df.pivot_table(index='FrameIdx', columns='ID',
                               values='TTC', aggfunc='first')
p10_t = ttc_matrix.quantile(0.1, axis=1)
p25_t = ttc_matrix.quantile(0.25, axis=1)
p50_t = ttc_matrix.quantile(0.5, axis=1)
p75_t = ttc_matrix.quantile(0.75, axis=1)
p90_t = ttc_matrix.quantile(0.9, axis=1)

ax.fill_between(range(50), p10_t, p90_t, alpha=0.15, color='red', label='P10-P90')
ax.fill_between(range(50), p25_t, p75_t, alpha=0.25, color='red', label='P25-P75')
ax.plot(range(50), p50_t, color='darkred', linewidth=2.5, label='中位数(P50)')
ax.axhline(y=2, color='orange', linestyle='--', linewidth=1.5,
           label='危险阈值 (TTC=2s)')
ax.set_xlabel('帧序号', fontsize=12)
ax.set_ylabel('TTC (s)', fontsize=12)
ax.set_title(f'TTC 分布 ({n_vehicles}辆)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('左变道 - 速度与 TTC 分布带', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_distribution_bands.png'), dpi=150)
plt.close()
print("✓ 图4: 分布带图")

print(f"\n✅ 所有图片已保存至: {output_dir}")