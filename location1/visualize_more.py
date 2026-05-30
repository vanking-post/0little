import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import os

from step_visualizeXY import (lane_coeffs_dir11, lane_coeffs_dir12,
                               lane_coeffs_dir21, lane_coeffs_dir22,
                               lane_coeffs_dir31, lane_coeffs_dir32)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
input_dir = r"E:\0little\location1"
output_base = r"E:\0little\location1"
loc5_dir = r"E:\0little\read\CQSkyEyedata5\location5t"

# 路段 → 方向 → 车道多项式系数
LANE_COEFFS = {
    '1-1':  {1: lane_coeffs_dir11, 2: lane_coeffs_dir12},
    '1-2':  {1: lane_coeffs_dir21, 2: lane_coeffs_dir22},
    'loc5': {1: lane_coeffs_dir31, 2: lane_coeffs_dir32},
}

# 各路段配置: X范围, Y范围(按方向), 方向判定阈值, 全幅Y范围
SRC_CONFIG = {
    '1-1':  {'x_range': (0, 420), 'y_dir': {1: (25, 5), 2: (45, 25)}, 'y_thresh': 25, 'y_full': (45, 5)},
    '1-2':  {'x_range': (0, 420), 'y_dir': {1: (25, 5), 2: (45, 25)}, 'y_thresh': 25, 'y_full': (45, 5)},
    'loc5': {'x_range': (0, 350), 'y_dir': {1: (20, 0), 2: (40, 20)}, 'y_thresh': 20, 'y_full': (40, 0)},
}


def get_direction(y_values, threshold=25):
    """根据车辆 Y 中位数判断方向: 1 (Y<threshold) 或 2 (Y>=threshold)"""
    return 1 if np.median(y_values) < threshold else 2


def draw_lane_lines(ax, source, x_range=None):
    """在指定 axes 上绘制当前路段所有车道线（方向1+方向2），高对比度虚线"""
    if x_range is None:
        x_range = SRC_CONFIG[source]['x_range']
    x_vals = np.linspace(x_range[0], x_range[1], 500)
    styles = {1: (0.8, (8, 4)), 2: (0.8, (3, 5))}  # (linewidth, dash_pattern)
    for d, (lw, dash) in styles.items():
        for coeffs in LANE_COEFFS[source][d]:
            y_vals = np.polyval(coeffs, x_vals)
            # 白色外描边提高对比度
            ax.plot(x_vals, y_vals, color='white', linewidth=lw + 2.5, linestyle='-',
                    alpha=0.95, zorder=1)
            ax.plot(x_vals, y_vals, color='black', linewidth=lw, linestyle=(0, dash),
                    alpha=0.9, zorder=2)


def process_behavior(df, behavior_label, source, output_dir):
    """对单个路段+变道行为生成 4 张可视化图"""
    os.makedirs(output_dir, exist_ok=True)

    cfg = SRC_CONFIG[source]
    x_range = cfg['x_range']
    y_thresh = cfg['y_thresh']
    y_full = cfg['y_full']
    y_dir = cfg['y_dir']

    df = df.sort_values(['ID', 'Frame']).reset_index(drop=True)
    df['FrameIdx'] = df.groupby('ID').cumcount()
    df['Velocity_ms'] = df['Velocity']
    df['Velocity_kmh'] = (df['Velocity'] * 3.6).round(1)

    # 确定每辆车所属方向
    vehicle_dir = df.groupby('ID')['Y'].median().apply(lambda v: get_direction(v, y_thresh))
    df['Dir'] = df['ID'].map(vehicle_dir)
    dominant_dir = df['Dir'].mode().iloc[0]

    n_vehicles = df['ID'].nunique()
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(df['ID'].unique(), min(30, n_vehicles), replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(sample_ids)))

    print(f"[{source} {behavior_label}] 总车辆数: {n_vehicles}, 总行数: {len(df)}")
    print(f"  主导方向: {dominant_dir}")

    # ==================== 图1. X-Y 轨迹总览 ====================
    fig = plt.figure(figsize=(24, 16))
    ax_xy = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)

    for idx, vid in enumerate(sample_ids):
        veh = df[df['ID'] == vid]
        x, y = veh['X'].values, veh['Y'].values
        ax_xy.plot(x, y, color=colors[idx], linewidth=1.5, alpha=0.8)
        ax_xy.scatter(x[-1], y[-1], color=colors[idx], s=60, marker='D',
                      edgecolors='black', linewidths=0.5)
        ax_xy.scatter(x[0], y[0], color=colors[idx], s=40, marker='o',
                      edgecolors='black', linewidths=0.5)

    draw_lane_lines(ax_xy, source)
    ax_xy.set_xlim(x_range)
    ax_xy.set_ylim(y_full)
    ax_xy.set_xlabel('X (m)', fontsize=14)
    ax_xy.set_ylabel('Y (m)', fontsize=14)
    ax_xy.set_title(f'{source} {behavior_label}车辆 X-Y 轨迹 (随机{min(30, n_vehicles)}/{n_vehicles}辆)',
                    fontsize=16, fontweight='bold')
    ax_xy.legend(['轨迹', '终点(变道前)', '起点'], loc='upper right', fontsize=10)
    # ax_xy.grid(True, alpha=0.3)
    ax_xy.set_aspect('equal')

    ax_vel = plt.subplot2grid((3, 3), (2, 0))
    for idx, vid in enumerate(sample_ids[:12]):
        veh = df[df['ID'] == vid]
        ax_vel.plot(veh['FrameIdx'], veh['Velocity_ms'], color=colors[idx], linewidth=1.5, alpha=0.8)
    ax_vel.set_xlabel('变道前帧序号 (0->49)', fontsize=12)
    ax_vel.set_ylabel('速度 (m/s)', fontsize=12)
    ax_vel.set_title(f'{behavior_label}车速度变化 (随机12辆)', fontsize=13)
    # ax_vel.grid(True, alpha=0.3)

    ax_ttc = plt.subplot2grid((3, 3), (2, 1))
    for idx, vid in enumerate(sample_ids[:12]):
        veh = df[df['ID'] == vid]
        if 'TTC' in veh.columns:
            ax_ttc.plot(veh['FrameIdx'], veh['TTC'], color=colors[idx], linewidth=1.5, alpha=0.8)
    ax_ttc.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='危险阈值 2s')
    ax_ttc.set_xlabel('变道前帧序号 (0->49)', fontsize=12)
    ax_ttc.set_ylabel('TTC (s)', fontsize=12)
    ax_ttc.set_title(f'{behavior_label}车 TTC 变化 (随机12辆)', fontsize=13)
    ax_ttc.legend(fontsize=8)
    # ax_ttc.grid(True, alpha=0.3)

    ax_acc = plt.subplot2grid((3, 3), (2, 2))
    for idx, vid in enumerate(sample_ids[:12]):
        veh = df[df['ID'] == vid]
        if 'Acceleration' in veh.columns:
            ax_acc.plot(veh['FrameIdx'], veh['Acceleration'], color=colors[idx], linewidth=1.5, alpha=0.8)
    ax_acc.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_acc.set_xlabel('变道前帧序号 (0->49)', fontsize=12)
    ax_acc.set_ylabel('加速度 (m/s^2)', fontsize=12)
    ax_acc.set_title(f'{behavior_label}车加速度变化 (随机12辆)', fontsize=13)
    # ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_trajectory_overview.png'), dpi=150)
    plt.close()
    print("  ✓ 图1: X-Y轨迹总览")

    # ==================== 图2. 单辆车详细可视化（随机3辆） ====================
    sample_vids = rng.choice(df['ID'].unique(), min(3, n_vehicles), replace=False)

    for vid in sample_vids:
        veh = df[df['ID'] == vid]
        pet_val = veh['PET'].iloc[0] if 'PET' in veh.columns else np.nan
        lane_ids = veh['LaneID'].unique() if 'LaneID' in veh.columns else []

        fig = plt.figure(figsize=(20, 12))
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_safe = fig.add_subplot(2, 2, 2)
        ax_dist = fig.add_subplot(2, 2, 3)
        ax_kinem = fig.add_subplot(2, 2, 4)

        indicator_groups_ordered = [
            ('位置与速度', [('X', 'X (m)'), ('Y', 'Y (m)'), ('Velocity_ms', '速度 (m/s)')]),
            ('安全指标',    [('TTC', 'TTC (s)'), ('mTTC', 'mTTC (s)'),
                             ('Time_Headway', 'THW (s)'), ('Following_dist', '跟驰距离 (m)')]),
            ('距离指标',    [('LB_Dist', '左后距 (m)'), ('LF_Dist', '左前距 (m)'),
                             ('B_Dist', '后距 (m)'), ('RF_Dist', '右前距 (m)')]),
            ('运动学',      [('Acceleration', '加速度 (m/s^2)'), ('lat_Acc', '横向加速度 (m/s^2)'),
                             ('Lateral_Jerk', '横向急动度')]),
        ]

        for group_name, cols in indicator_groups_ordered:
            if group_name == '位置与速度':
                x_vals = veh['X'].values
                y_vals = veh['Y'].values
                v_vals = veh['Velocity_ms'].values
                time_idx = veh['FrameIdx'].values

                sc = ax_3d.scatter(x_vals, y_vals, v_vals,
                                   c=time_idx, cmap='viridis', s=30,
                                   edgecolors='black', linewidths=0.3, alpha=0.9)
                ax_3d.plot(x_vals, y_vals, v_vals, color='gray', linewidth=1.0, alpha=0.5)
                ax_3d.scatter([x_vals[0]], [y_vals[0]], [v_vals[0]],
                              color='green', s=80, marker='o',
                              edgecolors='black', linewidths=1.0, label='起点')
                ax_3d.scatter([x_vals[-1]], [y_vals[-1]], [v_vals[-1]],
                              color='red', s=80, marker='D',
                              edgecolors='black', linewidths=1.0, label='变道前终点')
                cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.5, aspect=20, pad=0.05)
                cbar.set_label('帧序号 (0->49)', fontsize=9)
                ax_3d.set_xlabel('X (m)', fontsize=10)
                ax_3d.set_ylabel('Y (m)', fontsize=10)
                ax_3d.set_zlabel('速度 (m/s)', fontsize=10)
                ax_3d.set_title('三维轨迹 (X-Y-速度)', fontsize=12, fontweight='bold')
                ax_3d.legend(fontsize=9, loc='upper left')

            elif group_name == '安全指标':
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
                        v = veh[col].dropna()
                        if len(v) > 0:
                            f_min, f_max = v.min(), v.max()
                            f_range = f_max - f_min if f_max > f_min else 1
                            ax_right.set_ylim(f_min - 0.1 * f_range - 1, f_max + 0.1 * f_range + 1)
                    else:
                        line = ax_left.plot(veh['FrameIdx'], veh[col],
                                            marker='.', markersize=3, linewidth=1.5, label=label)
                        lines_left.append(line[0])
                        labels_left.append(label)
                ax_left.set_xlabel('帧序号', fontsize=12)
                ax_left.set_ylabel('TTC / THW (s)', fontsize=12)
                ax_left.set_title(group_name, fontsize=13)
                # ax_left.grid(True, alpha=0.3)
                ax_right.set_ylabel('跟驰距离 (m)', fontsize=12, color='tab:olive')
                ax_right.tick_params(axis='y', labelcolor='tab:olive')
                ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                               fontsize=9, loc='best')

            elif group_name == '距离指标':
                for col, label in cols:
                    if col in veh.columns:
                        ax_dist.plot(veh['FrameIdx'], veh[col], marker='.', markersize=3,
                                     linewidth=1.5, label=label)
                ax_dist.set_xlabel('帧序号', fontsize=12)
                ax_dist.set_ylabel('距离 (m)', fontsize=12)
                ax_dist.set_title(group_name, fontsize=13)
                ax_dist.legend(fontsize=9, loc='best')
                # ax_dist.grid(True, alpha=0.3)

            elif group_name == '运动学':
                for col, label in cols:
                    if col in veh.columns:
                        ax_kinem.plot(veh['FrameIdx'], veh[col], marker='.', markersize=3,
                                      linewidth=1.5, label=label)
                ax_kinem.set_xlabel('帧序号', fontsize=12)
                ax_kinem.set_ylabel('值', fontsize=12)
                ax_kinem.set_title(group_name, fontsize=13)
                ax_kinem.legend(fontsize=9, loc='best')
                # ax_kinem.grid(True, alpha=0.3)

        fig.suptitle(f'{source} {behavior_label}车辆 {int(vid)} | 速度={veh["Velocity_kmh"].iloc[0]:.1f}km/h '
                     f'| PET={pet_val:.2f}s | 车道={list(lane_ids)}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'02_single_vehicle_{int(vid)}.png'), dpi=150)
        plt.close()
        print(f"  ✓ 图2: 车辆 {int(vid)} 详细可视化")

    # ==================== 图3. 车道分组平均轨迹 ====================
    df_lanes = df.copy()
    if 'StartLane' not in df_lanes.columns:
        df_lanes['StartLane'] = df_lanes.groupby('ID')['LaneID'].transform('first')

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # 3a) 各起始车道平均 X-Y 轨迹
    ax = axes[0, 0]
    for lane in sorted(df_lanes['StartLane'].dropna().unique()):
        lane_data = df_lanes[df_lanes['StartLane'] == lane]
        pivot_x = lane_data.pivot_table(index='FrameIdx', columns='ID', values='X', aggfunc='first')
        pivot_y = lane_data.pivot_table(index='FrameIdx', columns='ID', values='Y', aggfunc='first')
        mean_x, mean_y = pivot_x.mean(axis=1), pivot_y.mean(axis=1)
        ax.plot(mean_x, mean_y, linewidth=2, label=f'起始车道 {int(lane)} (n={pivot_x.shape[1]})')
        ax.scatter(mean_x.iloc[0], mean_y.iloc[0], s=60, marker='o', edgecolors='black', linewidths=0.8)
        ax.scatter(mean_x.iloc[-1], mean_y.iloc[-1], s=80, marker='D', edgecolors='black', linewidths=0.8)

    draw_lane_lines(ax, source)
    ax.set_xlim(x_range)
    ax.set_ylim(y_dir[dominant_dir])
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('各起始车道平均 X-Y 轨迹', fontsize=14)
    ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 3b) 速度平均曲线
    ax = axes[0, 1]
    for lane in sorted(df_lanes['StartLane'].dropna().unique()):
        lane_data = df_lanes[df_lanes['StartLane'] == lane]
        pivot_v = lane_data.pivot_table(index='FrameIdx', columns='ID', values='Velocity_ms', aggfunc='first')
        ax.plot(pivot_v.index, pivot_v.mean(axis=1), linewidth=2, label=f'起始车道 {int(lane)}')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('速度 (m/s)', fontsize=12)
    ax.set_title('各起始车道平均速度', fontsize=14)
    ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)

    # 3c) TTC 平均曲线
    ax = axes[0, 2]
    for lane in sorted(df_lanes['StartLane'].dropna().unique()):
        lane_data = df_lanes[df_lanes['StartLane'] == lane]
        if 'TTC' in lane_data.columns:
            pivot_ttc = lane_data.pivot_table(index='FrameIdx', columns='ID', values='TTC', aggfunc='first')
            ax.plot(pivot_ttc.index, pivot_ttc.mean(axis=1), linewidth=2, label=f'起始车道 {int(lane)}')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('TTC (s)', fontsize=12)
    ax.set_title('各起始车道平均 TTC', fontsize=14)
    ax.legend(fontsize=9)
    # ax.grid(True, alpha=0.3)

    # 3d) Velocity 带标准差
    ax = axes[1, 0]
    all_pivot_v = df_lanes.pivot_table(index='FrameIdx', columns='ID', values='Velocity_ms', aggfunc='first')
    mean_all_v = all_pivot_v.mean(axis=1)
    std_all_v = all_pivot_v.std(axis=1)
    ax.plot(all_pivot_v.index, mean_all_v, color='blue', linewidth=2.5, label='均值')
    ax.fill_between(all_pivot_v.index, mean_all_v - std_all_v, mean_all_v + std_all_v, alpha=0.2, color='blue')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('速度 (m/s)', fontsize=12)
    ax.set_title(f'全部 {n_vehicles} 辆平均速度 (±1σ)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)

    # 3e) TTC 带标准差
    ax = axes[1, 1]
    if 'TTC' in df_lanes.columns:
        all_pivot_ttc = df_lanes.pivot_table(index='FrameIdx', columns='ID', values='TTC', aggfunc='first')
        mean_all_ttc = all_pivot_ttc.mean(axis=1)
        std_all_ttc = all_pivot_ttc.std(axis=1)
        ax.plot(all_pivot_ttc.index, mean_all_ttc, color='red', linewidth=2.5, label='均值')
        ax.fill_between(all_pivot_ttc.index, mean_all_ttc - std_all_ttc, mean_all_ttc + std_all_ttc, alpha=0.2, color='red')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('TTC (s)', fontsize=12)
    ax.set_title(f'全部 {n_vehicles} 辆平均 TTC (±1σ)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)

    # 3f) mTTC 带标准差
    ax = axes[1, 2]
    if 'mTTC' in df_lanes.columns:
        all_pivot_mttc = df_lanes.pivot_table(index='FrameIdx', columns='ID', values='mTTC', aggfunc='first')
        mean_all_mttc = all_pivot_mttc.mean(axis=1)
        std_all_mttc = all_pivot_mttc.std(axis=1)
        ax.plot(all_pivot_mttc.index, mean_all_mttc, color='orange', linewidth=2.5, label='均值')
        ax.fill_between(all_pivot_mttc.index, mean_all_mttc - std_all_mttc, mean_all_mttc + std_all_mttc, alpha=0.2, color='orange')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('mTTC (s)', fontsize=12)
    ax.set_title(f'全部 {n_vehicles} 辆平均 mTTC (±1σ)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)

    plt.suptitle(f'{source} {behavior_label}车辆轨迹组分析', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_group_analysis.png'), dpi=150)
    plt.close()
    print("  ✓ 图3: 车道分组分析")

    # ==================== 图4. 速度与TTC分布带 ====================
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    ax = axes[0]
    vel_matrix = df.pivot_table(index='FrameIdx', columns='ID', values='Velocity_ms', aggfunc='first')
    x_idx = vel_matrix.index
    p10, p25, p50, p75, p90 = [vel_matrix.quantile(q, axis=1) for q in [0.1, 0.25, 0.5, 0.75, 0.9]]
    ax.fill_between(x_idx, p10, p90, alpha=0.15, color='blue', label='P10-P90')
    ax.fill_between(x_idx, p25, p75, alpha=0.25, color='blue', label='P25-P75')
    ax.plot(x_idx, p50, color='darkblue', linewidth=2.5, label='中位数(P50)')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('速度 (m/s)', fontsize=12)
    ax.set_title(f'速度分布 ({n_vehicles}辆)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'TTC' in df.columns:
        ttc_matrix = df.pivot_table(index='FrameIdx', columns='ID', values='TTC', aggfunc='first')
        x_idx_t = ttc_matrix.index
        p10_t, p25_t, p50_t, p75_t, p90_t = [ttc_matrix.quantile(q, axis=1) for q in [0.1, 0.25, 0.5, 0.75, 0.9]]
        ax.fill_between(x_idx_t, p10_t, p90_t, alpha=0.15, color='red', label='P10-P90')
        ax.fill_between(x_idx_t, p25_t, p75_t, alpha=0.25, color='red', label='P25-P75')
        ax.plot(x_idx_t, p50_t, color='darkred', linewidth=2.5, label='中位数(P50)')
    ax.axhline(y=2, color='orange', linestyle='--', linewidth=1.5, label='危险阈值 (TTC=2s)')
    ax.set_xlabel('帧序号', fontsize=12)
    ax.set_ylabel('TTC (s)', fontsize=12)
    ax.set_title(f'TTC 分布 ({n_vehicles}辆)', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)

    plt.suptitle(f'{source} {behavior_label} - 速度与 TTC 分布带', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_distribution_bands.png'), dpi=150)
    plt.close()
    print("  ✓ 图4: 分布带图")


def main():
    # ---- location1 路段 × 左右变道 ----
    tasks = [
        ('1-1', 'left',  '左变道'),
        ('1-1', 'right', '右变道'),
        ('1-2', 'left',  '左变道'),
        ('1-2', 'right', '右变道'),
    ]

    for source, suffix, behavior in tasks:
        filepath = os.path.join(input_dir, f'traffic_{source}_{suffix}_change.csv')
        if not os.path.exists(filepath):
            print(f"[!] 文件不存在, 跳过: {filepath}")
            continue
        print(f"\n{'='*50}")
        print(f"处理 {source} {behavior} 数据: {filepath}")
        print(f"{'='*50}")
        df = pd.read_csv(filepath)
        output_dir = os.path.join(output_base, f'{source}_{behavior}_trajectory_vis')
        process_behavior(df, behavior, source, output_dir)

    # ---- location5 左右变道 ----
    loc5_tasks = [
        ('loc5', 'left',  '左变道'),
        ('loc5', 'right', '右变道'),
    ]
    for source, suffix, behavior in loc5_tasks:
        filepath = os.path.join(loc5_dir, f'traffic_{suffix}_change.csv')
        if not os.path.exists(filepath):
            print(f"[!] 文件不存在, 跳过: {filepath}")
            continue
        print(f"\n{'='*50}")
        print(f"处理 {source} {behavior} 数据: {filepath}")
        print(f"{'='*50}")
        df = pd.read_csv(filepath)
        output_dir = os.path.join(output_base, f'{source}_{behavior}_trajectory_vis')
        process_behavior(df, behavior, source, output_dir)

    print(f"\n✅ 全部图片已保存至: {output_base}")


if __name__ == "__main__":
    main()
