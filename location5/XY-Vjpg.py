import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import os

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
random.seed(42)
np.random.seed(42)

# ==================== 路径 ====================
data_dir = r"E:\0little\read\CQSkyEyedata5\location5t"
smooth_file = os.path.join(data_dir, "traffic_flows_east_smooth.csv")
sampling_file = os.path.join(data_dir, "traffic_flows_east_sampling.csv")
output_dir = data_dir

# ==================== 加载数据 ====================
print("加载数据...")
df_smooth = pd.read_csv(smooth_file)
df_sampling = pd.read_csv(sampling_file)

# 确保有 Velocity 列（大小写兼容）
vel_col = None
for c in df_smooth.columns:
    if c.lower() == 'velocity':
        vel_col = c
        break
if vel_col is None:
    raise KeyError(f"df_smooth 中没有 Velocity 列，现有列: {df_smooth.columns.tolist()}")
if vel_col != 'Velocity':
    df_smooth['Velocity'] = df_smooth[vel_col]

print(f"smooth 数据: {df_smooth.shape}, 列: {df_smooth.columns.tolist()}")
print(f"sampling 数据: {df_sampling.shape}, 列: {df_sampling.columns.tolist()}")

# ==================== 提取左变道车辆 ====================
if '左变道' in df_sampling['Label'].values:
    LEFT_LABEL = '左变道'
elif 'left' in df_sampling['Label'].values:
    LEFT_LABEL = 'left'
else:
    LEFT_LABEL = df_sampling['Label'].unique()[0]

left_ids_all = sorted(df_sampling[df_sampling['Label'] == LEFT_LABEL]['ID'].unique())
n_left = len(left_ids_all)
print(f"左变道车辆: {n_left} 辆, 标签='{LEFT_LABEL}'")

# ==================== 逐个生成5张3D图并显示 ====================
N = 5
if n_left < N:
    print(f"车辆不足 {N}，实际只生成 {n_left} 张")
    N = n_left

chosen = random.sample(left_ids_all, N)
print(f"随机选择ID: {[int(x) for x in chosen]}\n")

for i, vid in enumerate(chosen):
    print(f">>> 图 {i+1}/{N} — ID = {int(vid)}")

    # ---- 完整轨迹 ----
    df_v = df_smooth[df_smooth['ID'] == vid] \
        .dropna(subset=['X', 'Y', 'Velocity']) \
        .sort_values('Frame')

    if df_v.empty:
        print("    ⚠ 无完整轨迹数据，跳过")
        continue

    # ---- 左变道样本片段 ----
    samp_frames = df_sampling[(df_sampling['ID'] == vid)
                              & (df_sampling['Label'] == LEFT_LABEL)]['Frame'].unique()
    df_vs = df_v[df_v['Frame'].isin(samp_frames)]

    # ---- 创建单张 3D 图 ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 蓝色 = 完整轨迹（散点 + 连线）
    ax.scatter(df_v['X'], df_v['Y'], df_v['Velocity'],
               c='blue', s=10, alpha=0.55, depthshade=True, label='全部轨迹')
    ax.plot(df_v['X'], df_v['Y'], df_v['Velocity'],
            color='gray', linewidth=0.6, alpha=0.35)

    # 红色 = 左变道样本
    if not df_vs.empty:
        ax.scatter(df_vs['X'], df_vs['Y'], df_vs['Velocity'],
                   c='red', s=40, alpha=0.95, edgecolors='darkred',
                   linewidths=0.6, depthshade=False, label='左变道样本')

    # 起点 / 终点
    ax.scatter([df_v['X'].iloc[0]], [df_v['Y'].iloc[0]], [df_v['Velocity'].iloc[0]],
               c='green', s=120, marker='o', edgecolors='black', label='起点')
    ax.scatter([df_v['X'].iloc[-1]], [df_v['Y'].iloc[-1]], [df_v['Velocity'].iloc[-1]],
               c='orange', s=120, marker='D', edgecolors='black', label='终点')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('速度 (m/s)')
    ax.set_title(f'左变道车辆 ID: {int(vid)} — 3D轨迹 (X-Y-速度)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # ---- 保存 ----
    out_path = os.path.join(output_dir, f'left_3d_{int(vid)}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    已保存: {out_path}")

    # ---- 显示到屏幕（关闭当前窗口后自动弹出下一张） ----
    plt.show(block=True)
    plt.close()

print(f"\n✅ 3D 图完成！共生成 {N} 张。")

# ============================================================
# ===== 补充：对同一批5辆车生成二维 X-Y 图（蓝=全部，红=样本）=====
# ============================================================
print(f"\n{'='*50}")
print("开始生成同一批车辆的二维 X-Y 轨迹图 ...")

for i, vid in enumerate(chosen):
    print(f">>> 2D图 {i+1}/{N} — ID = {int(vid)}")

    # 完整轨迹
    df_v = df_smooth[df_smooth['ID'] == vid] \
        .dropna(subset=['X', 'Y']) \
        .sort_values('Frame')

    if df_v.empty:
        print("    ⚠ 无完整轨迹数据，跳过")
        continue

    # 左变道样本
    samp_frames = df_sampling[(df_sampling['ID'] == vid)
                              & (df_sampling['Label'] == LEFT_LABEL)]['Frame'].unique()
    df_vs = df_v[df_v['Frame'].isin(samp_frames)]

    # 判断车道线
    y_min, y_max = df_v['Y'].min(), df_v['Y'].max()
    if y_min >= 0 and y_max <= 20:
        y_range = (20, 0)
        lane_lines = [6.5, 10, 13.75]
        arrow_xy = (0.15, 0.1)
        arrow_text = '行驶方向: 从东向西'
    else:
        y_range = (40, 20)
        lane_lines = [23.5, 27.25, 31]
        arrow_xy = (0.85, 0.1)
        arrow_text = '行驶方向: 从西向东'

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 9))

    # 蓝色 = 完整轨迹
    ax.scatter(df_v['X'], df_v['Y'],
               c='blue', s=8, alpha=0.55, label='全部轨迹')
    # 红色 = 左变道样本
    if not df_vs.empty:
        ax.scatter(df_vs['X'], df_vs['Y'],
                   c='red', s=35, alpha=0.95, edgecolors='darkred',
                   linewidths=0.5, label='左变道样本')

    # 起点/终点
    ax.scatter(df_v['X'].iloc[0], df_v['Y'].iloc[0],
               c='green', s=120, marker='o', edgecolors='black', label='起点')
    ax.scatter(df_v['X'].iloc[-1], df_v['Y'].iloc[-1],
               c='orange', s=120, marker='D', edgecolors='black', label='终点')

    # 车道线
    for ly in lane_lines:
        ax.plot([0, 340], [ly, ly], color='black', linewidth=0.8,
                linestyle='--', alpha=0.6)

    # 方向箭头
    ax.annotate(arrow_text,
                xy=arrow_xy, xytext=(0.5, 0.1),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='green', edgecolor='gray',
                                width=1.5, headwidth=7, shrink=0.05, alpha=0.8),
                ha='center', va='center', fontsize=11, color='black', alpha=0.8, zorder=5)

    ax.set_xlim(0, 340)
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'左变道车辆 ID: {int(vid)} — 2D轨迹 (X-Y)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 保存
    out_path = os.path.join(output_dir, f'left_2d_{int(vid)}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    已保存: {out_path}")

    # 显示
    plt.show(block=True)
    plt.close()

print(f"\n✅ 全部完成！3D图 {N} 张 + 2D图 {N} 张已生成。")
