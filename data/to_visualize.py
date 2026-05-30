import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# ===============================
# 坐标范围控制参数
# ===============================
NORMALIZED_X_MIN, NORMALIZED_X_MAX = -2, 2
NORMALIZED_Y_MIN, NORMALIZED_Y_MAX = 2, -2



REFERENCE_LINES = [6.5, 10,13.75, 17.5]
SMOOTH_X_MIN, SMOOTH_X_MAX = 0, 325
SMOOTH_Y_MIN, SMOOTH_Y_MAX = 20, 0
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
files = [
    ("Raw", "traffic_flows_east.pkl"),
    ("Processed", "traffic_flows_east_processed.pkl"),
    ("Complete", "traffic_flows_complete0.pkl"),
    ("Smooth", "traffic_flows_smooth.pkl"),
    ("Normalized", "traffic_flows_guiyi.pkl"),]

# 数据路径
# ===============================
# REFERENCE_LINES = [23.5, 27.25, 31]
# SMOOTH_X_MIN, SMOOTH_X_MAX = 0, 325
# SMOOTH_Y_MIN, SMOOTH_Y_MAX = 40, 20
# save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
# files = [
#     ("Raw", "traffic_flows_west.pkl"),
#     ("Processed", "traffic_flows_west_processed.pkl"),
#     ("Complete", "traffic_flows_complete0.pkl"),
#     ("Smooth", "traffic_flows_smooth.pkl"),
#     ("Normalized", "traffic_flows_guiyi.pkl"),]

# ===============================
# 创建 2x3 大图（1920x1080）
# ===============================
fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8))
axes = axes.flatten()
# ===============================
# 逐个数据集绘图
# ===============================
for i, (name, fname) in enumerate(files):
    ax = axes[i]
    path = os.path.join(save_dir, fname)

    df = pd.read_pickle(path)

    if "Velocity" not in df.columns:
        ax.set_title(f"{name}\nVelocity not found")
        ax.axis("off")
        continue

    v = df["Velocity"].dropna().values

    if len(v) < 10:
        ax.set_title(f"{name}\nInsufficient data")
        ax.axis("off")
        continue

    # 高斯分布拟合
    mu, sigma = stats.norm.fit(v)

    x = np.linspace(v.min(), v.max(), 500)
    y_gauss = stats.norm.pdf(x, mu, sigma)

    # 密度直方图（蓝色实心）
    ax.hist(
        v,
        bins=50,
        density=True,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Histogram (density)"
    )

    # 高斯分布曲线（红色）
    ax.plot(
        x,
        y_gauss,
        color="red",
        linewidth=2,
        label="Gaussian fit"
    )

    ax.set_title(
        f"{name}\nμ = {mu:.4f}, σ = {sigma:.4f}",
        fontsize=11
    )
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True)

# ===============================
# 隐藏多余的第 6 个子图
# ===============================
for j in range(len(files), len(axes)):
    axes[j].axis("off")

# ===============================
# 布局调整 & 显示
# ===============================
plt.suptitle("Velocity Distribution & Gaussian Fit (All Datasets)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ===============================
# 车辆轨迹可视化（5行2列，1920x1080）
# ===============================
# 读取平滑和归一化后的数据
smooth_path = os.path.join(save_dir, "traffic_flows_smooth.pkl")
normalized_path = os.path.join(save_dir, "traffic_flows_guiyi.pkl")

df_smooth = pd.read_pickle(smooth_path)
df_normalized = pd.read_pickle(normalized_path)

# 确保X和Y列存在
if "X" not in df_smooth.columns or "Y" not in df_smooth.columns:
    print("平滑数据中缺少X或Y列")
else:
    # 获取车辆ID列表
    smooth_vehicle_ids = df_smooth['ID'].unique()
    normalized_vehicle_ids = df_normalized['ID'].unique()

    # 随机选择5个车辆ID
    selected_ids = np.random.choice(smooth_vehicle_ids, size=min(5, len(smooth_vehicle_ids)), replace=False)

    # 创建新的大图用于显示车辆轨迹（5行2列）
    fig_traj, axes_traj = plt.subplots(5, 2, figsize=(19.2, 10.8))

    for idx, vehicle_id in enumerate(selected_ids):
        # 左边列：平滑数据
        ax_left = axes_traj[idx, 0]
        vehicle_data_smooth = df_smooth[df_smooth['ID'] == vehicle_id]

        if not vehicle_data_smooth.empty:
            ax_left.scatter(vehicle_data_smooth['X'], vehicle_data_smooth['Y'],
                            s=1, alpha=0.6, c='blue', label=f'Vehicle {vehicle_id}')
            ax_left.set_xlim(SMOOTH_X_MIN, SMOOTH_X_MAX)
            ax_left.set_ylim(SMOOTH_Y_MIN, SMOOTH_Y_MAX)

            # 添加参考线
            for y_pos in REFERENCE_LINES:
                ax_left.axhline(y=y_pos, color='green', linestyle='--', linewidth=1, alpha=0.7)

            ax_left.set_title(f'Smoothed - ID: {vehicle_id}', fontsize=10)
            ax_left.set_xlabel('X')
            ax_left.set_ylabel('Y')
            ax_left.grid(True)
            ax_left.legend()
        else:
            ax_left.text(0.5, 0.5, f'No data for ID: {vehicle_id}',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_left.transAxes)
            ax_left.set_xlim(SMOOTH_X_MIN, SMOOTH_X_MAX)
            ax_left.set_ylim(SMOOTH_Y_MIN, SMOOTH_Y_MAX)

            # 添加参考线
            for y_pos in REFERENCE_LINES:
                ax_left.axhline(y=y_pos, color='green', linestyle='--', linewidth=1, alpha=0.7)

            ax_left.set_title(f'Smoothed - ID: {vehicle_id}', fontsize=10)

        # 右边列：归一化数据
        ax_right = axes_traj[idx, 1]
        vehicle_data_normalized = df_normalized[df_normalized['ID'] == vehicle_id]

        if not vehicle_data_normalized.empty:
            ax_right.scatter(vehicle_data_normalized['X'], vehicle_data_normalized['Y'],
                             s=1, alpha=0.6, c='red', label=f'Vehicle {vehicle_id}')
            ax_right.set_xlim(NORMALIZED_X_MIN, NORMALIZED_X_MAX)
            ax_right.set_ylim(NORMALIZED_Y_MIN, NORMALIZED_Y_MAX)
            ax_right.set_title(f'Normalized - ID: {vehicle_id}', fontsize=10)
            ax_right.set_xlabel('X')
            ax_right.set_ylabel('Y')
            ax_right.grid(True)
            ax_right.legend()
        else:
            ax_right.text(0.5, 0.5, f'No data for ID: {vehicle_id}',
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_right.transAxes)
            ax_right.set_xlim(NORMALIZED_X_MIN, NORMALIZED_X_MAX)
            ax_right.set_ylim(NORMALIZED_Y_MIN, NORMALIZED_Y_MAX)
            ax_right.set_title(f'Normalized - ID: {vehicle_id}', fontsize=10)

    # 隐藏多余的行（如果有）
    remaining_rows = 5 - len(selected_ids)
    if remaining_rows > 0:
        for i in range(len(selected_ids), 5):
            axes_traj[i, 0].axis('off')
            axes_traj[i, 1].axis('off')

    plt.suptitle('Vehicle Trajectories Comparison: Smoothed vs Normalized Data', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print(f"随机选择了以下车辆ID进行轨迹对比: {selected_ids}")
    print(f"平滑数据中的车辆总数: {len(smooth_vehicle_ids)}")
    print(f"归一化数据中的车辆总数: {len(normalized_vehicle_ids)}")
    print(f"平滑数据X轴范围: [{SMOOTH_X_MIN}, {SMOOTH_X_MAX}]")
    print(f"平滑数据Y轴范围: [{SMOOTH_Y_MIN}, {SMOOTH_Y_MAX}]")
    print(f"归一化数据X轴范围: [{NORMALIZED_X_MIN}, {NORMALIZED_X_MAX}]")
    print(f"归一化数据Y轴范围: [{NORMALIZED_Y_MIN}, {NORMALIZED_Y_MAX}]")
    print(f"平滑数据参考线位置: Y={REFERENCE_LINES}")
