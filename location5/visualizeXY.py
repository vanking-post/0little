import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
#save_dir = r"E:\0little\read\CQSkyEyedata5\location5t"

def visualize_trajectory_samples(df_east_smooth, df_west_smooth,
                                 df_east_sampling, df_west_sampling,save_dir = None,
                                 random_seed=42):
    """
    可视化样本轨迹与完整轨迹的对比图

    参数:
        df_east_smooth, df_west_smooth: 东西向平滑后的完整轨迹数据（包含X,Y,ID,Frame,Label等）
        df_east_sampling, df_west_sampling: 东西向截取的样本数据（包含ID,Frame,Label等）
        random_seed: 随机种子，用于保证可重复性（默认为42）

    输出:
        显示三张图（跟驰、左变道、右变道），每张图包含10个子图（5行2列），
        蓝色点为该车辆的所有轨迹点，红色点为样本片段对应的轨迹点。
    """
    # 设置中文字体和随机种子
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 1. 合并完整轨迹数据
    full_data = pd.concat([df_east_smooth, df_west_smooth], ignore_index=True)
    print(f"完整轨迹数据合并后形状: {full_data.shape}")

    # 2. 合并样本数据
    sample_data = pd.concat([df_east_sampling, df_west_sampling], ignore_index=True)
    print(f"样本数据合并后形状: {sample_data.shape}")

    # 检查必要列是否存在
    required_cols_full = ['ID', 'X', 'Y']
    required_cols_sample = ['ID', 'Frame', 'Label']
    for col in required_cols_full:
        if col not in full_data.columns:
            raise ValueError(f"完整数据中缺少列: {col}")
    for col in required_cols_sample:
        if col not in sample_data.columns:
            raise ValueError(f"样本数据中缺少列: {col}")

    # 3. 获取每个标签下的所有车辆ID（从样本数据中提取去重）
    # 假设样本数据中的标签列名为 'Label'，值为 '跟驰', '左变道', '右变道'
    # 如果实际标签是英文，可以调整下面的映射
    label_map = {
        '跟驰': 'following',
        '左变道': 'left',
        '右变道': 'right'
    }
    # 找出实际存在的标签（可能为中文或英文）
    unique_labels = sample_data['Label'].unique()
    print(f"样本中的标签: {unique_labels}")

    # 确定三个标签的字符串（兼容中英文）
    following_label = None
    left_label = None
    right_label = None
    for lbl in unique_labels:
        lbl_lower = str(lbl).lower()
        if '跟' in lbl_lower or 'follow' in lbl_lower:
            following_label = lbl
        elif '左' in lbl_lower or 'left' in lbl_lower:
            left_label = lbl
        elif '右' in lbl_lower or 'right' in lbl_lower:
            right_label = lbl

    # 辅助函数：从样本中获取指定标签的车辆ID列表
    def get_vehicle_ids(label_value):
        if label_value is None:
            return []
        ids = sample_data[sample_data['Label'] == label_value]['ID'].unique()
        return ids.tolist()

    following_ids = get_vehicle_ids(following_label)
    left_ids = get_vehicle_ids(left_label)
    right_ids = get_vehicle_ids(right_label)

    print(f"跟驰车辆数: {len(following_ids)}")
    print(f"左变道车辆数: {len(left_ids)}")
    print(f"右变道车辆数: {len(right_ids)}")

    # 辅助函数：为每个标签绘制一张大图
    def plot_for_label(vehicle_ids, label_name, title_prefix,save_dir):
        if not vehicle_ids:
            print(f"没有找到{label_name}的车辆，跳过绘图")
            return

        # 随机选择最多10辆车（不足10则全选）
        n_select = min(10, len(vehicle_ids))
        selected_ids = random.sample(vehicle_ids, n_select)

        # 创建5行2列的子图
        fig, axes = plt.subplots(5, 2, figsize=(19.2, 10.8))
        axes = axes.flatten()

        for idx, vid in enumerate(selected_ids):
            ax = axes[idx]

            # 获取该车辆的所有轨迹点（从full_data）
            vehicle_full = full_data[full_data['ID'] == vid][['X', 'Y']].dropna()
            if vehicle_full.empty:
                ax.text(0.5, 0.5, f'ID {vid} 无轨迹数据', ha='center', va='center')
                ax.set_title(f'{label_name} ID: {vid}')
                continue

            # 获取该车辆在样本中的Frame列表
            sample_frames = sample_data[(sample_data['ID'] == vid)]['Frame'].unique()
            # 从full_data中提取这些Frame对应的点
            vehicle_sample = full_data[(full_data['ID'] == vid) &
                                       (full_data['Frame'].isin(sample_frames))][['X', 'Y']].dropna()

            # 确定车道方向和车道线位置
            y_min, y_max = vehicle_full['Y'].min(), vehicle_full['Y'].max()
            if y_min >= 0 and y_max <= 20:
                # 东向西行驶方向 (Y值0-20)
                y_range = (20, 0)  # 倒置Y轴
                lane_lines = [6.5, 10, 13.75]
                # 添加行驶方向注释
                ax.annotate('行驶方向为从东向西',
                            xy=(0.15, 0.1), xytext=(0.5, 0.1),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                            shrink=0.05, alpha=0.8),
                            ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)
            else:
                # 西向东行驶方向 (Y值20-40) 或 其他范围
                y_range = (40, 20)  # 倒置Y轴
                lane_lines = [23.5, 27.25, 31]
                ax.annotate('行驶方向为从西向东',
                            xy=(0.85, 0.1), xytext=(0.5, 0.1),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                            shrink=0.05, alpha=0.8),
                            ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)

            # 绘制所有轨迹点（蓝色）
            ax.scatter(vehicle_full['X'], vehicle_full['Y'],
                       c='blue', s=1, alpha=0.6, label='完整轨迹')
            # 绘制样本片段点（红色）
            if not vehicle_sample.empty:
                ax.scatter(vehicle_sample['X'], vehicle_sample['Y'],
                           c='red', s=2, alpha=0.8, label='样本片段')

            # 绘制车道线
            x_limits = [0, 340]
            for ly in lane_lines:
                ax.plot(x_limits, [ly, ly], color='black', linewidth=0.8, linestyle='--', alpha=0.6)

            ax.set_xlim(0, 340)
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_title(f'{title_prefix} ID: {vid}')
            ax.set_xlabel('X 坐标 (m)')
            ax.set_ylabel('Y 坐标 (m)')
            ax.legend(loc='upper right', fontsize=8)

        # 隐藏多余的子图
        for idx in range(len(selected_ids), 10):
            axes[idx].axis('off')

        name_map = {
            '跟驰': 'following_trajectory.png',
            '左变道': 'left_lanechange_trajectory.png',
            '右变道': 'right_lanechange_trajectory.png'
        }
        filename = name_map.get(label_name, f"{label_name}_trajectory.png")
        save_path = os.path.join(save_dir, filename) if save_dir else None
        plt.suptitle(f'{title_prefix} 轨迹对比 (蓝色: 完整轨迹, 红色: 样本片段)', fontsize=16)
        plt.tight_layout()
        #plt.show(block=Flase)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        plt.close()

    # 为三个标签分别绘图
    plot_for_label(following_ids, "跟驰", "跟驰车辆",save_dir)
    plot_for_label(left_ids, "左变道", "左变道车辆",save_dir)
    plot_for_label(right_ids, "右变道", "右变道车辆",save_dir)

# 使用示例（假设已经加载了四个DataFrame）：
# visualize_trajectory_samples(df_east_smooth, df_west_smooth,
#                              df_east_sampling, df_west_sampling)