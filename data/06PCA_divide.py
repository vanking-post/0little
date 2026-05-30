#同时读取东向和西向的交通流样本数据进行合并，将合并后的变道数据进行均衡取样
# 并使用PCA降维，保持数据完整度大于0.95，之后将合并降维的数据输出traffic_flow_pca_result
#读取traffic_flows_sampling.pkl和traffic_flows_smooth.pkl，
# 输出样本分割数据集traffic_flow_train和traffic_flow_val.csv
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载路径
save_dir_e = r"E:\0little\read\CQSkyEyedata5\location5e"
save_dir_w = r"E:\0little\read\CQSkyEyedata5\location5"
save_dir = r"E:\0little\read\CQSkyEyedata5"
files = [
    ("save_dir_e", "traffic_flows_sampling.pkl"),
    ("save_dir_w", "traffic_flows_sampling.pkl"),
    ("save_dir_e", "traffic_flows_smooth.pkl"),
    ("save_dir_w", "traffic_flows_smooth.pkl"),
]


def load_and_merge_data(file_configs):
    """加载并合并数据"""
    dfs = []
    for path_var, filename in file_configs:
        path = save_dir_e if path_var.endswith('_e') else save_dir_w
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)
            df['direction'] = 'east' if path_var.endswith('_e') else 'west'
            dfs.append(df)
            print(f"Loaded: {filepath}, Shape: {df.shape}")

    # 额外加载两个平滑后数据库
    smooth_files = [
        os.path.join(save_dir_e, "traffic_flows_smooth.pkl"),
        os.path.join(save_dir_w, "traffic_flows_smooth.pkl")
    ]

    for smooth_file in smooth_files:
        if os.path.exists(smooth_file):
            df_smooth = pd.read_pickle(smooth_file)
            df_smooth['direction'] = 'smooth_east' if 'location5e' in smooth_file else 'smooth_west'
            dfs.append(df_smooth)
            print(f"Loaded smooth data: {smooth_file}, Shape: {df_smooth.shape}")

    merged_data = pd.concat(dfs, ignore_index=True) if dfs else None

    # 保存合并后的原始数据
    merged_output_path = os.path.join(save_dir, "merged_traffic_data.pkl")
    merged_data.to_pickle(merged_output_path)
    merged_csv_path = os.path.join(save_dir, "merged_traffic_data.csv")
    merged_data.to_csv(merged_csv_path, index=False)
    print(f"合并后的数据已保存至: {merged_output_path} 和 {merged_csv_path}")

    return merged_data


# 加载数据
data = load_and_merge_data(files)
print(f"Merged data shape: {data.shape}")

# 检查新增字段
additional_columns = ['Frame', 'ID']
for col in additional_columns:
    if col in data.columns:
        print(f"Found additional column: {col}")
    else:
        print(f"Warning: Missing additional column: {col}")

# 检查标签列
label_col = 'Label' if 'Label' in data.columns else 'label'
if label_col in data.columns:
    print(f"Found label column: {label_col}")
    print(f"Label distribution:\n{data[label_col].value_counts()}")
else:
    print("No label column found")
    label_col = None

# 指定需要进行降维的16个特征
features_for_pca = [
    'Velocity', 'Acceleration', 'lat_Vel',
    'lat_Acc', 'long_Vel', 'long_Acc', 'Following_dist', 'Time_Headway',
    'TTC', 'LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist',
    'RB_Dist', 'RS_Dist', 'RF_Dist'
]

# 检查特征是否存在于数据中
available_features = [f for f in features_for_pca if f in data.columns]
print(f"Available features for PCA: {available_features}")
print(f"Number of features for PCA: {len(available_features)}")

if len(available_features) != 16:
    print(f"Warning: Expected 16 features, but found {len(available_features)}")
    print(f"Missing features: {[f for f in features_for_pca if f not in data.columns]}")

# 标准化所有16个特征
scaler = StandardScaler()
data[available_features] = scaler.fit_transform(data[available_features])
print(f"Standardized all {len(available_features)} features for PCA")

# 保留额外的列用于后续样本分割验证
extra_columns = [col for col in additional_columns if col in data.columns]
print(f"Extra columns for sample validation: {extra_columns}")

# 按ID和标签排序数据
if 'ID' in data.columns:
    sort_columns = ['ID']
    if label_col and label_col in data.columns:
        sort_columns = [label_col] + sort_columns
    data = data.sort_values(sort_columns).reset_index(drop=True)
    print(f"数据已按{sort_columns}排序")

# PCA降维
pca = PCA(n_components=0.95)  # 保留95%方差
X_pca = pca.fit_transform(data[available_features])

# 创建结果DataFrame，包含PCA结果、标签和其他必要列
pca_df = pd.DataFrame(X_pca,
                      columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])])
# 保留标签列
if label_col and label_col in data.columns:
    pca_df[label_col] = data[label_col]
# 保留额外列用于样本验证
for col in extra_columns:
    if col in data.columns:
        pca_df[col] = data[col]

print(f"Original dimensions: {len(available_features)}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Reduction: {len(available_features)}D -> {X_pca.shape[1]}D")
print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}...")  # 显示前5个
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[:]}...")  # 显示前5个

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 方差解释图
n_show = min(11, len(pca.explained_variance_ratio_))
axes[0].bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show])
axes[0].set_title('Variance Explained by Each PC')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Ratio')

# 累计方差图
cumvar = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
axes[1].set_title('Cumulative Variance Explained')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Ratio')
axes[1].legend()

plt.tight_layout()
plt.show()

# 保存PCA降维后的数据
pca_output_path = os.path.join(save_dir, "traffic_flow_pca_result.pkl")
pca_df.to_pickle(pca_output_path)
pca_csv_path = os.path.join(save_dir, "traffic_flow_pca_result.csv")
pca_df.to_csv(pca_csv_path, index=False)

print(f"PCA降维后的结果保存至: {pca_output_path} 和 {pca_csv_path}")
print(f"PCA后数据形状: {pca_df.shape}")

# =========================
# 获取不同标签下的车辆ID
# =========================

print("\n" + "=" * 70)
print("获取不同标签下的车辆ID")
print("=" * 70)

if label_col and label_col in pca_df.columns:
    # 获取唯一的标签值
    unique_labels = pca_df[label_col].unique()
    print(f"唯一标签值: {unique_labels}")

    # 定义三个特定标签
    left_lane_label = None
    right_lane_label = None
    following_label = None

    for label in unique_labels:
        label_lower = str(label).lower()
        if 'left' in label_lower or '左' in label_lower:
            left_lane_label = label
        elif 'right' in label_lower or '右' in label_lower:
            right_lane_label = label
        elif 'follow' in label_lower or '跟' in label_lower:
            following_label = label

    print(f"识别出的标签: 左变道={left_lane_label}, 右变道={right_lane_label}, 跟驰={following_label}")

    # 获取每个标签下的车辆ID
    left_lane_ids = []
    right_lane_ids = []
    following_ids = []

    if left_lane_label is not None:
        left_lane_ids = pca_df[pca_df[label_col] == left_lane_label]['ID'].unique().tolist()
        print(f"\n左变道标签({left_lane_label})下的车辆ID数量: {len(left_lane_ids)}")
        print("左变道车辆ID (每行10个):")
        for i in range(0, len(left_lane_ids), 10):
            print("  ", left_lane_ids[i:i + 10])

    if right_lane_label is not None:
        right_lane_ids = pca_df[pca_df[label_col] == right_lane_label]['ID'].unique().tolist()
        print(f"\n右变道标签({right_lane_label})下的车辆ID数量: {len(right_lane_ids)}")
        print("右变道车辆ID (每行10个):")
        for i in range(0, len(right_lane_ids), 10):
            print("  ", right_lane_ids[i:i + 10])

    if following_label is not None:
        following_ids = pca_df[pca_df[label_col] == following_label]['ID'].unique().tolist()
        print(f"\n跟驰标签({following_label})下的车辆ID数量: {len(following_ids)}")
        print("跟驰车辆ID (每行20个):")
        for i in range(0, len(following_ids), 20):
            print("  ", following_ids[i:i + 20])

    # 进行8:2随机分割
    print(f"\n开始8:2随机分割...")

    # 分割左变道ID
    left_train_ids, left_val_ids = [], []
    if left_lane_ids:
        left_train_ids, left_val_ids = train_test_split(left_lane_ids, test_size=0.2, random_state=42)
        print(f"左变道: 训练集 {len(left_train_ids)} 个ID, 验证集 {len(left_val_ids)} 个ID")

    # 分割右变道ID
    right_train_ids, right_val_ids = [], []
    if right_lane_ids:
        right_train_ids, right_val_ids = train_test_split(right_lane_ids, test_size=0.2, random_state=42)
        print(f"右变道: 训练集 {len(right_train_ids)} 个ID, 验证集 {len(right_val_ids)} 个ID")

    # 计算左右变道的平均值（目标数量）
    target_train_size = int((len(left_train_ids) + len(right_train_ids)) / 2)
    target_val_size = int((len(left_val_ids) + len(right_val_ids)) / 2)

    # 分割跟驰ID，并按平均值采样，防止类别不平衡
    following_train_ids, following_val_ids = [], []
    if following_ids:
        # 先按 8:2 正常分割
        full_follow_train, full_follow_val = train_test_split(following_ids, test_size=0.2, random_state=42)

        # 再按目标数量随机采样（不超过自身总量）
        following_train_ids = np.random.choice(
            full_follow_train, size=min(target_train_size, len(full_follow_train)), replace=False
        ).tolist()

        following_val_ids = np.random.choice(
            full_follow_val, size=min(target_val_size, len(full_follow_val)), replace=False
        ).tolist()

        print(f"跟驰均衡后: 训练集 {len(following_train_ids)} 个ID, 验证集 {len(following_val_ids)} 个ID")
        print(f"→ 目标数量：训练集≈{target_train_size}，验证集≈{target_val_size}")

    # 生成训练和验证数据集
    print(f"\n开始生成训练和验证数据集...")

    # 分别构建训练集
    train_dfs = []
    if left_train_ids:
        left_train_data = pca_df[(pca_df[label_col] == left_lane_label) & (pca_df['ID'].isin(left_train_ids))]
        train_dfs.append(left_train_data)
        print(f"左变道训练数据: {len(left_train_data)} 行")
    if right_train_ids:
        right_train_data = pca_df[(pca_df[label_col] == right_lane_label) & (pca_df['ID'].isin(right_train_ids))]
        train_dfs.append(right_train_data)
        print(f"右变道训练数据: {len(right_train_data)} 行")
    if following_train_ids:
        following_train_data = pca_df[(pca_df[label_col] == following_label) & (pca_df['ID'].isin(following_train_ids))]
        train_dfs.append(following_train_data)
        print(f"跟驰训练数据: {len(following_train_data)} 行")

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()

    # 分别构建验证集
    val_dfs = []
    if left_val_ids:
        left_val_data = pca_df[(pca_df[label_col] == left_lane_label) & (pca_df['ID'].isin(left_val_ids))]
        val_dfs.append(left_val_data)
        print(f"左变道验证数据: {len(left_val_data)} 行")
    if right_val_ids:
        right_val_data = pca_df[(pca_df[label_col] == right_lane_label) & (pca_df['ID'].isin(right_val_ids))]
        val_dfs.append(right_val_data)
        print(f"右变道验证数据: {len(right_val_data)} 行")
    if following_val_ids:
        following_val_data = pca_df[(pca_df[label_col] == following_label) & (pca_df['ID'].isin(following_val_ids))]
        val_dfs.append(following_val_data)
        print(f"跟驰验证数据: {len(following_val_data)} 行")

    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()

    print(f"训练集形状: {train_df.shape}")
    print(f"验证集形状: {val_df.shape}")

    # 保存训练集和验证集
    train_path = os.path.join(save_dir, "traffic_flow_train.csv")
    val_path = os.path.join(save_dir, "traffic_flow_val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # 保存ID列表
    id_lists_path = os.path.join(save_dir, "id_lists.pkl")
    id_lists = {
        'left_train_ids': left_train_ids,
        'left_val_ids': left_val_ids,
        'right_train_ids': right_train_ids,
        'right_val_ids': right_val_ids,
        'following_train_ids': following_train_ids,
        'following_val_ids': following_val_ids
    }
    pd.to_pickle(id_lists, id_lists_path)

    print(f"\nID列表保存至: {id_lists_path}")
    print(f"训练集保存至: {train_path}")
    print(f"验证集保存至: {val_path}")

    # 检查标签分布
    if label_col and label_col in train_df.columns:
        print(f"\n训练集标签分布:\n{train_df[label_col].value_counts()}")
        print(f"\n验证集标签分布:\n{val_df[label_col].value_counts()}")

    print(f"\n样本分割流程:")
    print(f"  1. 合并两个数据库并保存")
    print(f"  2. PCA降维")
    print(f"  3. 按标签获取车辆ID")
    print(f"  4. 8:2随机分割每类标签的ID")
    print(f"  5. 根据ID筛选原始数据构建训练集和验证集")
    print(f"  6. 保存训练集、验证集和ID列表")

    # 加载合并后的平滑数据用于可视化
    smooth_files = [
        os.path.join(save_dir_e, "traffic_flows_smooth.pkl"),
        os.path.join(save_dir_w, "traffic_flows_smooth.pkl")
    ]

    smooth_dfs = []
    for smooth_file in smooth_files:
        if os.path.exists(smooth_file):
            df_smooth = pd.read_pickle(smooth_file)
            smooth_dfs.append(df_smooth)

    if smooth_dfs: #可视化
        smooth_merged_df = pd.concat(smooth_dfs, ignore_index=True)

        # 从每个标签中随机选择10个车辆ID
        selected_following_ids = []
        selected_left_ids = []
        selected_right_ids = []

        if following_ids:
            selected_following_ids = np.random.choice(following_ids, min(10, len(following_ids)),
                                                      replace=False).tolist()
        if left_lane_ids:
            selected_left_ids = np.random.choice(left_lane_ids, min(10, len(left_lane_ids)), replace=False).tolist()
        if right_lane_ids:
            selected_right_ids = np.random.choice(right_lane_ids, min(10, len(right_lane_ids)), replace=False).tolist()

        print(f"选择的跟驰车辆ID: {selected_following_ids}")
        print(f"选择的左变道车辆ID: {selected_left_ids}")
        print(f"选择的右变道车辆ID: {selected_right_ids}")

        # 为每个标签创建可视化图
        figs = []

        # 跟驰标签可视化
        if selected_following_ids:
            fig, axes = plt.subplots(5, 2, figsize=(19.2, 10.8))
            axes = axes.flatten()

            for idx, vehicle_id in enumerate(selected_following_ids):
                if idx >= 10:
                    break

                # 获取该车辆的所有XY坐标
                all_coords = smooth_merged_df[smooth_merged_df['ID'] == vehicle_id][['X', 'Y']].dropna()

                # 获取该车辆在样本数据中的帧数信息
                sample_frames = train_df[train_df['ID'] == vehicle_id][
                    'Frame'].values if 'Frame' in train_df.columns else []
                if len(sample_frames) == 0:
                    sample_frames = val_df[val_df['ID'] == vehicle_id][
                        'Frame'].values if 'Frame' in val_df.columns else []

                # 获取样本中的XY坐标
                sample_coords = smooth_merged_df[
                    (smooth_merged_df['ID'] == vehicle_id) &
                    (smooth_merged_df['Frame'].isin(sample_frames))
                    ][['X', 'Y']].dropna()

                ax = axes[idx]
                # 根据Y值确定Y轴范围
                if not all_coords.empty:
                    y_min = all_coords['Y'].min()
                    y_max = all_coords['Y'].max()

                    if y_min >= 0 and y_max <= 20:
                        y_range = (20, 0)  # 从高到低
                        lane_lines = [6.5, 10, 13.75]
                        ax.annotate('行驶方向为从东向西',
                                    xy=(0.15, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向左侧(0.15)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)
                    else:
                        y_range = (40, 20)  # 从高到低
                        lane_lines = [23.5, 27.25, 31]
                        ax.annotate('行驶方向为从西向东',
                                    xy=(0.85, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向右侧(0.85)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)

                    # 绘制所有轨迹点（蓝色）
                    ax.scatter(all_coords['X'], all_coords['Y'], c='blue', s=1, alpha=0.6, label='All Trajectory')

                    # 绘制采样点（红色）
                    if not sample_coords.empty:
                        ax.scatter(sample_coords['X'], sample_coords['Y'], c='red', s=2, alpha=0.8,
                                   label='Sampled Frames')

                    # 绘制车道线
                    x_range = [0, 340]
                    for lane_y in lane_lines:
                        ax.plot(x_range, [lane_y, lane_y], color='black', linewidth=0.8, linestyle='--', alpha=0.6)

                    ax.set_xlim(0, 340)
                    ax.set_ylim(y_range[0], y_range[1])  # 修正：Y轴从高到低显示
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)

                ax.set_title(f'Following ID: {vehicle_id}')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.legend()

            # 隐藏多余的子图
            for idx in range(len(selected_following_ids), 10):
                if idx < len(axes):
                    axes[idx].axis('off')

            plt.suptitle('Following Vehicle Trajectories (Blue: All, Red: Sampled)', fontsize=16)
            plt.tight_layout()
            figs.append(('following_trajectories.png', fig))

        # 左变道标签可视化
        if selected_left_ids:
            fig, axes = plt.subplots(5, 2, figsize=(19.2, 10.8))
            axes = axes.flatten()

            for idx, vehicle_id in enumerate(selected_left_ids):
                if idx >= 10:
                    break

                # 获取该车辆的所有XY坐标
                all_coords = smooth_merged_df[smooth_merged_df['ID'] == vehicle_id][['X', 'Y']].dropna()

                # 获取该车辆在样本数据中的帧数信息
                sample_frames = train_df[train_df['ID'] == vehicle_id][
                    'Frame'].values if 'Frame' in train_df.columns else []
                if len(sample_frames) == 0:
                    sample_frames = val_df[val_df['ID'] == vehicle_id][
                        'Frame'].values if 'Frame' in val_df.columns else []

                # 获取样本中的XY坐标
                sample_coords = smooth_merged_df[
                    (smooth_merged_df['ID'] == vehicle_id) &
                    (smooth_merged_df['Frame'].isin(sample_frames))
                    ][['X', 'Y']].dropna()

                ax = axes[idx]

                # 根据Y值确定Y轴范围
                if not all_coords.empty:
                    y_min = all_coords['Y'].min()
                    y_max = all_coords['Y'].max()

                    if y_min >= 0 and y_max <= 20:
                        y_range = (20, 0)  # 从高到低
                        lane_lines = [6.5, 10, 13.75]
                        ax.annotate('行驶方向为从东向西',
                                    xy=(0.15, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向左侧(0.15)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)
                    else:
                        y_range = (40, 20)  # 从高到低
                        lane_lines = [23.5, 27.25, 31]
                        ax.annotate('行驶方向为从西向东',
                                    xy=(0.85, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向右侧(0.85)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)

                    # 绘制所有轨迹点（蓝色）
                    ax.scatter(all_coords['X'], all_coords['Y'], c='blue', s=1, alpha=0.6, label='All Trajectory')

                    # 绘制采样点（红色）
                    if not sample_coords.empty:
                        ax.scatter(sample_coords['X'], sample_coords['Y'], c='red', s=2, alpha=0.8,
                                   label='Sampled Frames')

                    # 绘制车道线
                    x_range = [0, 340]
                    for lane_y in lane_lines:
                        ax.plot(x_range, [lane_y, lane_y], color='black', linewidth=0.8, linestyle='--', alpha=0.6)

                    ax.set_xlim(0, 340)
                    ax.set_ylim(y_range[0], y_range[1])  # 修正：Y轴从高到低显示
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)

                ax.set_title(f'Left Change ID: {vehicle_id}')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.legend()

            # 隐藏多余的子图
            for idx in range(len(selected_left_ids), 10):
                if idx < len(axes):
                    axes[idx].axis('off')

            plt.suptitle('Left Lane Change Vehicle Trajectories (Blue: All, Red: Sampled)', fontsize=16)
            plt.tight_layout()
            figs.append(('left_change_trajectories.png', fig))

        # 右变道标签可视化
        if selected_right_ids:
            fig, axes = plt.subplots(5, 2, figsize=(19.2, 10.8))
            axes = axes.flatten()

            for idx, vehicle_id in enumerate(selected_right_ids):
                if idx >= 10:
                    break

                # 获取该车辆的所有XY坐标
                all_coords = smooth_merged_df[smooth_merged_df['ID'] == vehicle_id][['X', 'Y']].dropna()

                # 获取该车辆在样本数据中的帧数信息
                sample_frames = train_df[train_df['ID'] == vehicle_id][
                    'Frame'].values if 'Frame' in train_df.columns else []
                if len(sample_frames) == 0:
                    sample_frames = val_df[val_df['ID'] == vehicle_id][
                        'Frame'].values if 'Frame' in val_df.columns else []

                # 获取样本中的XY坐标
                sample_coords = smooth_merged_df[
                    (smooth_merged_df['ID'] == vehicle_id) &
                    (smooth_merged_df['Frame'].isin(sample_frames))
                    ][['X', 'Y']].dropna()

                ax = axes[idx]

                # 根据Y值确定Y轴范围
                if not all_coords.empty:
                    y_min = all_coords['Y'].min()
                    y_max = all_coords['Y'].max()

                    if y_min >= 0 and y_max <= 20:
                        y_range = (20, 0)  # 从高到低
                        lane_lines = [6.5, 10, 13.75]
                        ax.annotate('行驶方向为从东向西',
                                    xy=(0.15, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向左侧(0.15)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)
                    else:
                        y_range = (40, 20)  # 从高到低
                        lane_lines = [23.5, 27.25, 31]
                        ax.annotate('行驶方向为从西向东',
                                    xy=(0.85, 0.1), xytext=(0.5, 0.1),  # 从中间(0.5)指向右侧(0.85)
                                    xycoords='axes fraction', textcoords='axes fraction',
                                    arrowprops=dict(facecolor='green', edgecolor='gray', width=1.5, headwidth=7,
                                                    shrink=0.05, alpha=0.8),
                                    ha='center', va='center', fontsize=10, color='black', alpha=0.8, zorder=5)

                    # 绘制所有轨迹点（蓝色）
                    ax.scatter(all_coords['X'], all_coords['Y'], c='blue', s=1, alpha=0.6, label='All Trajectory')

                    # 绘制采样点（红色）
                    if not sample_coords.empty:
                        ax.scatter(sample_coords['X'], sample_coords['Y'], c='red', s=2, alpha=0.8,
                                   label='Sampled Frames')

                    # 绘制车道线
                    x_range = [0, 340]
                    for lane_y in lane_lines:
                        ax.plot(x_range, [lane_y, lane_y], color='black', linewidth=0.8, linestyle='--', alpha=0.6)

                    ax.set_xlim(0, 340)
                    ax.set_ylim(y_range[0], y_range[1])  # 修正：Y轴从高到低显示
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)

                ax.set_title(f'Right Change ID: {vehicle_id}')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.legend()

            # 隐藏多余的子图
            for idx in range(len(selected_right_ids), 10):
                if idx < len(axes):
                    axes[idx].axis('off')

            plt.suptitle('Right Lane Change Vehicle Trajectories (Blue: All, Red: Sampled)', fontsize=16)
            plt.tight_layout()
            figs.append(('right_change_trajectories.png', fig))

        # 保存可视化图
        for filename, fig in figs:
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            print(f"可视化图已保存至: {filepath}")

        plt.show()

else:
    print("未找到标签列，无法按标签进行分割")

print(f"\n分割完成！")
