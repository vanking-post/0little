import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import gc

# from data.ratio import save_dir
save_dir = r"E:\0little\read\CQSkyEyedata5\location5t"
# import data.ratio
# print("ratio.py文件的路径:", data.ratio.__file__)
def data_pca_divide(df_east_sample, df_west_sample):
    """
    Step 07: 归一化、PCA降维与均衡样本分割
    1. 合并东西向数据，对 16 个核心特征进行 Z-Score 归一化
    2. 使用 PCA 降维，保留 95% 以上的方差信息，并输出方差图
    3. 按车辆 ID 进行 8:2 分割，并对“跟驰”样本进行数量均衡下采样
    """
    print("\n=== 开始执行 Step 07: 降维与样本分割 ===")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 合并数据
    df_east_sample['Direction'] = 'East'
    df_west_sample['Direction'] = 'West'
    data = pd.concat([df_east_sample, df_west_sample], ignore_index=True)
    print(f"合并后总样本行数: {len(data)}")

    # 2. 定义 16 个核心特征
    features_for_pca = [
        'Velocity', 'Acceleration', 'lat_Vel', 'lat_Acc', 'long_Vel', 'long_Acc',
        'Following_dist', 'Time_Headway', 'TTC',
        'LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RS_Dist', 'RF_Dist'
    ]

    # 检查特征是否完整
    available_features = [f for f in features_for_pca if f in data.columns]
    if len(available_features) < 16:
        print(
            f"⚠️ 警告：期望 16 个特征，实际找到 {len(available_features)} 个。缺失: {set(features_for_pca) - set(available_features)}")

    # 3. Z-Score 归一化
    print(f"正在对 {len(available_features)} 个特征进行标准化...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[available_features])

    # 4. PCA 降维 (保留 95% 方差)
    print("正在执行 PCA 降维...")
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(scaled_features)

    print(f"  - 原始维度: {len(available_features)}D -> 降维后维度: {X_pca.shape[1]}D")
    print(f"  - 累计方差解释率: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 5. 绘制并保存方差解释图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：单个主成分方差
    n_show = min(11, len(pca.explained_variance_ratio_))
    axes[0].bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show])
    axes[0].set_title('各主成分方差解释率')
    axes[0].set_xlabel('主成分 (PC)')
    axes[0].set_ylabel('方差比例')

    # 图2：累计方差
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'bo-')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% 阈值')
    axes[1].set_title('累计方差解释率')
    axes[1].set_xlabel('主成分数量')
    axes[1].set_ylabel('累计比例')
    axes[1].legend()
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "PCA_Variance_Plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"PCA 方差图已保存至: {plot_path}")

    # 6. 重构 PCA 数据集（拼接回非特征列，如 ID, Label, Frame）
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])])
    meta_cols = [col for col in data.columns if col not in features_for_pca]
    for col in meta_cols:
        pca_df[col] = data[col].values

    # 7. 类别均衡与 8:2 样本分割
    print("\n--- 正在按车辆 ID 进行 8:2 分割与跟驰均衡化 ---")

    # 获取各个标签下独有的车辆 ID
    left_lane_ids = pca_df[pca_df['Label'] == '左变道']['ID'].unique().tolist()
    right_lane_ids = pca_df[pca_df['Label'] == '右变道']['ID'].unique().tolist()
    following_ids = pca_df[pca_df['Label'] == '跟驰']['ID'].unique().tolist()

    print(f"车辆库分布 - 左变道: {len(left_lane_ids)}辆, 右变道: {len(right_lane_ids)}辆, 跟驰: {len(following_ids)}辆")

    # 分割左变道
    left_train_ids, left_val_ids = train_test_split(left_lane_ids, test_size=0.2,
                                                    ) if left_lane_ids else ([], [])
    # 分割右变道
    right_train_ids, right_val_ids = train_test_split(right_lane_ids, test_size=0.2,
                                                      ) if right_lane_ids else ([], [])

    # 均衡跟驰样本数量（目标为左右变道数量的平均值）
    target_train_size = max(1, int((len(left_train_ids) + len(right_train_ids)) / 2))
    target_val_size = max(1, int((len(left_val_ids) + len(right_val_ids)) / 2))

    following_train_ids, following_val_ids = [], []
    if following_ids:
        # 先对跟驰车辆进行基础 8:2 分割
        full_follow_train, full_follow_val = train_test_split(following_ids, test_size=0.2)

        # 再对跟驰池进行随机下采样
        #np.random.seed(42)
        following_train_ids = np.random.choice(
            full_follow_train, size=min(target_train_size, len(full_follow_train)), replace=False).tolist()
        following_val_ids = np.random.choice(
            full_follow_val, size=min(target_val_size, len(full_follow_val)), replace=False).tolist()

    # 8. 根据筛选出的 ID 重构训练集与验证集
    def build_dataset(ids_dict):
        dfs = []
        for label, ids in ids_dict.items():
            if ids:
                dfs.append(pca_df[(pca_df['Label'] == label) & (pca_df['ID'].isin(ids))])
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    train_id_dict = {'左变道': left_train_ids, '右变道': right_train_ids, '跟驰': following_train_ids}
    val_id_dict = {'左变道': left_val_ids, '右变道': right_val_ids, '跟驰': following_val_ids}

    df_train = build_dataset(train_id_dict)
    df_val = build_dataset(val_id_dict)

    print("\n=== 分割完成 ===")
    print(f"训练集行数 (df_train): {len(df_train)}")
    print(f"验证集行数 (df_val): {len(df_val)}")

    print("\n训练集 车辆级 标签分布:")
    train_veh_counts = df_train.groupby('Label')['ID'].nunique()
    for lbl, count in train_veh_counts.items(): print(f"  * {lbl}: {count} 辆车")

    print("\n验证集 车辆级 标签分布:")
    val_veh_counts = df_val.groupby('Label')['ID'].nunique()
    for lbl, count in val_veh_counts.items(): print(f"  * {lbl}: {count} 辆车")

    # 清理内存
    del data, scaled_features, pca_df
    gc.collect()

    return df_train, df_val