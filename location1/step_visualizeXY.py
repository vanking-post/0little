import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#方向1车道参数
lane_coeffs_dir11 = [
    [-2.8697043709983996e-13, 4.5383524891190506e-10, -2.4913474893073317e-07, 5.594087860276569e-05, 0.002852015341176784, 10.131663115144596],
    [-2.8697043709983996e-13, 4.5383524891190506e-10, -2.4913474893073317e-07, 5.594087860276569e-05, 0.002852015341176784, 13.98166],
    [-2.8697043709983996e-13, 4.5383524891190506e-10, -2.4913474893073317e-07, 5.594087860276569e-05, 0.002852015341176784, 17.73166],
    [-2.8697043709983996e-13, 4.5383524891190506e-10, -2.4913474893073317e-07, 5.594087860276569e-05, 0.002852015341176784, 21.58166]
]
lane_coeffs_dir12 = [
    [-4.789827429287594e-13, 5.017566267511211e-10, -1.7713573746720953e-07, 2.7532549979390646e-05, 0.006586834111928092, 25.52697],
    [-4.789827429287594e-13, 5.017566267511211e-10, -1.7713573746720953e-07, 2.7532549979390646e-05, 0.006586834111928092, 29.37697],
    [-4.789827429287594e-13, 5.017566267511211e-10, -1.7713573746720953e-07, 2.7532549979390646e-05, 0.006586834111928092, 33.12697],
    [-4.789827429287594e-13, 5.017566267511211e-10, -1.7713573746720953e-07, 2.7532549979390646e-05, 0.006586834111928092, 36.97697525604672]
]
#方向2车道参数
lane_coeffs_dir21=[
    [1.8844489886462196e-13, -1.0384994378972738e-10, -9.880871231949187e-09, 1.1783340817251928e-05, 0.010783233707546627, 8.496840530238801],
    [1.8844489886462196e-13, -1.0384994378972738e-10, -9.880871231949187e-09, 1.1783340817251928e-05, 0.010783233707546627, 12.3468405],
    [1.8844489886462196e-13, -1.0384994378972738e-10, -9.880871231949187e-09, 1.1783340817251928e-05, 0.010783233707546627, 16.0968405],
    [1.8844489886462196e-13, -1.0384994378972738e-10, -9.880871231949187e-09, 1.1783340817251928e-05, 0.010783233707546627, 19.9468405]
                   ]
lane_coeffs_dir22 = [
    [-1.0731962291125249e-12, 1.0624119286628631e-09, -3.562947358995208e-07, 4.868882263223829e-05, 0.010659902636325108, 24.0622055],
    [-1.0731962291125249e-12, 1.0624119286628631e-09, -3.562947358995208e-07, 4.868882263223829e-05, 0.010659902636325108, 27.9122055],
    [-1.0731962291125249e-12, 1.0624119286628631e-09, -3.562947358995208e-07, 4.868882263223829e-05, 0.010659902636325108, 31.6622055],
    [-1.0731962291125249e-12, 1.0624119286628631e-09, -3.562947358995208e-07, 4.868882263223829e-05, 0.010659902636325108, 35.51220559230983]
    ]

lane_coeffs_dir31 = [[2.052764343667073e-12, -2.017590077569685e-09, 6.233377833261625e-07, -7.549216882164122e-05, 0.004276731915163275, -3.262900],
                 [2.052764343667073e-12, -2.017590077569685e-09, 6.233377833261625e-07, -7.549216882164122e-05, 0.004276731915163275, 6.47001011],
                 [2.052764343667073e-12, -2.017590077569685e-09, 6.233377833261625e-07, -7.549216882164122e-05, 0.004276731915163275, 10.170010],
                 [2.052764343667073e-12, -2.017590077569685e-09, 6.233377833261625e-07, -7.549216882164122e-05, 0.004276731915163275, 13.870010114151755],
                 [2.052764343667073e-12, -2.017590077569685e-09, 6.233377833261625e-07, -7.549216882164122e-05, 0.004276731915163275, 17.57001011]]

lane_coeffs_dir32 = [ [-8.727222260211208e-12, 5.386349784088073e-09, -1.0030981610495853e-06, 4.557362124886899e-05, 0.002243123675313165, 20.1897852],
                 [-8.727222260211208e-12, 5.386349784088073e-09, -1.0030981610495853e-06, 4.557362124886899e-05, 0.002243123675313165, 23.889785254536367],
                 [-8.727222260211208e-12, 5.386349784088073e-09, -1.0030981610495853e-06, 4.557362124886899e-05, 0.002243123675313165, 27.5897852],
                 [-8.727222260211208e-12, 5.386349784088073e-09, -1.0030981610495853e-06, 4.557362124886899e-05, 0.002243123675313165, 31.28978525],
                 [-8.727222260211208e-12, 5.386349784088073e-09, -1.0030981610495853e-06, 4.557362124886899e-05, 0.002243123675313165, 41.0226852]

]
def visualize_lane_change_samples(
    df_full1, df_full2,          # 方向1和方向2的完整轨迹数据
    df_sample1, df_sample2,      # 左变道样本、右变道样本（可能混合方向）
    lane_params_dir1,            # 方向1车道线系数
    lane_params_dir2,            # 方向2车道线系数
    save_dir=None,
    random_seed=42
):
    """
    可视化左/右变道样本轨迹（完整轨迹 vs 样本片段）
    自动识别每个样本车辆属于方向1还是方向2，并调用对应车道线/坐标。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 车道线绘制函数
    def plot_lane_lines(ax, x_range, coeffs_list, color='black', linestyle='--', linewidth=0.8):
        x_vals = np.linspace(x_range[0], x_range[1], 500)
        for coeffs in coeffs_list:
            y_vals = np.polyval(coeffs, x_vals)
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.6)

    # 方向配置（含坐标范围，Y倒置实现道路俯视效果）
    dir_config = {
        1: {
            'full_df': df_full1,
            'lane_params': lane_params_dir1,
            'x_range': (0, 420),
            'y_range': (25, 5)      # 方向1可见Y范围（可根据实际数据微调）
        },
        2: {
            'full_df': df_full2,
            'lane_params': lane_params_dir2,
            'x_range': (0, 420),
            'y_range': (45, 25)     # 方向2可见Y范围
        }
    }

    # ----- 1. 合并左右样本，添加Label列（若原数据无Label则根据来源补充）-----
    # 假设 df_sample1 全是左变道，df_sample2 全是右变道
    sample_left = df_sample1.copy()
    sample_left['Label'] = '左变道'
    sample_right = df_sample2.copy()
    sample_right['Label'] = '右变道'
    merged_samples = pd.concat([sample_left, sample_right], ignore_index=True)

    # 去除可能的重复ID（同一辆车不可能既左又右，但安全起见）
    merged_samples = merged_samples.drop_duplicates(subset=['ID', 'Frame'])

    # ----- 2. 确定每辆车所属方向（根据ID在哪个full_df中存在）-----
    ids_dir1 = set(df_full1['ID'].unique())
    ids_dir2 = set(df_full2['ID'].unique())

    # 为每个样本添加direction列
    def get_direction(id_val):
        if id_val in ids_dir1:
            return 1
        elif id_val in ids_dir2:
            return 2
        else:
            return None

    merged_samples['direction'] = merged_samples['ID'].apply(get_direction)

    # 剔除无法确定方向的车辆
    unknown_ids = merged_samples[merged_samples['direction'].isna()]['ID'].unique()
    if len(unknown_ids) > 0:
        print(f"警告: 以下车辆ID在方向1和方向2的完整数据中均不存在，已跳过: {unknown_ids}")
    merged_samples = merged_samples.dropna(subset=['direction']).copy()
    merged_samples['direction'] = merged_samples['direction'].astype(int)

    # ----- 3. 按行为（左/右）分别处理 -----
    for behavior in ['左变道', '右变道']:
        behavior_df = merged_samples[merged_samples['Label'] == behavior]
        if behavior_df.empty:
            print(f"没有{behavior}样本，跳过")
            continue

        # 获取所有车辆ID及其方向（去重）
        vehicles = behavior_df[['ID', 'direction']].drop_duplicates()
        n_vehicles = len(vehicles)
        print(f"{behavior} 共有 {n_vehicles} 辆车 (方向1: {sum(vehicles['direction']==1)}, 方向2: {sum(vehicles['direction']==2)})")

        n_select = min(10, n_vehicles)
        if n_select == 0:
            continue

        # 随机抽样（保证可复现）
        selected_vehicles = vehicles.sample(n=n_select, random_state=random_seed)

        # 创建子图
        fig, axes = plt.subplots(5, 2, figsize=(19.2, 10.8))
        axes = axes.flatten()

        for idx, (_, row) in enumerate(selected_vehicles.iterrows()):
            vid = row['ID']
            direction = row['direction']
            cfg = dir_config[direction]

            ax = axes[idx]

            # 获取该车辆的完整轨迹
            vehicle_full = cfg['full_df'][cfg['full_df']['ID'] == vid][['X', 'Y']].dropna()
            if vehicle_full.empty:
                ax.text(0.5, 0.5, f'ID {vid} 无轨迹数据', ha='center', va='center')
                ax.set_title(f'{behavior} ID: {vid} (方向{direction})')
                continue

            # 获取该车辆在样本中的帧号及对应轨迹点
            sample_frames = behavior_df[behavior_df['ID'] == vid]['Frame'].unique()
            vehicle_sample = cfg['full_df'][(cfg['full_df']['ID'] == vid) &
                                            (cfg['full_df']['Frame'].isin(sample_frames))][['X', 'Y']].dropna()

            # 绘图
            ax.scatter(vehicle_full['X'], vehicle_full['Y'],
                       c='blue', s=1, alpha=0.6, label='完整轨迹')
            if not vehicle_sample.empty:
                ax.scatter(vehicle_sample['X'], vehicle_sample['Y'],
                           c='red', s=2, alpha=0.8, label='样本片段')

            plot_lane_lines(ax, cfg['x_range'], cfg['lane_params'])

            ax.set_xlim(cfg['x_range'][0], cfg['x_range'][1])
            ax.set_ylim(cfg['y_range'][0], cfg['y_range'][1])
            ax.set_title(f'{behavior} ID: {vid} (方向{direction})')
            ax.set_xlabel('X 坐标 (m)')
            ax.set_ylabel('Y 坐标 (m)')
            ax.legend(loc='upper right', fontsize=8)

        # 隐藏多余子图
        for idx in range(n_select, 10):
            axes[idx].axis('off')

        # 保存或显示
        filename = f"combined_{behavior}_trajectory.png"
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            plt.suptitle(f'{behavior} 轨迹对比 (蓝:完整轨迹, 红:样本片段)', fontsize=16)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"已保存: {save_path}")
        else:
            plt.suptitle(f'{behavior} 轨迹对比 (蓝:完整轨迹, 红:样本片段)', fontsize=16)
            plt.tight_layout()
            plt.show()
        plt.close()