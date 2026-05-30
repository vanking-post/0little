import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import gc

def smooth_by_direction(df_1, df_2, label_1='1', label_2='2'):
    """
    对两个方向的轨迹数据进行 SG 滤波平滑（方向标识为 1 和 2）

    参数:
        df_1, df_2: 待平滑的 DataFrame
        label_1, label_2: 方向标签（用于打印），默认为 '1' 和 '2'

    返回:
        df_1_smooth, df_2_smooth: 平滑后的 DataFrame
    """

    # SG 滤波参数配置（与原始一致）
    params = {
        'xy': (9, 3),   # X, Y 坐标平滑
        'vel': (5, 3),  # 速度分量平滑
        'acc': (9, 3),  # 加速度分量平滑
        'other': (9, 3) # 标线距离等物理参数
    }

    def apply_sg_smoothing(df, label):
        print(f"\n--- 正在执行方向 {label} 数据平滑 (数据行数: {len(df)}) ---")
        df_res = df.copy()

        cols_xy = ['X', 'Y']
        cols_vel = ['Velocity', 'long_Vel', 'lat_Vel']
        cols_acc = ['Acceleration', 'long_Acc', 'lat_Acc']
        cols_other = ['Dist_to_right_edge_marking', 'Dist_to_right_marking', 'Dist_to_left_marking']

        smoothed_count = 0

        for vid, group in df_res.groupby('ID'):
            if len(group) < 11:
                continue
            idx = group.index

            for col in cols_xy:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['xy'])
            for col in cols_vel:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['vel'])
            for col in cols_acc:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['acc'])
            for col in cols_other:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['other'])

            smoothed_count += 1

        # 保留两位小数
        float_cols = [c for c in (cols_xy + cols_vel + cols_acc + cols_other) if c in df_res.columns]
        df_res[float_cols] = df_res[float_cols].round(2)

        # 速度单位转换：km/h -> m/s
        if 'Velocity' in df_res.columns:
            df_res['Velocity'] = (df_res['Velocity'] / 3.6).round(2)

        print(f"✅ 方向 {label} 平滑完成，共处理 {smoothed_count} 辆车")
        return df_res

    df_1_smooth = apply_sg_smoothing(df_1, label_1)
    df_2_smooth = apply_sg_smoothing(df_2, label_2)

    gc.collect()
    return df_1_smooth, df_2_smooth