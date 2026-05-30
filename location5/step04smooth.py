import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import gc


def data_smooth(df_east, df_west):
    """
    仅对东西向轨迹数据的基础运动学参数进行 SG 滤波平滑
    保留真实的物理尺度，为后续重算 TTC 和距离提供平滑底座
    """

    # --- SG 滤波参数配置 ---
    # 针对高频轨迹数据优化的窗口和阶数
    params = {
        'xy': (9, 3),  # X, Y 坐标平滑
        'vel': (5, 3),  # 速度分量平滑
        'acc': (9, 3),  # 加速度分量平滑
        'other': (5, 3)  # 标线距离等物理参数
    }

    def apply_sg_smoothing(df, direction_label):
        print(f"\n--- 正在执行 {direction_label} 向数据平滑 (数据行数: {len(df)}) ---")
        df_res = df.copy()

        # 定义需要平滑的基础物理列
        cols_xy = ['X', 'Y']
        cols_vel = ['Velocity', 'long_Vel', 'lat_Vel']
        cols_acc = ['Acceleration', 'long_Acc', 'lat_Acc']
        cols_other = ['Dist_to_right_edge_marking', 'Dist_to_right_marking', 'Dist_to_left_marking']

        # 统计成功平滑的车辆数
        smoothed_count = 0

        # 按 ID 分组进行平滑
        for vid, group in df_res.groupby('ID'):
            if len(group) < 11:
                continue  # 数据点过少，跳过滤波

            idx = group.index

            # 1. 坐标平滑
            for col in cols_xy:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['xy'])

            # 2. 速度平滑
            for col in cols_vel:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['vel'])

            # 3. 加速度平滑
            for col in cols_acc:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['acc'])

            # 4. 其他标线距离平滑
            for col in cols_other:
                if col in group.columns:
                    df_res.loc[idx, col] = savgol_filter(group[col], *params['other'])

            smoothed_count += 1

        # 保留两位小数，确保物理精度统一
        float_cols = [c for c in (cols_xy + cols_vel + cols_acc + cols_other) if c in df_res.columns]
        df_res[float_cols] = df_res[float_cols].round(2)
        #将速度单位调整为m/s
        if 'Velocity' in df_res.columns:
            df_res['Velocity'] = (df_res['Velocity'] / 3.6).round(2)

        print(f"✅ {direction_label} 向平滑完成，共处理 {smoothed_count} 辆车")
        return df_res

    # 执行纯平滑操作
    df_east_smooth = apply_sg_smoothing(df_east, "东")
    df_west_smooth = apply_sg_smoothing(df_west, "西")

    gc.collect()

    return df_east_smooth, df_west_smooth