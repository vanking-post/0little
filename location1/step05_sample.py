import pandas as pd
import numpy as np
import gc

def compute_features_with_mttc(df_1_smooth, df_2_smooth, label_1='1', label_2='2',
                               reaction_time=2.0):
    """
    对两个方向的平滑数据进行特征重构，包括：
    1. 将周边 7 个方向的 ID 转化为欧几里得距离，该距离考虑到了车辆自身的长宽度，为保险杠到保险杠的距离，并非质心距离。
    2. 核算跟车距离、车头时距 (THW)、TTC
    3. 计算 mTTC（修正碰撞时间）
    4. 计算 Lateral Jerk（横向加加速度）
    5. 计算 RSD / LF_ERSD / RF_ERSD（危险停车距离）
    6. 清理冗余的 ID 列

    参数:
        df_1_smooth, df_2_smooth: 平滑后的 DataFrame
        label_1, label_2: 方向标签（用于打印）
        reaction_time: 制动反应时间（秒），默认 2.5s

    返回:
        df_1_sample, df_2_sample: 特征重构后的 DataFrame
    """

    def _sd(v):
        """停车距离: v²/(2gf) + t·v,  f=0.6, g=9.81"""
        return v**2 / 11.772 + reaction_time * v

    def process_features(df, label):
        print(f"\n--- 正在重构方向 {label} 特征 (数据行数: {len(df)}) ---")

        # 检查必要列
        required_cols = ['ID', 'Time', 'X', 'Y', 'Velocity']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"数据中缺少必需列: {col}")

        # 1. 准备用于匹配的坐标字典表（增加加速度列）
        acc_col = 'long_Acc' if 'long_Acc' in df.columns else 'Acceleration'
        lookup_cols = ['ID', 'Time', 'X', 'Y', 'Velocity', acc_col, 'Length', 'Width',
                       'long_Vel', 'lat_Vel']
        lookup_table = df[lookup_cols].copy()
        lookup_table.rename(columns={
            'ID': 'OtherID',
            'X': 'OtherX',
            'Y': 'OtherY',
            'Velocity': 'OtherVel',
            acc_col: 'OtherAcc',
            'Length': 'OtherLength',
            'Width': 'OtherWidth',
            'long_Vel': 'OtherLongVel',
            'lat_Vel': 'OtherLatVel',
        }, inplace=True)

        # 定义需要转换的 ID 列及其对应的距离列名
        id_dist_map = {
            'LeftBehindID': 'LB_Dist', 'LeftSideID': 'LS_Dist', 'LeftFrontID': 'LF_Dist',
            'BehindID': 'B_Dist', 'RightBehindID': 'RB_Dist', 'RightSideID': 'RS_Dist',
            'RightFrontID': 'RF_Dist', 'FrontID': 'Following_dist'
        }

        # 2. 向量化计算所有方向的距离
        for id_col, dist_col in id_dist_map.items():
            df = df.merge(
                lookup_table,
                left_on=['Time', id_col],
                right_on=['Time', 'OtherID'],
                how='left'
            )

            # 分解为纵横向分量，减去对应车辆半长/半宽 → 矩形间最近距离
            lon_gap = np.abs(df['X'] - df['OtherX']) - (df['Length'] + df['OtherLength']) / 2
            lat_gap = np.abs(df['Y'] - df['OtherY']) - (df['Width'] + df['OtherWidth']) / 2
            df[dist_col] = np.sqrt(
                np.maximum(lon_gap, 0) ** 2 + np.maximum(lat_gap, 0) ** 2
            ).fillna(0).round(3)

            if dist_col == 'Following_dist':
                # 车头时距
                df['Time_Headway'] = (df['Following_dist'] / df['Velocity']).replace([np.inf, -np.inf], 0).fillna(0).round(2)

                # TTC — 使用纵轴速度 long_Vel（纯纵向碰撞时间）
                v_diff_long = df['long_Vel'] - df['OtherLongVel']
                df['TTC'] = np.where(
                    (v_diff_long > 0) & (df['Following_dist'] > 0),
                    (df['Following_dist'] / v_diff_long).round(2),
                    0.0
                )

                # mTTC — 改用纵轴速度和加速度
                ego_acc_col = 'long_Acc' if 'long_Acc' in df.columns else 'Acceleration'
                if ego_acc_col in df.columns and 'OtherAcc' in df.columns:
                    delta_v = df['long_Vel'] - df['OtherLongVel']
                    delta_a = df[ego_acc_col] - df['OtherAcc']
                    dist = df['Following_dist']
                    discriminant = delta_v**2 + 2 * delta_a * dist
                    mttc = np.full(len(df), np.inf)
                    valid = (delta_v > 0) & (discriminant > 0) & (delta_a != 0)
                    if valid.any():
                        mttc_valid = (-delta_v[valid] + np.sqrt(discriminant[valid])) / delta_a[valid]
                        mttc[valid] = mttc_valid
                    df['mTTC'] = np.round(mttc, 2)
                else:
                    df['mTTC'] = 0.0
                df['mTTC'] = df['mTTC'].replace([np.inf, -np.inf], 0)

                # ETTC — 向量点积求接近速率 AR（捕捉横向接近）
                dx = df['OtherX'] - df['X']
                dy = df['OtherY'] - df['Y']
                dv_lon = df['OtherLongVel'] - df['long_Vel']
                dv_lat = df['OtherLatVel'] - df['lat_Vel']
                dot = dx * dv_lon + dy * dv_lat          # D · V_rel
                center_sq = dx**2 + dy**2
                approaching = (dot < 0) & (center_sq > 0) & (df['Following_dist'] > 0)
                df['ETTC'] = np.where(
                    approaching,
                    (-center_sq / dot).round(2),
                    0.0
                )
                df['ETTC'] = df['ETTC'].replace([np.inf, -np.inf], 0)

                # RSD — 危险停车距离（考虑当前车间距）
                sd_ego = _sd(df['long_Vel'])
                brake_front = df['OtherLongVel']**2 / 11.772  # 前车纯制动距离
                df['RSD'] = np.where(
                    df['TTC'] > 0,
                    (sd_ego - df['Following_dist'] - brake_front).clip(lower=0).round(3),
                    0.0
                )

            # 删除连接产生的冗余列
            df.drop(columns=['OtherID', 'OtherX', 'OtherY', 'OtherVel', 'OtherAcc',
                             'OtherLength', 'OtherWidth', 'OtherLongVel', 'OtherLatVel'],
                    inplace=True, errors='ignore')

        # 3. 计算横向加加速度 (Lateral Jerk)
        df = df.sort_values(['ID', 'Time']).reset_index(drop=True)
        if 'lat_Acc' in df.columns:
            df['Lateral_Jerk'] = df.groupby('ID')['lat_Acc'].diff() / df.groupby('ID')['Time'].diff()
            df['Lateral_Jerk'] = df['Lateral_Jerk'].fillna(0).round(4)
        else:
            print(f"警告: 方向 {label} 缺少 lat_Acc 列，无法计算 Lateral Jerk，填充为 0")
            df['Lateral_Jerk'] = 0.0

        # 4. 删除原始邻车 ID 列
        id_cols_to_drop = [
            'LeftBehindID', 'LeftSideID', 'LeftFrontID', 'BehindID',
            'EgoVehicleID', 'FrontID', 'RightBehindID', 'RightSideID',
            'RightFrontID', 'Original_TTC'
        ]
        existing_drop_cols = [c for c in id_cols_to_drop if c in df.columns]
        df.drop(columns=existing_drop_cols, inplace=True)

        print(f"✅ 方向 {label} 特征重构完成")
        return df

    df_1_sample = process_features(df_1_smooth, label_1)
    df_2_sample = process_features(df_2_smooth, label_2)

    gc.collect()
    return df_1_sample, df_2_sample