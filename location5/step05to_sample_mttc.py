import pandas as pd
import numpy as np
import gc


def data_features(df_east_smooth, df_west_smooth):
    """
    Step 05: 特征重构
    1. 将周边 7 个方向的 ID 转化为欧几里得距离
    2. 基于平滑后的坐标和速度，重新核算跟车距离、车头时距和 TTC
    3. 计算 mTTC（修正碰撞时间）和 Lateral Jerk（横向加加速度）
    4. 清理冗余的 ID 列
    """

    def process_features(df, direction_label):
        print(f"\n--- 正在重构 {direction_label} 向特征 (数据行数: {len(df)}) ---")

        # 1. 准备用于匹配的坐标字典表（增加加速度列）
        # 需要用到前车的 long_Acc，因此加入 Acceleration 或 long_Acc（优先使用 long_Acc）
        acc_col = 'long_Acc' if 'long_Acc' in df.columns else 'Acceleration'
        lookup_cols = ['ID', 'time', 'X', 'Y', 'Velocity', acc_col]
        lookup_table = df[lookup_cols].copy()
        lookup_table.rename(columns={
            'ID': 'OtherID',
            'X': 'OtherX',
            'Y': 'OtherY',
            'Velocity': 'OtherVel',
            acc_col: 'OtherAcc'
        }, inplace=True)

        # 定义需要转换的 ID 列及其对应的距离列名
        id_dist_map = {
            'LeftBehindID': 'LB_Dist', 'LeftSideID': 'LS_Dist', 'LeftFrontID': 'LF_Dist',
            'BehindID': 'B_Dist', 'RightBehindID': 'RB_Dist', 'RightSideID': 'RS_Dist',
            'RightFrontID': 'RF_Dist', 'FrontID': 'Following_dist'
        }

        # 2. 向量化计算所有方向的距离
        for id_col, dist_col in id_dist_map.items():
            # 建立连接：自车的周围 ID 与 lookup_table 的 ID 匹配
            df = df.merge(
                lookup_table,
                left_on=['time', id_col],
                right_on=['time', 'OtherID'],
                how='left'
            )

            # 计算欧几里得距离
            df[dist_col] = np.sqrt(
                (df['X'] - df['OtherX']) ** 2 + (df['Y'] - df['OtherY']) ** 2
            ).fillna(0).round(3)

            # 3. 如果是 FrontID，核算危险指标（THW, TTC, mTTC）
            if dist_col == 'Following_dist':
                # 车头时距
                df['Time_Headway'] = (df['Following_dist'] / df['Velocity']).replace([np.inf, -np.inf], 0).fillna(
                    0).round(2)

                # TTC = 距离 / 速度差 (当前速度 - 前车速度)
                v_diff = df['Velocity'] - df['OtherVel']
                df['TTC'] = np.where(
                    (v_diff > 0) & (df['Following_dist'] > 0),
                    (df['Following_dist'] / v_diff).round(2),
                    0.0
                )

                # ----- 计算 mTTC (修正碰撞时间) -----
                # 公式: mTTC = [-Δv + sqrt(Δv^2 + 2*Δa*d)] / Δa
                # 其中 Δv = v_ego - v_lead, Δa = a_ego - a_lead, d = Following_dist
                # 需要当前车的加速度 (long_Acc) 和前车的加速度 (OtherAcc)
                ego_acc_col = 'long_Acc' if 'long_Acc' in df.columns else 'Acceleration'
                if ego_acc_col in df.columns and 'OtherAcc' in df.columns:
                    delta_v = df['Velocity'] - df['OtherVel']
                    delta_a = df[ego_acc_col] - df['OtherAcc']
                    dist = df['Following_dist']

                    # 计算判别式
                    discriminant = delta_v ** 2 + 2 * delta_a * dist
                    # 初始化 mTTC 为 inf（无碰撞风险）
                    mttc = np.full(len(df), np.inf)

                    # 有效条件：delta_v > 0 且 判别式 > 0 且 delta_a != 0
                    valid = (delta_v > 0) & (discriminant > 0) & (delta_a != 0)
                    if valid.any():
                        mttc_valid = (-delta_v[valid] + np.sqrt(discriminant[valid])) / delta_a[valid]
                        # 过滤不合理值 (0~10秒之外认为无风险)
                        #mttc_valid = np.where((mttc_valid > 0) & (mttc_valid <= 10), mttc_valid, np.inf)
                        mttc[valid] = mttc_valid

                    df['mTTC'] = np.round(mttc, 2)
                else:
                    # 缺少加速度列时填充0
                    df['mTTC'] = 0.0
                df['mTTC'] = df['mTTC'].replace([np.inf, -np.inf], 0)

            # 删除连接产生的冗余列
            df.drop(columns=['OtherID', 'OtherX', 'OtherY', 'OtherVel', 'OtherAcc'], inplace=True, errors='ignore')

        # ----- 新增：计算横向加加速度 (Lateral Jerk) -----
        # 确保按 ID 和时间排序
        df = df.sort_values(['ID', 'time']).reset_index(drop=True)
        # 计算横向加速度对时间的导数 (差分法)
        df['Lateral_Jerk'] = df.groupby('ID')['lat_Acc'].diff() / df.groupby('ID')['time'].diff()
        # 填充 NaN 为 0
        df['Lateral_Jerk'] = df['Lateral_Jerk'].fillna(0).round(4)

        # 4. 删除原始 ID 列及冗余字段
        id_cols_to_drop = [
            'LeftBehindID', 'LeftSideID', 'LeftFrontID', 'BehindID',
            'EgoVehicleID', 'FrontID', 'RightBehindID', 'RightSideID',
            'RightFrontID', 'Original_TTC'
        ]
        existing_drop_cols = [c for c in id_cols_to_drop if c in df.columns]
        df.drop(columns=existing_drop_cols, inplace=True)

        print(f"✅ {direction_label} 向特征重构完成")
        return df

    # 分别处理东西向数据
    df_east_sample = process_features(df_east_smooth, "东")
    df_west_sample = process_features(df_west_smooth, "西")

    gc.collect()
    return df_east_sample, df_west_sample