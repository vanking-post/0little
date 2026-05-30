import pandas as pd
import numpy as np
import gc


def data_features(df_east_smooth, df_west_smooth):
    """
    Step 05: 特征重构
    1. 将周边 7 个方向的 ID 转化为欧几里得距离
    2. 基于平滑后的坐标和速度，重新核算跟车距离、车头时距和 TTC
    3. 清理冗余的 ID 列
    """

    def process_features(df, direction_label):
        print(f"\n--- 正在重构 {direction_label} 向特征 (数据行数: {len(df)}) ---")

        # 1. 准备用于匹配的坐标字典表（ID + time -> X, Y, Velocity）
        # 仅提取必要列以节省内存
        lookup_table = df[['ID', 'time', 'X', 'Y', 'Velocity']].copy()
        lookup_table.rename(columns={
            'ID': 'OtherID',
            'X': 'OtherX',
            'Y': 'OtherY',
            'Velocity': 'OtherVel'
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
            # 如果找不到对应车辆，距离填充为 0
            df[dist_col] = np.sqrt(
                (df['X'] - df['OtherX']) ** 2 + (df['Y'] - df['OtherY']) ** 2
            ).fillna(0).round(3)

            # 3. 如果是 FrontID，顺便核算三大危险指标
            if dist_col == 'Following_dist':
                # 车头时距 = 距离 / 速度 (处理速度为0的情况)
                df['Time_Headway'] = (df['Following_dist'] / df['Velocity']).replace([np.inf, -np.inf], 0).fillna(
                    0).round(2)

                # TTC = 距离 / 速度差 (当前速度 - 前车速度)
                v_diff = df['Velocity'] - df['OtherVel']
                df['TTC'] = np.where(
                    (v_diff > 0) & (df['Following_dist'] > 0),
                    (df['Following_dist'] / v_diff).round(2),
                    0.0
                )

            # 删除连接产生的冗余列
            df.drop(columns=['OtherID', 'OtherX', 'OtherY', 'OtherVel'], inplace=True)

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
