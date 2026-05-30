# step06_lc_extract.py

import pandas as pd
import numpy as np
from collections import defaultdict

def extract_lane_change_samples(df1, df2, full1=None, full2=None,
                                offset=5, conflict_tolerance=1.5,
                                reaction_time=2.5):
    """
    从方向1和方向2的轨迹数据中提取左变道和右变道样本，并计算 PET。

    参数:
        df1, df2: 两个方向的样本数据（已删除邻车ID，但包含 ID, Time, Frame, LaneID, X, Y 等）
        full1, full2: 对应的完整数据（保留 LeftBehindID, RightBehindID 等），用于 PET 计算，可选
        offset: 读取后车 ID 时相对于变道帧的偏移帧数（默认为5）
        conflict_tolerance: 冲突点 X 坐标匹配误差（米，默认为1.5）

    返回:
        df_left, df_right: 左变道和右变道的样本 DataFrame（每个样本50帧），包含 PET 列
    """

    # ---------- 方向配置 ----------
    config = {
        '1': {
            'name': '1',
            'lane_values': [0, 1, 2],        # 外侧→内侧
            'is_left_when_lane_up': True,    # 车道编号增大 = 左变道
        },
        '2': {
            'name': '2',
            'lane_values': [6, 5, 4],        # 外侧→内侧（注意数值6,5,4）
            'is_left_when_lane_up': False,   # 车道编号增大 = 右变道（即减小为左变道）
        }
    }

    # ---------- PET / OL-PET 计算核心 ----------
    def calculate_pet(full_df, ego_id, change_frame, conflict_x, target_dir):
        """返回 (PET, OL_PET), None 表示无法计算
        PET: 目标车道后车到达冲突点的时间差
        OL_PET: 原始车道后车到达同一冲突点的时间差
        target_dir: 'left' 或 'right'
        """
        target_frame = change_frame - offset
        if target_frame < 0:
            return np.inf, np.inf
        ego_rows = full_df[(full_df['ID'] == ego_id) & (full_df['Frame'] == target_frame)]
        if ego_rows.empty:
            return np.inf, np.inf
        ego_row = ego_rows.iloc[0]

        # 本车离开原车道的时间（变道前一帧）
        exit_row = full_df[(full_df['ID'] == ego_id) & (full_df['Frame'] == change_frame - 1)]
        if exit_row.empty:
            return np.inf, np.inf
        exit_time = exit_row.iloc[0]['Time']

        def _calc_one(col_name):
            if col_name not in ego_row:
                return np.inf
            rid = ego_row[col_name]
            if pd.isna(rid):
                return np.inf
            traj = full_df[(full_df['ID'] == rid) & (full_df['Time'] >= exit_time)]
            if traj.empty:
                return np.inf
            # 获取后车长度，调整冲突点到保险杠接触位置
            follower_row = full_df[full_df['ID'] == rid].iloc[0] if len(full_df[full_df['ID'] == rid]) > 0 else None
            follower_len = follower_row['Length'] if follower_row is not None else 0
            adjusted_x = conflict_x - (ego_length + follower_len) / 2
            arrival = traj[
                (traj['X'] >= adjusted_x - conflict_tolerance) &
                (traj['X'] <= adjusted_x + conflict_tolerance)
            ]
            if arrival.empty:
                return np.inf
            t = arrival['Time'].min()
            pet = t - exit_time
            return round(pet, 2) if pet >= 0 else np.inf

        # 本车长度（用于调整冲突点到保险杠接触位置）
        ego_length = float(ego_row['Length']) if 'Length' in ego_row else 0

        # 目标车道后车 = LeftBehindID(左变道) / RightBehindID(右变道)
        # 原始车道后车 = BehindID（同车道正后方，与变道方向无关）
        target_col = 'LeftBehindID' if target_dir == 'left' else 'RightBehindID'
        orig_col   = 'BehindID'

        return _calc_one(target_col), _calc_one(orig_col)

    # ---------- 处理单个方向 ----------
    def process_direction(df, full_df, dir_cfg):
        name = dir_cfg['name']
        is_left_up = dir_cfg['is_left_when_lane_up']
        can_compute_pet = full_df is not None

        left_samples = []
        right_samples = []
        source_counts = {'左变道': defaultdict(int), '右变道': defaultdict(int)}

        for vehicle_id, group in df.groupby('ID'):
            group = group.sort_values('Time')
            lanes = group['LaneID'].values
            if len(lanes) < 2:
                continue

            # 找出变道位置（车道变化点）
            change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
            if len(change_idx) == 0:
                continue

            # 只取第一次变道作为行为标签（简化）
            first_idx = change_idx[0]
            prev_lane = lanes[first_idx - 1]
            cur_lane = lanes[first_idx]

            # 判定左右变道
            if cur_lane > prev_lane:
                behavior = '左变道' if is_left_up else '右变道'
            else:  # cur_lane < prev_lane
                behavior = '右变道' if is_left_up else '左变道'

            # 提取样本：变道前 100 到 50 帧
            if first_idx >= 100:
                sample = group.iloc[first_idx - 100:first_idx - 50].copy()
                sample['Label'] = behavior

                # 计算 PET 和 OL-PET（原始车道后车 PET）
                pet, ol_pet = np.inf, np.inf
                if can_compute_pet:
                    change_frame = group.iloc[first_idx]['Frame']
                    conflict_x = group.iloc[first_idx - 1]['X']
                    target_dir = 'left' if behavior == '左变道' else 'right'
                    pet, ol_pet = calculate_pet(full_df, vehicle_id, change_frame, conflict_x, target_dir)
                sample['PET'] = pet
                sample['OL_PET'] = ol_pet

                if behavior == '左变道':
                    left_samples.append(sample)
                    source_counts['左变道']['普通变道'] += 1
                else:
                    right_samples.append(sample)
                    source_counts['右变道']['普通变道'] += 1

        # --- 计算目标车道前/后车 ETTC、ERSD ---
        def _build_with_nbr_metrics(samples_list, label):
            if not samples_list:
                return pd.DataFrame()
            result = pd.concat(samples_list, ignore_index=True)
            if full_df is None or 'long_Vel' not in full_df.columns:
                return result

            front_col = 'LeftFrontID' if label == 'left' else 'RightFrontID'
            behind_col = 'LeftBehindID' if label == 'left' else 'RightBehindID'

            # 从 full_df 获取目标车道邻车 ID
            nbr_ids = full_df[['ID', 'Frame', front_col, behind_col]].copy()
            result = result.merge(nbr_ids, on=['ID', 'Frame'], how='left')

            # 邻车坐标/速度表
            nbr_xyv = full_df[['ID', 'Frame', 'X', 'Y', 'long_Vel', 'lat_Vel']].copy()

            # --- 目标车道前车 ---
            front = nbr_xyv.rename(columns={'ID': 'FID', 'X': 'FX', 'Y': 'FY',
                                             'long_Vel': 'FLon', 'lat_Vel': 'FLat'})
            result = result.merge(front, left_on=[front_col, 'Frame'],
                                   right_on=['FID', 'Frame'], how='left')

            dx = result['FX'] - result['X']
            dy = result['FY'] - result['Y']
            dv_lon = result['FLon'] - result['long_Vel']
            dv_lat = result['FLat'] - result['lat_Vel']
            dot = dx * dv_lon + dy * dv_lat
            csq = dx ** 2 + dy ** 2
            app = (dot < 0) & (csq > 0)
            result['F_ETTC'] = np.where(app, (-csq / dot).round(2), 0.0)
            result['F_ETTC'] = result['F_ETTC'].replace([np.inf, -np.inf], 0)

            gap_front = 'LF_Dist' if label == 'left' else 'RF_Dist'
            gap_behind = 'LB_Dist' if label == 'left' else 'RB_Dist'

            sd_ego = result['long_Vel'] ** 2 / 11.772 + reaction_time * result['long_Vel']
            brake_front = result['FLon'] ** 2 / 11.772  # 前车纯制动
            result['F_ERSD'] = np.where(
                result['F_ETTC'] > 0,
                (sd_ego - result[gap_front] - brake_front).clip(lower=0).round(3), 0.0
            )

            # --- 目标车道后车（后车追尾本车） ---
            behind = nbr_xyv.rename(columns={'ID': 'BID', 'X': 'BX', 'Y': 'BY',
                                              'long_Vel': 'BLon', 'lat_Vel': 'BLat'})
            result = result.merge(behind, left_on=[behind_col, 'Frame'],
                                   right_on=['BID', 'Frame'], how='left')

            dx = result['BX'] - result['X']
            dy = result['BY'] - result['Y']
            dv_lon = result['BLon'] - result['long_Vel']
            dv_lat = result['BLat'] - result['lat_Vel']
            dot = dx * dv_lon + dy * dv_lat
            csq = dx ** 2 + dy ** 2
            app = (dot < 0) & (csq > 0)
            result['B_ETTC'] = np.where(app, (-csq / dot).round(2), 0.0)
            result['B_ETTC'] = result['B_ETTC'].replace([np.inf, -np.inf], 0)

            # B_ERSD = SD_后车 − 间距 − 本车纯制动（后车需多停的距离）
            sd_b = result['BLon'] ** 2 / 11.772 + reaction_time * result['BLon']
            brake_ego = result['long_Vel'] ** 2 / 11.772
            result['B_ERSD'] = np.where(
                result['B_ETTC'] > 0,
                (sd_b - result[gap_behind] - brake_ego).clip(lower=0).round(3), 0.0
            )

            # 清理 merge 产生的临时列
            result.drop(columns=['FID', 'FX', 'FY', 'FLon', 'FLat',
                                  'BID', 'BX', 'BY', 'BLon', 'BLat',
                                  front_col, behind_col],
                        inplace=True, errors='ignore')
            return result

        df_left = _build_with_nbr_metrics(left_samples, 'left')
        df_right = _build_with_nbr_metrics(right_samples, 'right')

        print(f"\n{'='*15} 方向 {name} 变道样本 {'='*15}")
        for label, src_dict in source_counts.items():
            total = sum(src_dict.values())
            if total > 0:
                print(f"  {label}: {total} 辆 (来源: {dict(src_dict)})")
        return df_left, df_right

    # 处理两个方向
    df_left1, df_right1 = process_direction(df1, full1, config['1'])
    df_left2, df_right2 = process_direction(df2, full2, config['2'])

    parts_l = [d for d in [df_left1, df_left2] if not d.empty]
    parts_r = [d for d in [df_right1, df_right2] if not d.empty]
    df_left = pd.concat(parts_l, ignore_index=True) if parts_l else pd.DataFrame()
    df_right = pd.concat(parts_r, ignore_index=True) if parts_r else pd.DataFrame()

    # 删除无用列（如标线距离，如果存在）
    to_drop = ['Dist_to_right_edge_marking', 'Dist_to_left_marking', 'Dist_to_right_marking']
    for d in (df_left, df_right):
        if not d.empty:
            cols_drop = [c for c in to_drop if c in d.columns]
            if cols_drop:
                d.drop(columns=cols_drop, inplace=True)

    # 打印统计信息
    print(f"\n{'='*20} 总体变道样本统计 {'='*20}")
    print(f"总左变道车辆数: {len(df_left)//50 if not df_left.empty else 0}")
    print(f"总右变道车辆数: {len(df_right)//50 if not df_right.empty else 0}")

    # if not df_left.empty and 'PET' in df_left.columns:
    #     finite = df_left['PET'][df_left['PET'] != np.inf]
    #     print(f"左变道 PET 统计: 有效样本 {len(finite)} 个, 均值={finite.mean():.2f}, 中位数={finite.median():.2f}")
    # if not df_right.empty and 'PET' in df_right.columns:
    #     finite = df_right['PET'][df_right['PET'] != np.inf]
    #     print(f"右变道 PET 统计: 有效样本 {len(finite)} 个, 均值={finite.mean():.2f}, 中位数={finite.median():.2f}")
    #
    # if not df_left.empty:
    #     left_valid = df_left[df_left['PET'] != np.inf]
    #     if not left_valid.empty:
    #         print(f"左变道 PET < 2秒比例: {(left_valid['PET'] < 2).mean():.1%}")
    # if not df_right.empty:
    #     right_valid = df_right[df_right['PET'] != np.inf]
    #     if not right_valid.empty:
    #         print(f"右变道 PET < 2秒比例: {(right_valid['PET'] < 2).mean():.1%}")

    return df_left, df_right