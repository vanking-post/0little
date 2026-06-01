import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

THRESHOLD_RAMP_SUBSEQUENT = 50  # 西向匝道后二次变道最小间隔


def extract_lane_change_samples(df_east, df_west, full_east=None, full_west=None,
                                reaction_time=2.5,
                                pre_frames=100, sample_frames=50):
    """
    从东西向轨迹数据中提取左变道和右变道样本（不包含跟驰），并计算 PET（后侵入时间）

    参数:
        df_east: 东向样本数据（已删除邻车ID，但保留 Frame, time, ID 等）
        df_west: 西向样本数据
        full_east: 东向完整数据（保留所有邻车ID，如 RightBehindID, LeftBehindID）
        full_west: 西向完整数据

    返回:
        df_left_change, df_right_change: 包含 PET 列的变道样本 DataFrame
    """

    # ---------- PET / OL-PET 计算核心 ----------
    def calculate_pet_for_event(full_df, ego_id, change_frame, conflict_x, target_lane_dir):
        """
        返回 (PET, OL_PET)
        PET: 目标车道后车到达冲突点的时间差
        OL_PET: 原始车道（同车道正后方）后车到达冲突点的时间差
        """
        offset = 5
        target_frame = change_frame - offset
        if target_frame < 0:
            return np.inf, np.inf
        ego_rows = full_df[(full_df['ID'] == ego_id) & (full_df['Frame'] == target_frame)]
        if ego_rows.empty:
            return np.inf, np.inf
        ego_row = ego_rows.iloc[0]

        # 本车离开原车道的时间
        exit_row = full_df[(full_df['ID'] == ego_id) & (full_df['Frame'] == change_frame - 1)]
        if exit_row.empty:
            return np.inf, np.inf
        exit_time = exit_row.iloc[0]['time']

        def _calc_one(col_name):
            if col_name not in ego_row:
                return np.inf
            rid = ego_row[col_name]
            if pd.isna(rid):
                return np.inf
            traj = full_df[(full_df['ID'] == rid) & (full_df['time'] >= exit_time)]
            if traj.empty:
                return np.inf
            # 获取后车长度，调整冲突点到保险杠接触位置
            follower_row = full_df[full_df['ID'] == rid].iloc[0] if len(full_df[full_df['ID'] == rid]) > 0 else None
            follower_len = follower_row['Length'] if follower_row is not None else 0
            adjusted_x = conflict_x - (ego_length + follower_len) / 2
            arrival = traj[(traj['X'] >= adjusted_x - 1.5) & (traj['X'] <= adjusted_x + 1.5)]
            if arrival.empty:
                return np.inf
            t = arrival['time'].min()
            pet = t - exit_time
            return round(pet, 2) if pet >= 0 else np.inf

        # 本车长度（用于调整冲突点到保险杠接触位置）
        ego_length = float(ego_row['Length']) if 'Length' in ego_row else 0

        target_col = 'LeftBehindID' if target_lane_dir == 'left' else 'RightBehindID'
        orig_col = 'BehindID'

        return _calc_one(target_col), _calc_one(orig_col)

    # ---------- 辅助函数：处理单个方向 ----------
    def process_direction(df, direction, full_df=None):
        """direction: 'east' 或 'west'，返回 left_samples, right_samples 列表"""
        left_samples = []
        right_samples = []
        source_counts = {'左变道': defaultdict(int), '右变道': defaultdict(int)}

        can_compute_pet = full_df is not None

        if direction == 'west':
            # 西向处理（车道 5~8）
            ramp_lane = 8

            def get_dir(cur, prev):
                return 'L' if cur < prev else 'R'  # 西向：车道减小为左变道

            for vehicle_id, group in df.groupby('ID'):
                group = group.sort_values('time')
                lanes = group['LaneID'].values
                if len(lanes) < 2:
                    continue

                change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
                unique_seq = [lanes[0]] + lanes[change_idx].tolist()

                # 匝道汇入车辆（起始车道为8）
                if unique_seq[0] == ramp_lane:
                    if len(change_idx) > 1:
                        second_idx = change_idx[1]
                        if (second_idx - change_idx[0]) >= THRESHOLD_RAMP_SUBSEQUENT and second_idx >= pre_frames:
                            sample = group.iloc[second_idx - pre_frames:second_idx - (pre_frames - sample_frames)].copy()
                            sample['Label'] = '左变道'
                            # 计算 PET 和 OL-PET
                            pet, ol_pet = np.inf, np.inf
                            if can_compute_pet:
                                change_frame = group.iloc[second_idx]['Frame']
                                conflict_x = group.iloc[second_idx - 1]['X']
                                pet, ol_pet = calculate_pet_for_event(full_df, vehicle_id, change_frame, conflict_x, 'left')
                            sample['PET'] = pet
                            sample['OL_PET'] = ol_pet
                            left_samples.append(sample)
                            source_counts['左变道']['匝道驶入后续变道'] += 1
                    continue

                # 主路车辆
                if len(unique_seq) < 2:
                    continue

                changes = [get_dir(unique_seq[i], unique_seq[i - 1]) for i in range(1, len(unique_seq))]
                if len(changes) == 1:
                    behavior = '左变道' if changes[0] == 'L' else '右变道'
                    src_tag = '普通变道'
                else:
                    mapping = {('L', 'L'): '左变道', ('L', 'R'): '左变道',
                               ('R', 'L'): '右变道', ('R', 'R'): '右变道'}
                    behavior = mapping.get((changes[0], changes[-1]), None)
                    src_tag = '二次变道'
                if behavior is None:
                    continue

                first_idx = change_idx[0]
                if first_idx >= pre_frames:
                    sample = group.iloc[first_idx - pre_frames:first_idx - (pre_frames - sample_frames)].copy()
                    sample['Label'] = behavior
                    # 计算 PET 和 OL-PET
                    pet, ol_pet = np.inf, np.inf
                    if can_compute_pet:
                        change_frame = group.iloc[first_idx]['Frame']
                        conflict_x = group.iloc[first_idx - 1]['X']
                        # 确定目标车道方向
                        prev_lane = group.iloc[first_idx - 1]['LaneID']
                        cur_lane = group.iloc[first_idx]['LaneID']
                        target_dir = 'left' if cur_lane < prev_lane else 'right'  # 西向：车道减小为左
                        pet, ol_pet = calculate_pet_for_event(full_df, vehicle_id, change_frame, conflict_x, target_dir)
                    sample['PET'] = pet
                    sample['OL_PET'] = ol_pet
                    if behavior == '左变道':
                        left_samples.append(sample)
                    else:
                        right_samples.append(sample)
                    source_counts[behavior][src_tag] += 1

        else:  # direction == 'east'
            # 东向处理（车道 0~3）
            ramp_lane = 0

            for vehicle_id, group in df.groupby('ID'):
                group = group.sort_values('time')
                lanes = group['LaneID'].values
                if len(lanes) < 2:
                    continue

                change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
                if len(change_idx) == 0:
                    continue

                first_idx = change_idx[0]
                prev_lane = lanes[first_idx - 1]
                cur_lane = lanes[first_idx]

                # 排除 1→0 驶离匝道
                if prev_lane == 1 and cur_lane == ramp_lane:
                    continue

                behavior = '左变道' if cur_lane > prev_lane else '右变道'
                if first_idx >= pre_frames:
                    sample = group.iloc[first_idx - pre_frames:first_idx - (pre_frames - sample_frames)].copy()
                    sample['Label'] = behavior
                    # 计算 PET 和 OL-PET
                    pet, ol_pet = np.inf, np.inf
                    if can_compute_pet:
                        change_frame = group.iloc[first_idx]['Frame']
                        conflict_x = group.iloc[first_idx - 1]['X']
                        target_dir = 'left' if cur_lane > prev_lane else 'right'
                        pet, ol_pet = calculate_pet_for_event(full_df, vehicle_id, change_frame, conflict_x, target_dir)
                    sample['PET'] = pet
                    sample['OL_PET'] = ol_pet
                    if behavior == '左变道':
                        left_samples.append(sample)
                    else:
                        right_samples.append(sample)
                    source_counts[behavior]['普通变道'] += 1

        # --- 计算目标车道前/后车 ETTC、ERSD ---
        def _build_with_nbr_metrics(samples_list, label):
            if not samples_list:
                return pd.DataFrame()
            result = pd.concat(samples_list, ignore_index=True)
            if full_df is None or 'long_Vel' not in full_df.columns:
                return result

            front_col = 'LeftFrontID' if label == 'left' else 'RightFrontID'
            behind_col = 'LeftBehindID' if label == 'left' else 'RightBehindID'

            nbr_ids = full_df[['ID', 'Frame', front_col, behind_col]].copy()
            result = result.merge(nbr_ids, on=['ID', 'Frame'], how='left')

            nbr_xyv = full_df[['ID', 'Frame', 'X', 'Y', 'long_Vel', 'lat_Vel', 'long_Acc']].copy()

            # 前车
            front = nbr_xyv.rename(columns={'ID': 'FID', 'X': 'FX', 'Y': 'FY',
                                             'long_Vel': 'FLon', 'lat_Vel': 'FLat',
                                             'long_Acc': 'FAcc'})
            result = result.merge(front, left_on=[front_col, 'Frame'],
                                   right_on=['FID', 'Frame'], how='left')
            dx = result['FX'] - result['X']
            dy = result['FY'] - result['Y']
            dv_lon = result['FLon'] - result['long_Vel']
            dv_lat = result['FLat'] - result['lat_Vel']
            dot = dx * dv_lon + dy * dv_lat
            csq = dx**2 + dy**2
            app = (dot < 0) & (csq > 0)
            result['F_ETTC'] = np.where(app, (-csq / dot).round(2), 0.0)
            result['F_ETTC'] = result['F_ETTC'].replace([np.inf, -np.inf], 0)

            gap_front = 'LF_Dist' if label == 'left' else 'RF_Dist'
            gap_behind = 'LB_Dist' if label == 'left' else 'RB_Dist'

            sd_ego = result['long_Vel']**2 / 11.772 + reaction_time * result['long_Vel']
            brake_front = result['FLon']**2 / 11.772
            result['F_ERSD'] = np.where(
                result['F_ETTC'] > 0,
                (sd_ego - result[gap_front] - brake_front).clip(lower=0).round(3), 0.0
            )

            # F_mTTC: 本车追目标车道前车（考虑加速度）
            f_dv = result['long_Vel'] - result['FLon']
            f_da = result['long_Acc'] - result['FAcc']
            f_dist = result[gap_front]
            f_disc = f_dv ** 2 + 2 * f_da * f_dist
            f_mttc = np.full(len(result), np.inf)
            f_valid = (f_dv > 0) & (f_disc > 0) & (f_da != 0)
            if f_valid.any():
                f_mttc[f_valid] = (-f_dv[f_valid] + np.sqrt(f_disc[f_valid])) / f_da[f_valid]
            result['F_mTTC'] = np.round(f_mttc, 2)
            result['F_mTTC'] = result['F_mTTC'].replace([np.inf, -np.inf], 0)

            # 后车
            behind = nbr_xyv.rename(columns={'ID': 'BID', 'X': 'BX', 'Y': 'BY',
                                              'long_Vel': 'BLon', 'lat_Vel': 'BLat',
                                              'long_Acc': 'BAcc'})
            result = result.merge(behind, left_on=[behind_col, 'Frame'],
                                   right_on=['BID', 'Frame'], how='left')
            dx = result['BX'] - result['X']
            dy = result['BY'] - result['Y']
            dv_lon = result['BLon'] - result['long_Vel']
            dv_lat = result['BLat'] - result['lat_Vel']
            dot = dx * dv_lon + dy * dv_lat
            csq = dx**2 + dy**2
            app = (dot < 0) & (csq > 0)
            result['B_ETTC'] = np.where(app, (-csq / dot).round(2), 0.0)
            result['B_ETTC'] = result['B_ETTC'].replace([np.inf, -np.inf], 0)

            sd_b = result['BLon']**2 / 11.772 + reaction_time * result['BLon']
            brake_ego = result['long_Vel']**2 / 11.772
            result['B_ERSD'] = np.where(
                result['B_ETTC'] > 0,
                (sd_b - result[gap_behind] - brake_ego).clip(lower=0).round(3), 0.0
            )

            # B_mTTC: 后车追本车（考虑加速度）
            b_dv = result['BLon'] - result['long_Vel']
            b_da = result['BAcc'] - result['long_Acc']
            b_dist = result[gap_behind]
            b_disc = b_dv ** 2 + 2 * b_da * b_dist
            b_mttc = np.full(len(result), np.inf)
            b_valid = (b_dv > 0) & (b_disc > 0) & (b_da != 0)
            if b_valid.any():
                b_mttc[b_valid] = (-b_dv[b_valid] + np.sqrt(b_disc[b_valid])) / b_da[b_valid]
            result['B_mTTC'] = np.round(b_mttc, 2)
            result['B_mTTC'] = result['B_mTTC'].replace([np.inf, -np.inf], 0)

            result.drop(columns=['FID', 'FX', 'FY', 'FLon', 'FLat', 'FAcc',
                                  'BID', 'BX', 'BY', 'BLon', 'BLat', 'BAcc',
                                  front_col, behind_col],
                        inplace=True, errors='ignore')
            return result

        df_left = _build_with_nbr_metrics(left_samples, 'left')
        df_right = _build_with_nbr_metrics(right_samples, 'right')

        # 打印方向统计
        print(f"\n{'='*15} step06 {direction.upper()} 向变道样本 {'='*15}")
        for label, src_dict in source_counts.items():
            total = sum(src_dict.values())
            if total > 0:
                print(f"  {label}: {total} 辆 (来源: {dict(src_dict)})")
        return df_left, df_right

    # ---------- 处理东西向，合并结果 ----------
    left_east, right_east = process_direction(df_east, 'east', full_east)
    left_west, right_west = process_direction(df_west, 'west', full_west)

    parts_l = [d for d in [left_east, left_west] if not d.empty]
    parts_r = [d for d in [right_east, right_west] if not d.empty]
    df_left = pd.concat(parts_l, ignore_index=True) if parts_l else pd.DataFrame()
    df_right = pd.concat(parts_r, ignore_index=True) if parts_r else pd.DataFrame()

    # 删除不需要的列
    to_drop = ['Dist_to_right_edge_marking', 'Dist_to_left_marking', 'Dist_to_right_marking']
    for df_out in (df_left, df_right):
        if not df_out.empty:
            cols_to_drop = [c for c in to_drop if c in df_out.columns]
            if cols_to_drop:
                df_out.drop(columns=cols_to_drop, inplace=True)

    # 总体统计
    print(f"\n{'='*20} step06 东西变道样本统计 {'='*20}")
    print(f"总左变道车辆数: {len(df_left)//sample_frames if not df_left.empty else 0}")
    print(f"总右变道车辆数: {len(df_right)//sample_frames if not df_right.empty else 0}")
    if not df_left.empty and 'PET' in df_left.columns:
        finite = df_left['PET'][df_left['PET'] != np.inf]
        print(f"左变道 PET 统计: 有效样本 {len(finite)} 个, 均值={finite.mean():.2f}, 中位数={finite.median():.2f}")
    if not df_right.empty and 'PET' in df_right.columns:
        finite = df_right['PET'][df_right['PET'] != np.inf]
        print(f"右变道 PET 统计: 有效样本 {len(finite)} 个, 均值={finite.mean():.2f}, 中位数={finite.median():.2f}")
    #对PET进行分析
    left_valid = df_left[df_left['PET'] != np.inf]
    right_valid = df_right[df_right['PET'] != np.inf]
    print(f"左变道 PET < 2秒比例: {(left_valid['PET'] < 2).mean():.1%}")
    print(f"右变道 PET < 2秒比例: {(right_valid['PET'] < 2).mean():.1%}")
    print(f"左变道 PET 最大值: {left_valid['PET'].max():.1f}")
    print(f"右变道 PET 最大值: {right_valid['PET'].max():.1f}")
    # 按车辆聚合，计算 PET 有效比例
    left_vehicle_pet = df_left.groupby('ID')['PET'].first()
    left_valid_vehicles = (left_vehicle_pet != np.inf).sum()
    print(f"左变道有效 PET 车辆数: {left_valid_vehicles} / {left_vehicle_pet.count()}")
    right_vehicle_pet = df_right.groupby('ID')['PET'].first()
    right_valid_vehicles = (right_vehicle_pet != np.inf).sum()
    print(f"右变道有效 PET 车辆数: {right_valid_vehicles} / {right_vehicle_pet.count()}")
    # OL-PET 统计
    for side_label, df_side in [('左变道', df_left), ('右变道', df_right)]:
        if not df_side.empty and 'OL_PET' in df_side.columns:
            olp = df_side['OL_PET']
            olp_finite = olp[~np.isinf(olp) & ~pd.isna(olp)]
            left_olp = df_side.groupby('ID')['OL_PET'].first()
            olp_valid_veh = (~np.isinf(left_olp) & ~pd.isna(left_olp)).sum()
            print(f"{side_label} OL-PET 统计: 有效 {len(olp_finite)} 条, "
                  f"均值={olp_finite.mean():.2f}s, <2s={(olp_finite<2).mean():.1%}, "
                  f"有效车辆 {olp_valid_veh}/{df_side['ID'].nunique()}")
    # 如果有效车辆比例明显低于总车辆数（234），说明很多车辆的后车 ID 未找到
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].hist(left_valid['PET'], bins=30, alpha=0.7, label='左变道')
    # axes[0].set_title('左变道 PET 分布')
    # axes[1].hist(right_valid['PET'], bins=30, alpha=0.7, label='右变道', color='orange')
    # axes[1].set_title('右变道 PET 分布')
    # plt.show()

    return df_left, df_right

def extract_following_samples(df_east, df_west, full_east=None, full_west=None, random_state=42,
                              pre_frames=100, sample_frames=50):
    """
    从非变道车辆中提取跟驰样本（每车随机截取 50 帧），并计算后车追尾冲突 B_mTTC。
    location5 专用版（使用 time 列名）。

    参数:
        df_east, df_west: 东西向样本数据（step05 输出，有距离列）
        full_east, full_west: 对应完整数据（有邻车 ID），用于 B_mTTC 计算
        random_state: 随机种子

    返回:
        df_following: 跟驰样本 DataFrame
    """
    rng = np.random.default_rng(random_state)

    def _process_direction(df, full_df):
        if df is None or df.empty:
            return pd.DataFrame()

        change_ids = set()
        for vid, grp in df.groupby('ID'):
            lanes = grp['LaneID'].values
            if len(lanes) >= 2 and (lanes[:-1] != lanes[1:]).any():
                change_ids.add(vid)

        follow_ids = set(df['ID'].unique()) - change_ids
        if not follow_ids:
            return pd.DataFrame()

        has_full = full_df is not None and 'BehindID' in full_df.columns

        samples = []
        for vid in follow_ids:
            grp = df[df['ID'] == vid].sort_values('time')
            if len(grp) <= pre_frames:
                continue
            cut_idx = rng.integers(pre_frames, len(grp))
            sample = grp.iloc[cut_idx - pre_frames:cut_idx - (pre_frames - sample_frames)].copy()
            sample['Label'] = '跟驰'

            if has_full:
                nbr_ids = full_df[['ID', 'Frame', 'BehindID']].copy()
                sample = sample.merge(nbr_ids, on=['ID', 'Frame'], how='left')

                nbr_kin = full_df[['ID', 'Frame', 'long_Vel', 'long_Acc']].copy()
                nbr_kin = nbr_kin.rename(
                    columns={'ID': 'BID', 'long_Vel': 'BLon', 'long_Acc': 'BAcc'})
                sample = sample.merge(nbr_kin, left_on=['BehindID', 'Frame'],
                                      right_on=['BID', 'Frame'], how='left')

                b_dv = sample['BLon'] - sample['long_Vel']
                b_da = sample['BAcc'] - sample['long_Acc']
                b_dist = sample['B_Dist']
                b_disc = b_dv ** 2 + 2 * b_da * b_dist
                b_mttc = np.full(len(sample), np.inf)
                b_valid = (b_dv > 0) & (b_disc > 0) & (b_da != 0)
                if b_valid.any():
                    b_mttc[b_valid] = (-b_dv[b_valid] + np.sqrt(b_disc[b_valid])) / b_da[b_valid]
                sample['B_mTTC'] = np.round(b_mttc, 2)
                sample.drop(columns=['BehindID', 'BID', 'BLon', 'BAcc'],
                            inplace=True, errors='ignore')
            else:
                sample['B_mTTC'] = 0.0
            sample['B_mTTC'] = sample['B_mTTC'].replace([np.inf, -np.inf], 0)

            for col in ['PET', 'OL_PET']:
                if col not in sample.columns:
                    sample[col] = np.inf
            for col in ['F_ETTC', 'F_ERSD', 'B_ETTC', 'B_ERSD', 'F_mTTC']:
                if col not in sample.columns:
                    sample[col] = 0.0

            samples.append(sample)

        if not samples:
            return pd.DataFrame()
        return pd.concat(samples, ignore_index=True)

    df_f_east = _process_direction(df_east, full_east) if df_east is not None else pd.DataFrame()
    df_f_west = _process_direction(df_west, full_west) if df_west is not None else pd.DataFrame()

    parts = [d for d in [df_f_east, df_f_west] if not d.empty]
    if not parts:
        print("\n[WARN] 无跟驰样本")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    print(f"\n{'='*20} 总体跟驰样本统计 {'='*20}")
    print(f"总跟驰车辆数: {len(result)//sample_frames if not result.empty else 0}")
    return result
