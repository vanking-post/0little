# from data.ratio import save_dir
from step_visualizeXY import lane_coeffs_dir11, lane_coeffs_dir12, lane_coeffs_dir21, lane_coeffs_dir22
from step01 import split_by_direction
from step02clean import clean_by_direction
from step03complete import complete_by_direction
from step04smooth import smooth_by_direction
from step05_sample import compute_features_with_mttc
from step06change import extract_lane_change_samples
from step_visualizeXY import visualize_lane_change_samples
import pandas as pd
import numpy as np
import os
file_path_1 = r"E:\0little\read\location1\1-1_trajectory.csv"
file_path_2 = r"E:\0little\read\location1\1-2_trajectory.csv"
save_dir = r"E:\0little\location1"
save_dir1 = r"E:\0little\location1\file1"
save_dir2 = r"E:\0little\location1\file2"


def encode_safety_categories(df):
    """为安全指标添加分类编码列，显式处理 inf/0 特殊值
    新增列: TTC_cat, PET_cat, mTTC_cat, THW_cat, has_front_vehicle, has_rear_vehicle
    """
    # TTC: 0=无前车, >0有效
    df['TTC_cat'] = 'safe'
    df.loc[df['TTC'] == 0, 'TTC_cat'] = 'no_leader'
    df.loc[(df['TTC'] > 0) & (df['TTC'] < 2), 'TTC_cat'] = 'dangerous'
    df.loc[(df['TTC'] >= 2) & (df['TTC'] < 5), 'TTC_cat'] = 'cautious'

    # PET: inf/NaN=无后车, >0有效
    df['PET_cat'] = 'safe'
    pet_invalid = np.isinf(df['PET'].values) | pd.isna(df['PET'].values)
    df.loc[pet_invalid, 'PET_cat'] = 'no_follower'
    df.loc[(df['PET'] > 0) & (df['PET'] < 2), 'PET_cat'] = 'dangerous'
    df.loc[(df['PET'] >= 2) & (df['PET'] < 5), 'PET_cat'] = 'cautious'

    # mTTC: inf/0/NaN=无前车, >0有效
    df['mTTC_cat'] = 'safe'
    mttc_invalid = np.isinf(df['mTTC'].values) | (df['mTTC'] == 0) | pd.isna(df['mTTC'].values)
    df.loc[mttc_invalid, 'mTTC_cat'] = 'no_leader'
    df.loc[(df['mTTC'] > 0) & (df['mTTC'] < 2), 'mTTC_cat'] = 'dangerous'
    df.loc[(df['mTTC'] >= 2) & (df['mTTC'] < 5), 'mTTC_cat'] = 'cautious'

    # THW: 0=无前车, 使用不同阈值 <1s危险, 1-2s谨慎
    df['THW_cat'] = 'safe'
    df.loc[df['Time_Headway'] == 0, 'THW_cat'] = 'no_leader'
    df.loc[(df['Time_Headway'] > 0) & (df['Time_Headway'] < 1), 'THW_cat'] = 'dangerous'
    df.loc[(df['Time_Headway'] >= 1) & (df['Time_Headway'] < 2), 'THW_cat'] = 'cautious'

    # OL-PET: inf/NaN=无原始车道后车, 阈值同 PET（若列不存在则跳过）
    if 'OL_PET' in df.columns:
        df['OL_PET_cat'] = 'safe'
        olp_invalid = np.isinf(df['OL_PET'].values) | pd.isna(df['OL_PET'].values)
        df.loc[olp_invalid, 'OL_PET_cat'] = 'no_follower'
        df.loc[(df['OL_PET'] > 0) & (df['OL_PET'] < 2), 'OL_PET_cat'] = 'dangerous'
        df.loc[(df['OL_PET'] >= 2) & (df['OL_PET'] < 5), 'OL_PET_cat'] = 'cautious'

    # F_ETTC: 目标车道前车 ETTC（若列存在则标注）
    if 'F_ETTC' in df.columns:
        df['F_ETTC_cat'] = 'safe'
        df.loc[(df['F_ETTC'] > 0) & (df['F_ETTC'] < 2), 'F_ETTC_cat'] = 'dangerous'
        df.loc[(df['F_ETTC'] >= 2) & (df['F_ETTC'] < 5), 'F_ETTC_cat'] = 'cautious'

    # 布尔标志
    df['has_front_vehicle'] = (df['TTC'] > 0) | (df['Following_dist'] > 0)
    df['has_rear_vehicle'] = ~(np.isinf(df['PET'].values) | pd.isna(df['PET'].values))

    return df


def main_pipeline():
    df1_1, df1_2 = split_by_direction(file_path_1)
    df2_1, df2_2 = split_by_direction(file_path_2)

    df_clean_11,df_clean_12 = clean_by_direction(df1_1,df1_2)
    df_clean_21,df_clean_22 = clean_by_direction(df2_1,df2_2)

    # 保存清洗后的数据
    # df_clean_11.to_csv(f"{save_dir}/df_clean_11.csv", index=False, encoding='utf-8-sig')
    # df_clean_12.to_csv(f"{save_dir}/df_clean_12.csv", index=False, encoding='utf-8-sig')
    # df1_1.to_csv(f"{save_dir}/df1_1.csv", index=False, encoding='utf-8-sig')
    # df1_2.to_csv(f"{save_dir}/df1_2.csv", index=False, encoding='utf-8-sig')
    # df_clean_21.to_csv(f"{save_dir}/df_clean_21.csv", index=False, encoding='utf-8-sig')
    # df_clean_22.to_csv(f"{save_dir}/df_clean_22.csv", index=False, encoding='utf-8-sig')

    df_comp_11,df_comp_12 = complete_by_direction(df_clean_11,df_clean_12,df1_1,df1_2,1,2)
    df_comp_21,df_comp_22 = complete_by_direction(df_clean_21,df_clean_22,df2_1,df2_2,1,2)

    # 假设 df1, df2 是从上一步得到的两个方向的数据
    df_smooth_11, df_smooth_12 = smooth_by_direction(df_comp_11, df_comp_12, label_1='1', label_2='2')
    df_smooth_21, df_smooth_22 = smooth_by_direction(df_comp_21, df_comp_22, label_1='1', label_2='2')

    df_sample_11,df_sample_12 = compute_features_with_mttc(df_smooth_11,df_smooth_12,1,2)
    df_sample_21,df_sample_22 = compute_features_with_mttc(df_smooth_21,df_smooth_22,1,2)

    # df_sample_11.to_csv(f"{save_dir}/df_sample_11.csv", index=False, encoding='utf-8-sig')
    # df_sample_12.to_csv(f"{save_dir}/df_sample_12.csv", index=False, encoding='utf-8-sig')

    df_left_1,df_right_1 = extract_lane_change_samples(df_sample_11,df_sample_12,
                            df_smooth_11,df_smooth_12,5,1.5)
    df_left_2, df_right_2 = extract_lane_change_samples(df_sample_21, df_sample_22,
                            df_smooth_21, df_smooth_22, 5, 1.5)

    # 安全指标分类编码
    df_left_1 = encode_safety_categories(df_left_1)
    df_right_1 = encode_safety_categories(df_right_1)
    df_left_2 = encode_safety_categories(df_left_2)
    df_right_2 = encode_safety_categories(df_right_2)

    # 按路段分别保存左右变道数据（不同路段车道线不同，不可合并）
    for name, df_data in [('1-1_left', df_left_1), ('1-1_right', df_right_1),
                           ('1-2_left', df_left_2), ('1-2_right', df_right_2)]:
        prefix = name.replace('_left', '').replace('_right', '')  # '1-1' 或 '1-2'
        df_data = df_data.copy()
        df_data['Source'] = prefix
        df_data.to_csv(f"{save_dir}/traffic_{name}_change.csv", index=False, encoding='utf-8-sig')
        print(f"{name} 变道数据已保存: {len(df_data)} 行, {df_data['ID'].nunique()} 辆车")

    visualize_lane_change_samples(df_sample_11, df_sample_12,
                   df_left_1,df_right_1,lane_coeffs_dir11,lane_coeffs_dir12,save_dir1,42)
    visualize_lane_change_samples(df_sample_21,df_sample_22,
                    df_left_2,df_right_2,lane_coeffs_dir21,lane_coeffs_dir22,save_dir2,42)

    # 对 location5 原始变道数据编码并保存
    loc5_dir = r"E:\0little\read\CQSkyEyedata5\location5t"
    save_dir_loc5 = r"E:\0little\location5"
    if not os.path.exists(save_dir_loc5):
        os.makedirs(save_dir_loc5)
    for suffix, fname in [('left', 'traffic_left_change.csv'), ('right', 'traffic_right_change.csv')]:
        fp = os.path.join(loc5_dir, fname)
        if os.path.exists(fp):
            df_loc5 = pd.read_csv(fp)
            df_loc5.loc[:, 'Source'] = 'loc5'

            # 根据 LaneID 添加 Direction 列：0~3 → 1，5~8 → 2
            dir_map = df_loc5.groupby('ID')['LaneID'].agg(lambda x: 1 if x.mode().iloc[0] <= 3 else 2)
            df_loc5['Direction'] = df_loc5['ID'].map(dir_map)

            df_loc5 = encode_safety_categories(df_loc5)
            out = os.path.join(save_dir_loc5, fname)
            df_loc5.to_csv(out, index=False, encoding='utf-8-sig')
            print(f"location5 {suffix} 编码数据已保存: {len(df_loc5)} 行, {df_loc5['ID'].nunique()} 辆车 -> {out}")

    # 合并子文件 → traffic_left/right_change.csv（与 process_all_locations.py 的输出格式对齐）
    for side, parts in [('left', ['1-1_left', '1-2_left']),
                        ('right', ['1-1_right', '1-2_right'])]:
        dfs = []
        for p in parts:
            fp = os.path.join(save_dir, f'traffic_{p}_change.csv')
            if os.path.exists(fp):
                dfs.append(pd.read_csv(fp))
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out = os.path.join(save_dir, f'traffic_{side}_change.csv')
            merged.to_csv(out, index=False, encoding='utf-8-sig')
            print(f"合并 traffic_{side}_change.csv: {len(merged)} 行, {merged['ID'].nunique()} 辆车 -> {out}")

    return

if __name__ == "__main__":
    main_pipeline()
