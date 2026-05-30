import pandas as pd
import numpy as np
import gc
from scipy.interpolate import UnivariateSpline, interp1d

# ---------- 辅助函数（与原始相同） ----------
def check_monotonicity(values):
    if len(values) < 2:
        return True
    diffs = np.diff(values)
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)
    total_diffs = len(diffs)
    if total_diffs == 0:
        return True
    return min(pos_count, neg_count) <= total_diffs * 0.2

def get_monotonicity_direction(values):
    if len(values) < 2:
        return 0
    diffs = np.diff(values)
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    else:
        return 0

def calculate_spline_interpolation(valid_indices, valid_values, target_idx):
    if len(valid_values) < 3:
        return None
    try:
        spline = UnivariateSpline(valid_indices, valid_values, s=0)
        return float(spline(target_idx))
    except Exception:
        try:
            linear_interp = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
            return float(linear_interp(target_idx))
        except Exception:
            return None

# ---------- 主清洗函数（通用版） ----------
def clean_by_direction(df_1, df_2, label_1='1', label_2='2'):
    """
    对方向为1和2的两个DataFrame进行数据清洗（尺寸异常剔除、漂移修复、异常车辆剔除）

    参数:
        df_1, df_2: 需要清洗的DataFrame
        label_1, label_2: 打印信息时使用的方向标签，默认为'1'和'2'

    返回:
        df_1_cleaned, df_2_cleaned: 清洗后的DataFrame

    注意: 假设DataFrame包含以下列：
        ID, Time, Frame, Length, Width, X, Y, Velocity, Acceleration,
        long_Vel, lat_Vel, long_Acc, lat_Acc (部分列若缺失可能影响对应处理步骤)
    """
    # 核心参数
    MIN_VELOCITY, MAX_VELOCITY = 0, 144  # km/h
    MIN_ACC, MAX_ACC = -10, 10           # m/s²
    VALID_RATIO_THRESH = 0.8             # 有效数据占比阈值
    QUANTILE = 0.999                     # 坐标异常分位数

    def fix_simple_drift_and_filter(df, col, min_val, max_val):
        """修复单点漂移，删除持续异常的车辆"""
        df[f"{col}_prev"] = df.groupby("ID")[col].shift(1)
        df[f"{col}_next"] = df.groupby("ID")[col].shift(-1)

        out_of_range = ~df[col].between(min_val, max_val)
        prev_in_range = df[f"{col}_prev"].between(min_val, max_val) & df[f"{col}_prev"].notna()
        next_in_range = df[f"{col}_next"].between(min_val, max_val) & df[f"{col}_next"].notna()
        drift_mask = out_of_range & prev_in_range & next_in_range

        if drift_mask.sum() > 0:
            df.loc[drift_mask, col] = np.nan
            df[col] = df.groupby("ID")[col].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))

        df[f"{col}_valid"] = df[col].between(min_val, max_val)
        valid_ratio = df.groupby("ID")[f"{col}_valid"].transform('mean')
        df = df[valid_ratio >= VALID_RATIO_THRESH].copy()
        df.drop(columns=[f"{col}_prev", f"{col}_next", f"{col}_valid"], inplace=True)
        return df

    def fix_accel_drift_with_monotonicity(df):
        """基于单调性修复加速度漂移"""
        drift_count = 0
        df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)

        for vid, vid_data in df.groupby('ID'):
            if len(vid_data) < 7:
                continue

            acc_values = vid_data['Acceleration'].values
            valid_mask = ~np.isnan(acc_values)
            if np.sum(valid_mask) < 7:
                continue

            out_of_range = ~pd.Series(acc_values).between(MIN_ACC, MAX_ACC).values
            start_global_idx = vid_data.index[0]

            for local_idx in range(len(vid_data)):
                if not out_of_range[local_idx] or not valid_mask[local_idx]:
                    continue

                start_idx = max(0, local_idx - 3)
                end_idx = min(len(vid_data), local_idx + 4)
                if end_idx - start_idx < 7:
                    continue

                before_values = acc_values[start_idx:local_idx]
                after_values = acc_values[local_idx + 1:end_idx]

                is_monotonic_before = check_monotonicity(before_values)
                is_monotonic_after = check_monotonicity(after_values)

                if is_monotonic_before and is_monotonic_after and \
                        get_monotonicity_direction(before_values) == get_monotonicity_direction(after_values):

                    valid_interp_indices = []
                    valid_interp_values = []
                    for i in range(start_idx, end_idx):
                        if valid_mask[i] and i != local_idx:
                            valid_interp_indices.append(i)
                            valid_interp_values.append(acc_values[i])

                    if len(valid_interp_values) >= 3:
                        corrected_value = calculate_spline_interpolation(
                            np.array(valid_interp_indices), np.array(valid_interp_values), local_idx
                        )
                        if corrected_value is not None:
                            global_idx = start_global_idx + local_idx
                            df.at[global_idx, 'Acceleration'] = corrected_value
                            drift_count += 1

        print(f"    - 加速度单调性修复完成，共修复 {drift_count} 个漂移点")

        df["acc_valid"] = df["Acceleration"].between(MIN_ACC, MAX_ACC)
        valid_ratio = df.groupby("ID")["acc_valid"].transform('mean')
        df = df[valid_ratio >= VALID_RATIO_THRESH].copy()
        df.drop(columns=["acc_valid"], inplace=True)
        return df

    def process_single_df(df, label):
        print(f"\n--- 正在清洗方向 {label} 数据 (原始行数: {len(df)}) ---")

        # 确保必要列存在（若缺少某些列可跳过对应处理，这里仅检查关键列）
        required_cols = ['ID', 'Time', 'Frame', 'Length', 'Width', 'X', 'Y', 'Velocity', 'Acceleration']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"数据中缺少必需列: {col}")

        df = df.sort_values(by=["ID", "Time"]).reset_index(drop=True)

        # 尺寸异常剔除
        df = df[~((df["Length"] < 2) & (df["Width"] < 1))].copy()

        # X, Y 漂移修复
        x_min, x_max = df["X"].quantile(1 - QUANTILE), df["X"].quantile(QUANTILE)
        y_min, y_max = df["Y"].quantile(1 - QUANTILE), df["Y"].quantile(QUANTILE)
        df = fix_simple_drift_and_filter(df, "X", x_min, x_max)
        df = fix_simple_drift_and_filter(df, "Y", y_min, y_max)

        # 速度漂移修复
        df = fix_simple_drift_and_filter(df, "Velocity", MIN_VELOCITY, MAX_VELOCITY)

        # 加速度漂移修复（需要 Acceleration 列）
        if 'Acceleration' in df.columns:
            df = fix_accel_drift_with_monotonicity(df)
        else:
            print("    - 警告: 缺少 Acceleration 列，跳过加速度漂移修复")

        print(f"✅ 方向 {label} 清洗完成，最终剩余行数: {len(df)}")
        return df

    df_1_cleaned = process_single_df(df_1, label_1)
    df_2_cleaned = process_single_df(df_2, label_2)

    gc.collect()
    return df_1_cleaned, df_2_cleaned

