import pandas as pd
import numpy as np
import gc
from scipy.interpolate import UnivariateSpline, interp1d
# ==========================================
# 辅助函数：单调性与插值运算 (提取自 02dataclean.py)
# ==========================================
def check_monotonicity(values):
    """检查数组是否单调"""
    if len(values) < 2: return True
    diffs = np.diff(values)
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)
    total_diffs = len(diffs)
    if total_diffs == 0: return True
    return min(pos_count, neg_count) <= total_diffs * 0.2


def get_monotonicity_direction(values):
    """获取单调方向：1为递增，-1为递减，0为无明显趋势"""
    if len(values) < 2: return 0
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
    """计算样条插值 (失败则回退到线性插值)"""
    if len(valid_values) < 3: return None
    try:
        spline = UnivariateSpline(valid_indices, valid_values, s=0)
        return float(spline(target_idx))
    except Exception:
        try:
            linear_interp = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
            return float(linear_interp(target_idx))
        except Exception:
            return None


# ==========================================
# 主清洗函数
# ==========================================
def data_clean(df_east, df_west):
    """
    完成尺寸异常剔除、漂移值插值修复、以及持续超限车辆剔除
    """
    # --- 核心配置参数 ---
    MIN_VELOCITY, MAX_VELOCITY = 0, 144  # km/h
    MIN_ACC, MAX_ACC = -10, 10  # m/s²
    VALID_RATIO_THRESH = 0.8  # 有效数据占比阈值
    QUANTILE = 0.999  # 坐标异常分位数提取比例

    def fix_simple_drift_and_filter(df, col, min_val, max_val):
        """修复单点漂移，并删除持续异常的车辆 (纯向量化加速)"""
        # 1. 寻找单点漂移
        df[f"{col}_prev"] = df.groupby("ID")[col].shift(1)
        df[f"{col}_next"] = df.groupby("ID")[col].shift(-1)

        out_of_range = ~df[col].between(min_val, max_val)
        prev_in_range = df[f"{col}_prev"].between(min_val, max_val) & df[f"{col}_prev"].notna()
        next_in_range = df[f"{col}_next"].between(min_val, max_val) & df[f"{col}_next"].notna()

        drift_mask = out_of_range & prev_in_range & next_in_range

        # 2. 将漂移点设为NaN并进行双向线性插值
        if drift_mask.sum() > 0:
            df.loc[drift_mask, col] = np.nan
            df[col] = df.groupby("ID")[col].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))

        # 3. 统计有效率并剔除严重异常的整车
        df[f"{col}_valid"] = df[col].between(min_val, max_val)
        valid_ratio = df.groupby("ID")[f"{col}_valid"].transform('mean')
        df = df[valid_ratio >= VALID_RATIO_THRESH].copy()

        # 4. 清理辅助列
        df.drop(columns=[f"{col}_prev", f"{col}_next", f"{col}_valid"], inplace=True)
        return df

    def fix_accel_drift_with_monotonicity(df):
        """基于单调性修复加速度漂移"""
        drift_count = 0
        df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)

        # 遍历每辆车进行复杂的单调性趋势判断
        for vid, vid_data in df.groupby('ID'):
            if len(vid_data) < 7: continue

            acc_values = vid_data['Acceleration'].values
            valid_mask = ~np.isnan(acc_values)
            if np.sum(valid_mask) < 7: continue

            out_of_range = ~pd.Series(acc_values).between(MIN_ACC, MAX_ACC).values

            # 获取当前车辆在原df中的起始索引，便于直接修改原表
            start_global_idx = vid_data.index[0]

            for local_idx in range(len(vid_data)):
                if not out_of_range[local_idx] or not valid_mask[local_idx]: continue

                # 前后各3帧
                start_idx = max(0, local_idx - 3)
                end_idx = min(len(vid_data), local_idx + 4)
                if end_idx - start_idx < 7: continue

                before_values = acc_values[start_idx:local_idx]
                after_values = acc_values[local_idx + 1:end_idx]

                is_monotonic_before = check_monotonicity(before_values)
                is_monotonic_after = check_monotonicity(after_values)

                if is_monotonic_before and is_monotonic_after and \
                        get_monotonicity_direction(before_values) == get_monotonicity_direction(after_values):

                    # 准备插值
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
                            # 直接修改 df 中的值
                            global_idx = start_global_idx + local_idx
                            df.at[global_idx, 'Acceleration'] = corrected_value
                            drift_count += 1

        print(f"    - 加速度单调性修复完成，共修复 {drift_count} 个漂移点")

        # 剔除严重异常的整车
        df["acc_valid"] = df["Acceleration"].between(MIN_ACC, MAX_ACC)
        valid_ratio = df.groupby("ID")["acc_valid"].transform('mean')
        df = df[valid_ratio >= VALID_RATIO_THRESH].copy()
        df.drop(columns=["acc_valid"], inplace=True)
        return df

    def process_single_df(df, direction_label):
        print(f"\n--- 正在清洗 {direction_label} 向数据 (原始行数: {len(df)}) ---")

        # 确保按时间和ID排序
        df = df.sort_values(by=["ID", "time"]).reset_index(drop=True)

        # 0. 尺寸异常剔除
        df = df[~((df["Length"] < 2) & (df["Width"] < 1))].copy()

        # 1. 修复 X, Y 漂移
        x_min, x_max = df["X"].quantile(1 - QUANTILE), df["X"].quantile(QUANTILE)
        y_min, y_max = df["Y"].quantile(1 - QUANTILE), df["Y"].quantile(QUANTILE)
        df = fix_simple_drift_and_filter(df, "X", x_min, x_max)
        df = fix_simple_drift_and_filter(df, "Y", y_min, y_max)

        # 2. 修复速度漂移
        df = fix_simple_drift_and_filter(df, "Velocity", MIN_VELOCITY, MAX_VELOCITY)

        # 3. 修复加速度漂移 (基于单调性)
        df = fix_accel_drift_with_monotonicity(df)

        print(f"✅ {direction_label} 向清洗完成，最终剩余行数: {len(df)}")
        return df

    # 执行清洗
    df_east_cleaned = process_single_df(df_east, "东")
    df_west_cleaned = process_single_df(df_west, "西")

    # 强制内存回收
    gc.collect()

    return df_east_cleaned, df_west_cleaned