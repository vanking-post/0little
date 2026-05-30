import pandas as pd
import numpy as np
import gc
import os
from scipy.interpolate import UnivariateSpline

# ---------------------- 数据加载 ----------------------
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
east_pkl_path = os.path.join(save_dir, "traffic_flows_east.pkl")
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")
traffic_flows_east = pd.read_pickle(east_pkl_path)
traffic_flows_west = pd.read_pickle(west_pkl_path)
trace_csv_path = os.path.join(save_dir, "traffic_flows_west_modify_trace.csv")
processed_path = os.path.join(save_dir, "traffic_flows_west_processed.pkl")
processed_path_csv = os.path.join(save_dir, "traffic_flows_west_processed.csv")

# ---------------------- 基础配置 ----------------------
# 速度配置（单位：km/h）
MIN_VELOCITY = 0
MAX_VELOCITY = 144
# 加速度配置（单位：m/s²）
MIN_ACCELERATION = -10
MAX_ACCELERATION = 10
# 分量配置
LAT_VEL_MAX = 10  # 横向速度最大±10 m/s
LAT_ACC_MAX = 5  # 横向加速度最大±5 m/s²
VEC_DEV_THRESH = 5  # 速度合值与分量偏差阈值（km/h）
# TTC/跟车距离/车头时距配置
FOLLOWING_DIST_DEL_THRESH = 0.5  # 跟车距离0~0.5删除行
TIME_HEADWAY_DEL_THRESH = 0.5  # 车头时距0~0.5删除行
TTC_DEL_THRESH = 0.1  # TTC 0~0.1删除行
# 通用配置
QUANTILE = 0.999
VALID_RATIO_THRESH = 0.8  # 有效数据占比<90%则删除车辆

# ---------------------- 初始化溯源记录容器 ----------------------
trace_records = []

# ---------------------- 步骤1：删除Length<2且Width<1的行 ----------------------
print(f"原始数据行数：{len(traffic_flows_west)}")

delete_size_rows = traffic_flows_west[
    (traffic_flows_west["Length"] < 2) & (traffic_flows_west["Width"] < 1)
    ].copy()
delete_size_rows["data_type"] = "长宽不符合-删除"
delete_size_rows["modify_before"] = delete_size_rows.apply(
    lambda x: f"Length={x['Length']}, Width={x['Width']}", axis=1
)
delete_size_rows["modify_after"] = None
trace_records.append(delete_size_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                       "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

traffic_flows_west = traffic_flows_west[
    ~((traffic_flows_west["Length"] < 2) & (traffic_flows_west["Width"] < 1))
].reset_index(drop=True)

print(f"步骤1后行数：{len(traffic_flows_west)}（删除长宽不符合行：{len(delete_size_rows)}）")
del delete_size_rows
gc.collect()


# ---------------------- 步骤2：X/Y坐标预处理 ----------------------
def clean_xy_coords(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    x_min, x_max = df["X"].quantile(1 - QUANTILE), df["X"].quantile(QUANTILE)
    y_min, y_max = df["Y"].quantile(1 - QUANTILE), df["Y"].quantile(QUANTILE)
    print(f"X合理范围：[{x_min:.2f}, {x_max:.2f}]，Y合理范围：[{y_min:.2f}, {y_max:.2f}]")

    drift_xy_rows = pd.DataFrame()
    for coord in ["X", "Y"]:
        df[f"{coord}_prev"] = df.groupby("ID")[coord].shift(1)
        df[f"{coord}_next"] = df.groupby("ID")[coord].shift(-1)

        out_of_range = ~df[coord].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        prev_in_range = df[f"{coord}_prev"].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        next_in_range = df[f"{coord}_next"].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        df[f"{coord}_single_drift"] = out_of_range & prev_in_range & next_in_range

        coord_drift = df[df[f"{coord}_single_drift"]].copy()
        coord_drift[f"{coord}_before"] = coord_drift[coord]
        coord_drift[f"{coord}_after"] = coord_drift.groupby("ID")[coord].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        drift_xy_rows = pd.concat([drift_xy_rows, coord_drift])

    if not drift_xy_rows.empty:
        drift_xy_rows = drift_xy_rows.drop_duplicates(subset=["ID", "Frame"])
        drift_xy_rows["data_type"] = "坐标单点漂移-修改"
        drift_xy_rows["modify_before"] = drift_xy_rows.apply(
            lambda
                x: f"X={x['X_before'] if 'X_before' in x else x['X']:.2f}, Y={x['Y_before'] if 'Y_before' in x else x['Y']:.2f}",
            axis=1
        )
        drift_xy_rows["modify_after"] = drift_xy_rows.apply(
            lambda
                x: f"X={x['X_after'] if 'X_after' in x else x['X']:.2f}, Y={x['Y_after'] if 'Y_after' in x else x['Y']:.2f}",
            axis=1
        )
        trace_list.append(drift_xy_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                         "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

    for coord in ["X", "Y"]:
        df[coord] = df.groupby("ID")[coord].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    df["XY_valid"] = df["X"].between(x_min, x_max) & df["Y"].between(y_min, y_max)
    id_xy_valid_ratio = df.groupby("ID")["XY_valid"].mean()
    invalid_xy_ids = id_xy_valid_ratio[id_xy_valid_ratio < VALID_RATIO_THRESH].index
    delete_xy_rows = df[df["ID"].isin(invalid_xy_ids)].copy()

    if not delete_xy_rows.empty:
        delete_xy_rows["data_type"] = "坐标持续超出-删除车辆"
        delete_xy_rows["modify_before"] = delete_xy_rows.apply(
            lambda x: f"X={x['X']:.2f}, Y={x['Y']:.2f}（超出范围X[{x_min:.2f},{x_max:.2f}], Y[{y_min:.2f},{y_max:.2f}]）",
            axis=1
        )
        delete_xy_rows["modify_after"] = None
        trace_list.append(delete_xy_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

    df = df[~df["ID"].isin(invalid_xy_ids)].reset_index(drop=True)
    xy_aux_cols = [col for col in df.columns if
                   col.endswith(("_prev", "_next", "_single_drift", "_before", "_after", "XY_valid"))]
    df = df.drop(columns=xy_aux_cols)

    print(f"步骤2后行数：{len(df)}（删除坐标持续超出行：{len(delete_xy_rows)}，坐标漂移修改行：{len(drift_xy_rows)}）")
    del drift_xy_rows, delete_xy_rows
    return df


traffic_flows_west = clean_xy_coords(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤3：Velocity预处理 ----------------------
def clean_velocity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    vel_min, vel_max = MIN_VELOCITY, MAX_VELOCITY
    print(f"Velocity合理范围：[{vel_min:.2f}, {vel_max:.2f}] km/h（对应0~144km/h）")

    df["Velocity_prev"] = df.groupby("ID")["Velocity"].shift(1)
    df["Velocity_next"] = df.groupby("ID")["Velocity"].shift(-1)

    vel_out_of_range = ~df["Velocity"].between(vel_min, vel_max)
    vel_prev_in_range = df["Velocity_prev"].between(vel_min, vel_max) & df["Velocity_prev"].notna()
    vel_next_in_range = df["Velocity_next"].between(vel_min, vel_max) & df["Velocity_next"].notna()
    df["Velocity_single_drift"] = vel_out_of_range & vel_prev_in_range & vel_next_in_range

    drift_vel_rows = df[df["Velocity_single_drift"]].copy()
    if not drift_vel_rows.empty:
        drift_vel_rows["Velocity_before"] = drift_vel_rows["Velocity"]
        drift_vel_rows["Velocity_after"] = drift_vel_rows.groupby("ID")["Velocity"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

        drift_vel_rows["data_type"] = "速度单点漂移-修改"
        drift_vel_rows["modify_before"] = drift_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity_before']:.2f} km/h", axis=1
        )
        drift_vel_rows["modify_after"] = drift_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity_after']:.2f} km/h", axis=1
        )
        trace_list.append(drift_vel_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

    df["Velocity"] = df.groupby("ID")["Velocity"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    df["Velocity_valid"] = df["Velocity"].between(vel_min, vel_max)
    id_vel_valid_ratio = df.groupby("ID")["Velocity_valid"].mean()
    invalid_vel_ids = id_vel_valid_ratio[id_vel_valid_ratio < VALID_RATIO_THRESH].index
    delete_vel_rows = df[df["ID"].isin(invalid_vel_ids)].copy()

    if not delete_vel_rows.empty:
        delete_vel_rows["data_type"] = "速度持续偏离-删除车辆"
        delete_vel_rows["modify_before"] = delete_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity']:.2f} km/h（超出范围0~{vel_max:.2f}）", axis=1
        )
        delete_vel_rows["modify_after"] = None
        trace_list.append(delete_vel_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

    df = df[~df["ID"].isin(invalid_vel_ids)].reset_index(drop=True)
    vel_aux_cols = ["Velocity_prev", "Velocity_next", "Velocity_single_drift",
                    "Velocity_before", "Velocity_after", "Velocity_valid"]
    existing_vel_cols = [col for col in vel_aux_cols if col in df.columns]
    df = df.drop(columns=existing_vel_cols)

    print(f"步骤3后行数：{len(df)}（删除速度持续偏离行：{len(delete_vel_rows)}，速度漂移修改行：{len(drift_vel_rows)}）")
    del drift_vel_rows, delete_vel_rows
    return df


traffic_flows_west = clean_velocity(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 工具函数：单调性检测 ----------------------
def check_monotonicity(values):
    """检查数组是否单调"""
    if len(values) < 2:
        return True

    # 计算差值
    diffs = np.diff(values)

    # 检查是否所有差值符号相同（或为0）
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)

    # 如果正差值和负差值都不超过总差值的20%，认为是单调的
    total_diffs = len(diffs)
    if total_diffs == 0:
        return True

    # 允许少量反向变化（最多20%）
    return min(pos_count, neg_count) <= total_diffs * 0.2


def get_monotonicity_direction(values):
    """获取单调方向：1为递增，-1为递减，0为无明显趋势"""
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
    """计算样条插值"""
    if len(valid_values) < 3:
        return None

    try:
        # 使用样条插值
        spline = UnivariateSpline(valid_indices, valid_values, s=0)
        corrected_value = spline(target_idx)
        return corrected_value
    except Exception:
        # 如果样条插值失败，使用线性插值
        from scipy.interpolate import interp1d
        try:
            linear_interp = interp1d(valid_indices, valid_values, kind='linear', fill_value='extrapolate')
            corrected_value = linear_interp(target_idx)
            return corrected_value
        except Exception:
            return None


# ---------------------- 步骤4：Acceleration预处理（单调性检测优化版-修复索引问题） ----------------------
def clean_acceleration_with_monotonicity(df, trace_list):
    """
    使用单调性检测和样条插值的加速度漂移检测
    """
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    acc_min, acc_max = MIN_ACCELERATION, MAX_ACCELERATION
    print(f"Acceleration合理范围：[{acc_min:.2f}, {acc_max:.2f}] m/s²")

    drift_acc_rows = pd.DataFrame()

    # 为每个ID处理加速度漂移
    for vid in df['ID'].unique():
        vid_data = df[df['ID'] == vid].copy().reset_index(drop=True)

        if len(vid_data) < 7:  # 需要至少7个点进行单调性检测
            continue

        # 获取加速度数据
        acc_values = vid_data['Acceleration'].values
        valid_mask = ~np.isnan(acc_values)

        if np.sum(valid_mask) < 7:  # 有效数据点不足
            continue

        # 检查超出范围的点
        out_of_range = ~vid_data['Acceleration'].between(acc_min, acc_max)

        for idx in range(len(vid_data)):
            if not out_of_range.iloc[idx] or not valid_mask[idx]:
                continue

            # 检查前后各3帧的单调性
            start_idx = max(0, idx - 3)
            end_idx = min(len(vid_data), idx + 4)  # idx+3+1，包含idx+3

            if end_idx - start_idx < 7:  # 需要至少7个点
                continue

            # 提取前后各3帧的数据（不包括当前点）
            before_values = acc_values[start_idx:idx]
            after_values = acc_values[idx + 1:end_idx]

            # 检查单调性
            is_monotonic_before = check_monotonicity(before_values)
            is_monotonic_after = check_monotonicity(after_values)

            # 如果前后段都单调且单调性一致，当前点可能是漂移值
            if is_monotonic_before and is_monotonic_after and \
                    get_monotonicity_direction(before_values) == get_monotonicity_direction(after_values):

                # 准备插值数据：排除当前异常点
                range_acc_values = acc_values[start_idx:end_idx]
                range_valid_mask = valid_mask[start_idx:end_idx]

                # 确保数组长度一致
                if len(range_acc_values) != len(range_valid_mask):
                    continue

                # 找到有效索引
                valid_interp_indices = []
                valid_interp_values = []

                for local_idx in range(len(range_acc_values)):
                    global_idx = start_idx + local_idx
                    if range_valid_mask[local_idx] and global_idx != idx:  # 排除当前异常点
                        valid_interp_indices.append(global_idx)
                        valid_interp_values.append(range_acc_values[local_idx])

                if len(valid_interp_values) >= 3:
                    # 使用样条插值计算修正值
                    corrected_value = calculate_spline_interpolation(
                        np.array(valid_interp_indices), np.array(valid_interp_values), idx
                    )

                    if corrected_value is not None:
                        # 创建漂移行记录
                        drift_row = vid_data.iloc[[idx]].copy()
                        drift_row['Acceleration_before'] = drift_row['Acceleration'].iloc[0]
                        drift_row['Acceleration_after'] = corrected_value
                        drift_acc_rows = pd.concat([drift_acc_rows, drift_row])

    if not drift_acc_rows.empty:
        drift_acc_rows["data_type"] = "加速度单点漂移-修改"
        drift_acc_rows["modify_before"] = drift_acc_rows.apply(
            lambda x: f"Acceleration={x['Acceleration_before']:.2f} m/s²", axis=1
        )
        drift_acc_rows["modify_after"] = drift_acc_rows.apply(
            lambda x: f"Acceleration={x['Acceleration_after']:.2f} m/s²", axis=1
        )
        trace_list.append(drift_acc_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

        # 应用修正值
        for idx, row in drift_acc_rows.iterrows():
            mask = (df['ID'] == row['ID']) & (df['Frame'] == row['Frame'])
            df.loc[mask, 'Acceleration'] = row['Acceleration_after']

    df["Acceleration_valid"] = df["Acceleration"].between(acc_min, acc_max)
    id_acc_valid_ratio = df.groupby("ID")["Acceleration_valid"].mean()
    invalid_acc_ids = id_acc_valid_ratio[id_acc_valid_ratio < VALID_RATIO_THRESH].index
    delete_acc_rows = df[df["ID"].isin(invalid_acc_ids)].copy()

    if not delete_acc_rows.empty:
        delete_acc_rows["data_type"] = "加速度持续偏离-删除车辆"
        delete_acc_rows["modify_before"] = delete_acc_rows.apply(
            lambda x: f"Acceleration={x['Acceleration']:.2f} m/s²（超出范围{acc_min:.2f}~{acc_max:.2f}）", axis=1
        )
        delete_acc_rows["modify_after"] = None
        trace_list.append(delete_acc_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity", "Acceleration"]])

    df = df[~df["ID"].isin(invalid_acc_ids)].reset_index(drop=True)
    acc_aux_cols = ["Acceleration_prev", "Acceleration_next", "Acceleration_single_drift",
                    "Acceleration_before", "Acceleration_after", "Acceleration_valid"]
    existing_acc_cols = [col for col in acc_aux_cols if col in df.columns]
    df = df.drop(columns=existing_acc_cols)

    print(f"步骤4后行数：{len(df)}（删除加速度持续偏离行：{len(delete_acc_rows)}，加速度漂移修改行：{len(drift_acc_rows)}）")
    del drift_acc_rows, delete_acc_rows
    return df


traffic_flows_west = clean_acceleration_with_monotonicity(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤5：速度/加速度分量粗筛 ----------------------
def clean_vel_acc_components(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print("\n===== 开始处理速度/加速度分量 =====")

    # ------------- 横向分量物理约束校验 -------------
    comp_config = {
        "lat_Vel": {"max": LAT_VEL_MAX, "unit": "m/s", "name": "横向速度"},
        "lat_Acc": {"max": LAT_ACC_MAX, "unit": "m/s²", "name": "横向加速度"},
        "long_Vel": {"max": None, "unit": "m/s", "name": "纵向速度"},
        "long_Acc": {"max": None, "unit": "m/s²", "name": "纵向加速度"}
    }

    # 1. 横向分量单点漂移修正
    for comp in ["lat_Vel", "lat_Acc"]:
        if comp not in df.columns:
            print(f"⚠️ 未检测到{comp}列，跳过该分量处理")
            continue

        comp_min = -comp_config[comp]["max"]
        comp_max = comp_config[comp]["max"]
        df[f"{comp}_prev"] = df.groupby("ID")[comp].shift(1)
        df[f"{comp}_next"] = df.groupby("ID")[comp].shift(-1)

        comp_out_of_range = ~df[comp].between(comp_min, comp_max)
        comp_prev_in_range = df[f"{comp}_prev"].between(comp_min, comp_max) & df[f"{comp}_prev"].notna()
        comp_next_in_range = df[f"{comp}_next"].between(comp_min, comp_max) & df[f"{comp}_next"].notna()
        df[f"{comp}_single_drift"] = comp_out_of_range & comp_prev_in_range & comp_next_in_range

        drift_comp_rows = df[df[f"{comp}_single_drift"]].copy()
        if not drift_comp_rows.empty:
            drift_comp_rows[f"{comp}_before"] = drift_comp_rows[comp]
            drift_comp_rows[f"{comp}_after"] = drift_comp_rows.groupby("ID")[comp].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )

            drift_comp_rows["data_type"] = f"{comp_config[comp]['name']}单点漂移-修改"
            drift_comp_rows["modify_before"] = drift_comp_rows.apply(
                lambda x: f"{comp_config[comp]['name']}={x[f'{comp}_before']:.2f} {comp_config[comp]['unit']}", axis=1
            )
            drift_comp_rows["modify_after"] = drift_comp_rows.apply(
                lambda x: f"{comp_config[comp]['name']}修正为{x[f'{comp}_after']:.2f} {comp_config[comp]['unit']}",
                axis=1
            )
            trace_list.append(drift_comp_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                               "Length", "Width", "X", "Y", "Velocity", "Acceleration", comp]])
            df.loc[drift_comp_rows.index, comp] = drift_comp_rows[f"{comp}_after"].values

    # 2. 横向分量持续异常车辆删除
    for comp in ["lat_Vel", "lat_Acc"]:
        if comp not in df.columns:
            continue

        comp_min = -comp_config[comp]["max"]
        comp_max = comp_config[comp]["max"]
        df[f"{comp}_valid"] = df[comp].between(comp_min, comp_max)
        comp_valid_ratio = df.groupby("ID")[f"{comp}_valid"].mean()
        invalid_comp_ids = comp_valid_ratio[comp_valid_ratio < VALID_RATIO_THRESH].index
        delete_comp_rows = df[df["ID"].isin(invalid_comp_ids)].copy()

        if not delete_comp_rows.empty:
            delete_comp_rows["data_type"] = f"{comp_config[comp]['name']}持续异常-删除车辆"
            delete_comp_rows["modify_before"] = delete_comp_rows.apply(
                lambda
                    x: f"{comp_config[comp]['name']}={x[comp]:.2f} {comp_config[comp]['unit']}（超出±{comp_config[comp]['max']}）",
                axis=1
            )
            delete_comp_rows["modify_after"] = None
            trace_list.append(delete_comp_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                                "Length", "Width", "X", "Y", "Velocity", "Acceleration", comp]])
            df = df[~df["ID"].isin(invalid_comp_ids)].reset_index(drop=True)

    # ------------- 速度矢量一致性校验 -------------
    if all(col in df.columns for col in ["long_Vel", "lat_Vel"]):
        # 使用更高精度的计算方法
        df["vec_velocity_mps"] = np.sqrt(df["long_Vel"] ** 2 + df["lat_Vel"] ** 2)
        df["vec_velocity_kmh"] = df["vec_velocity_mps"] * 3.6
        # 使用Decimal进行精确比较
        from decimal import Decimal
        def precise_round(value, decimals=2):
            if pd.isna(value):
                return value
            d = Decimal(str(value))
            return float(d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
        
        df["vec_velocity_mps"] = df["vec_velocity_mps"].apply(precise_round)
        df["vec_velocity_kmh"] = df["vec_velocity_kmh"].apply(precise_round)
        df["vel_vec_dev"] = abs(df["Velocity"] - df["vec_velocity_kmh"])
        df["vel_vec_dev"] = df["vel_vec_dev"].apply(precise_round)

        df["vel_vec_drift"] = (df["vel_vec_dev"] > VEC_DEV_THRESH) & \
                              (df["vel_vec_dev"].shift(1).le(VEC_DEV_THRESH)) & \
                              (df["vel_vec_dev"].shift(-1).le(VEC_DEV_THRESH))
        drift_vec_rows = df[df["vel_vec_drift"]].copy()

        if not drift_vec_rows.empty:
            drift_vec_rows["Velocity_before"] = drift_vec_rows["Velocity"]
            drift_vec_rows["Velocity_after"] = drift_vec_rows["vec_velocity_kmh"]
            drift_vec_rows["data_type"] = "速度合值与分量偏差-修正"
            drift_vec_rows["modify_before"] = drift_vec_rows.apply(
                lambda
                    x: f"原始合速度={x['Velocity_before']:.2f} km/h，分量合成={x['vec_velocity_kmh']:.2f} km/h，偏差={x['vel_vec_dev']:.2f} km/h",
                axis=1
            )
            drift_vec_rows["modify_after"] = drift_vec_rows.apply(
                lambda x: f"合速度修正为分量合成值{x['vec_velocity_kmh']:.2f} km/h", axis=1
            )
            trace_list.append(drift_vec_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                              "Length", "Width", "X", "Y", "Velocity", "Acceleration",
                                              "long_Vel", "lat_Vel"]])
            df.loc[drift_vec_rows.index, "Velocity"] = drift_vec_rows["Velocity_after"].values

    # 清理辅助列
    aux_cols = [col for col in df.columns if
                any(col.startswith(c) for c in ["lat_Vel_", "lat_Acc_", "vec_velocity_", "vel_vec_"])]
    df = df.drop(columns=aux_cols)

    print(f"步骤5后行数：{len(df)}（速度/加速度分量处理完成）")
    return df


traffic_flows_west = clean_vel_acc_components(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤6：Following_dist（跟车距离）粗筛 ---------------------
# 小于零的数据重新计算

def clean_following_dist_with_monotonicity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print("\n===== 开始处理跟车距离（Following_dist） =====")
    col_name = "Following_dist"
    if col_name not in df.columns:
        print(f"⚠️ 未检测到{col_name}列，跳过处理")
        return df

    # 1. 删除0 < Following_dist < 0.5的行（无物理意义）
    delete_dist_rows = df[(df[col_name] > 0) & (df[col_name] < FOLLOWING_DIST_DEL_THRESH)].copy()
    if not delete_dist_rows.empty:
        delete_dist_rows["data_type"] = "跟车距离0~0.5-删除行"
        delete_dist_rows["modify_before"] = delete_dist_rows.apply(
            lambda x: f"Following_dist={x[col_name]:.2f} m（0~0.5m范围）", axis=1
        )
        delete_dist_rows["modify_after"] = None
        trace_list.append(delete_dist_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                            "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        # 执行删除
        df = df[~((df[col_name] > 0) & (df[col_name] < FOLLOWING_DIST_DEL_THRESH))].reset_index(drop=True)
    print(f"  - 删除0~0.5m跟车距离行：{len(delete_dist_rows)}")

    # 2. 单点漂移判断与修正（使用单调性检测）
    drift_dist_rows = pd.DataFrame()

    for vid in df['ID'].unique():
        vid_data = df[df['ID'] == vid].copy().reset_index(drop=True)

        if len(vid_data) < 7:  # 需要至少7个点进行单调性检测
            continue

        dist_values = vid_data[col_name].values
        valid_mask = ~np.isnan(dist_values)

        if np.sum(valid_mask) < 7:  # 有效数据点不足
            continue

        # 检查极值点（可能是漂移）
        for idx in range(3, len(vid_data) - 3):  # 前后各留3个点
            if not valid_mask[idx]:
                continue

            # 检查是否为极值点
            current_val = dist_values[idx]
            before_vals = dist_values[idx - 3:idx]
            after_vals = dist_values[idx + 1:idx + 4]

            # 检查单调性
            is_monotonic_before = check_monotonicity(before_vals)
            is_monotonic_after = check_monotonicity(after_vals)

            # 如果前后段都单调且单调性一致，当前点可能是漂移值
            if is_monotonic_before and is_monotonic_after and \
                    get_monotonicity_direction(before_vals) == get_monotonicity_direction(after_vals):

                # 检查当前点是否为异常值
                # 计算前后段的统计量
                all_valid = np.concatenate([before_vals[valid_mask[idx - 3:idx]],
                                            after_vals[valid_mask[idx + 1:idx + 4]]])

                if len(all_valid) >= 3:
                    mean_val = np.mean(all_valid)
                    std_val = np.std(all_valid)

                    # 如果当前值偏离均值超过2个标准差，且前后单调性一致，则认为是漂移
                    if abs(current_val - mean_val) > 2 * std_val:
                        # 准备插值数据：排除当前异常点
                        start_idx = idx - 3
                        end_idx = idx + 4
                        range_dist_values = dist_values[start_idx:end_idx]
                        range_valid_mask = valid_mask[start_idx:end_idx]

                        # 确保数组长度一致
                        if len(range_dist_values) != len(range_valid_mask):
                            continue

                        # 找到有效索引（排除当前异常点）
                        valid_interp_indices = []
                        valid_interp_values = []

                        for local_idx in range(len(range_dist_values)):
                            global_idx = start_idx + local_idx
                            if range_valid_mask[local_idx] and global_idx != idx:  # 排除当前异常点
                                valid_interp_indices.append(global_idx)
                                valid_interp_values.append(range_dist_values[local_idx])

                        if len(valid_interp_values) >= 3:
                            # 使用样条插值计算修正值
                            corrected_value = calculate_spline_interpolation(
                                np.array(valid_interp_indices), np.array(valid_interp_values), idx
                            )

                            if corrected_value is not None:
                                # 创建漂移行记录
                                drift_row = vid_data.iloc[[idx]].copy()
                                drift_row[f'{col_name}_before'] = drift_row[col_name].iloc[0]
                                drift_row[f'{col_name}_after'] = corrected_value
                                drift_dist_rows = pd.concat([drift_dist_rows, drift_row])

    if not drift_dist_rows.empty:
        drift_dist_rows[f"{col_name}_after"] = drift_dist_rows[f"{col_name}_after"].round(2)

        # 记录溯源
        drift_dist_rows["data_type"] = "跟车距离单点漂移-修改"
        drift_dist_rows["modify_before"] = drift_dist_rows.apply(
            lambda x: f"Following_dist={x[f'{col_name}_before']:.2f} m（跳变异常）", axis=1
        )
        drift_dist_rows["modify_after"] = drift_dist_rows.apply(
            lambda x: f"修正为样条插值{x[f'{col_name}_after']:.2f} m", axis=1
        )
        trace_list.append(drift_dist_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])

        # 应用修正
        for idx, row in drift_dist_rows.iterrows():
            mask = (df['ID'] == row['ID']) & (df['Frame'] == row['Frame'])
            df.loc[mask, col_name] = row[f'{col_name}_after']

    print(f"  - 跟车距离单点漂移修改行：{len(drift_dist_rows)}")

    # 3. 持续异常车辆删除（有效占比<90%）
    df[f"{col_name}_valid"] = (df[col_name] == 0) | (df[col_name] >= FOLLOWING_DIST_DEL_THRESH)
    dist_valid_ratio = df.groupby("ID")[f"{col_name}_valid"].mean()
    invalid_dist_ids = dist_valid_ratio[dist_valid_ratio < VALID_RATIO_THRESH].index
    delete_dist_veh_rows = df[df["ID"].isin(invalid_dist_ids)].copy()

    if not delete_dist_veh_rows.empty:
        delete_dist_veh_rows["data_type"] = "跟车距离持续异常-删除车辆"
        delete_dist_veh_rows["modify_before"] = delete_dist_veh_rows.apply(
            lambda x: f"Following_dist={x[col_name]:.2f} m（有效占比<90%）", axis=1
        )
        delete_dist_veh_rows["modify_after"] = None
        trace_list.append(delete_dist_veh_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                                "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        # 执行删除
        df = df[~df["ID"].isin(invalid_dist_ids)].reset_index(drop=True)
    print(f"  - 删除跟车距离持续异常车辆行：{len(delete_dist_veh_rows)}")

    # 清理辅助列
    aux_cols = [col for col in df.columns if col.startswith(f"{col_name}_")]
    df = df.drop(columns=aux_cols)
    print(f"步骤6后行数：{len(df)}（跟车距离处理完成）")
    return df


traffic_flows_west = clean_following_dist_with_monotonicity(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤7：Time_Headway（车头时距）粗筛 ----------------------
def clean_time_headway_with_monotonicity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print("\n===== 开始处理车头时距（Time_Headway） =====")
    col_name = "Time_Headway"
    if col_name not in df.columns:
        print(f"⚠️ 未检测到{col_name}列，跳过处理")
        return df

    # 1. 删除0 < Time_Headway < 0.5的行
    delete_thw_rows = df[(df[col_name] > 0) & (df[col_name] < TIME_HEADWAY_DEL_THRESH)].copy()
    if not delete_thw_rows.empty:
        delete_thw_rows["data_type"] = "车头时距0~0.5-删除行"
        delete_thw_rows["modify_before"] = delete_thw_rows.apply(
            lambda x: f"Time_Headway={x[col_name]:.2f} s（0~0.5s范围）", axis=1
        )
        delete_thw_rows["modify_after"] = None
        trace_list.append(delete_thw_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        # 执行删除
        df = df[~((df[col_name] > 0) & (df[col_name] < TIME_HEADWAY_DEL_THRESH))].reset_index(drop=True)
    print(f"  - 删除0~0.5s车头时距行：{len(delete_thw_rows)}")

    # 2. 单点漂移判断与修正（使用单调性检测）
    drift_thw_rows = pd.DataFrame()

    for vid in df['ID'].unique():
        vid_data = df[df['ID'] == vid].copy().reset_index(drop=True)

        if len(vid_data) < 7:  # 需要至少7个点进行单调性检测
            continue

        thw_values = vid_data[col_name].values
        valid_mask = ~np.isnan(thw_values)

        if np.sum(valid_mask) < 7:  # 有效数据点不足
            continue

        # 检查极值点（可能是漂移）
        for idx in range(3, len(vid_data) - 3):  # 前后各留3个点
            if not valid_mask[idx]:
                continue

            # 检查是否为极值点
            current_val = thw_values[idx]
            before_vals = thw_values[idx - 3:idx]
            after_vals = thw_values[idx + 1:idx + 4]

            # 检查单调性
            is_monotonic_before = check_monotonicity(before_vals)
            is_monotonic_after = check_monotonicity(after_vals)

            # 如果前后段都单调且单调性一致，当前点可能是漂移值
            if is_monotonic_before and is_monotonic_after and \
                    get_monotonicity_direction(before_vals) == get_monotonicity_direction(after_vals):

                # 检查当前点是否为异常值
                # 计算前后段的统计量
                all_valid = np.concatenate([before_vals[valid_mask[idx - 3:idx]],
                                            after_vals[valid_mask[idx + 1:idx + 4]]])

                if len(all_valid) >= 3:
                    mean_val = np.mean(all_valid)
                    std_val = np.std(all_valid)

                    # 如果当前值偏离均值超过2个标准差，且前后单调性一致，则认为是漂移
                    if abs(current_val - mean_val) > 2 * std_val:
                        # 准备插值数据：排除当前异常点
                        start_idx = idx - 3
                        end_idx = idx + 4
                        range_thw_values = thw_values[start_idx:end_idx]
                        range_valid_mask = valid_mask[start_idx:end_idx]

                        # 确保数组长度一致
                        if len(range_thw_values) != len(range_valid_mask):
                            continue

                        # 找到有效索引（排除当前异常点）
                        valid_interp_indices = []
                        valid_interp_values = []

                        for local_idx in range(len(range_thw_values)):
                            global_idx = start_idx + local_idx
                            if range_valid_mask[local_idx] and global_idx != idx:  # 排除当前异常点
                                valid_interp_indices.append(global_idx)
                                valid_interp_values.append(range_thw_values[local_idx])

                        if len(valid_interp_values) >= 3:
                            # 使用样条插值计算修正值
                            corrected_value = calculate_spline_interpolation(
                                np.array(valid_interp_indices), np.array(valid_interp_values), idx
                            )

                            if corrected_value is not None:
                                # 创建漂移行记录
                                drift_row = vid_data.iloc[[idx]].copy()
                                drift_row[f'{col_name}_before'] = drift_row[col_name].iloc[0]
                                drift_row[f'{col_name}_after'] = corrected_value
                                drift_thw_rows = pd.concat([drift_thw_rows, drift_row])

    if not drift_thw_rows.empty:
        drift_thw_rows[f"{col_name}_after"] = drift_thw_rows[f"{col_name}_after"].round(2)

        drift_thw_rows["data_type"] = "车头时距单点漂移-修改"
        drift_thw_rows["modify_before"] = drift_thw_rows.apply(
            lambda x: f"Time_Headway={x[f'{col_name}_before']:.2f} s（跳变异常）", axis=1
        )
        drift_thw_rows["modify_after"] = drift_thw_rows.apply(
            lambda x: f"修正为样条插值{x[f'{col_name}_after']:.2f} s", axis=1
        )
        trace_list.append(drift_thw_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])

        # 应用修正
        for idx, row in drift_thw_rows.iterrows():
            mask = (df['ID'] == row['ID']) & (df['Frame'] == row['Frame'])
            df.loc[mask, col_name] = row[f'{col_name}_after']

    print(f"  - 车头时距单点漂移修改行：{len(drift_thw_rows)}")

    # 3. 持续异常车辆删除
    df[f"{col_name}_valid"] = (df[col_name] == 0) | (df[col_name] >= TIME_HEADWAY_DEL_THRESH)
    thw_valid_ratio = df.groupby("ID")[f"{col_name}_valid"].mean()
    invalid_thw_ids = thw_valid_ratio[thw_valid_ratio < VALID_RATIO_THRESH].index
    delete_thw_veh_rows = df[df["ID"].isin(invalid_thw_ids)].copy()

    if not delete_thw_veh_rows.empty:
        delete_thw_veh_rows["data_type"] = "车头时距持续异常-删除车辆"
        delete_thw_veh_rows["modify_before"] = delete_thw_veh_rows.apply(
            lambda x: f"Time_Headway={x[col_name]:.2f} s（有效占比<90%）", axis=1
        )
        delete_thw_veh_rows["modify_after"] = None
        trace_list.append(delete_thw_veh_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                               "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        df = df[~df["ID"].isin(invalid_thw_ids)].reset_index(drop=True)
    print(f"  - 删除车头时距持续异常车辆行：{len(delete_thw_veh_rows)}")

    # 清理辅助列
    aux_cols = [col for col in df.columns if col.startswith(f"{col_name}_")]
    df = df.drop(columns=aux_cols)
    print(f"步骤7后行数：{len(df)}（车头时距处理完成）")
    return df


traffic_flows_west = clean_time_headway_with_monotonicity(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤8：TTC（碰撞时间）粗筛 ----------------------
def clean_ttc_with_monotonicity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print("\n===== 开始处理碰撞时间（TTC） =====")
    col_name = "TTC"
    if col_name not in df.columns:
        print(f"⚠️ 未检测到{col_name}列，跳过处理")
        return df

    # 1. 删除0 < TTC < 0.1的行
    delete_ttc_rows = df[(df[col_name] > 0) & (df[col_name] < TTC_DEL_THRESH)].copy()
    if not delete_ttc_rows.empty:
        delete_ttc_rows["data_type"] = "TTC0~0.1-删除行"
        delete_ttc_rows["modify_before"] = delete_ttc_rows.apply(
            lambda x: f"TTC={x[col_name]:.2f} s（0~0.1s范围）", axis=1
        )
        delete_ttc_rows["modify_after"] = None
        trace_list.append(delete_ttc_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        # 执行删除
        df = df[~((df[col_name] > 0) & (df[col_name] < TTC_DEL_THRESH))].reset_index(drop=True)
    print(f"  - 删除0~0.1s TTC行：{len(delete_ttc_rows)}")

    # 2. 单点漂移判断与修正（使用单调性检测）
    drift_ttc_rows = pd.DataFrame()

    for vid in df['ID'].unique():
        vid_data = df[df['ID'] == vid].copy().reset_index(drop=True)

        if len(vid_data) < 7:  # 需要至少7个点进行单调性检测
            continue

        ttc_values = vid_data[col_name].values
        valid_mask = ~np.isnan(ttc_values)

        if np.sum(valid_mask) < 7:  # 有效数据点不足
            continue

        # 检查极值点（可能是漂移）
        for idx in range(3, len(vid_data) - 3):  # 前后各留3个点
            if not valid_mask[idx]:
                continue

            # 检查是否为极值点
            current_val = ttc_values[idx]
            before_vals = ttc_values[idx - 3:idx]
            after_vals = ttc_values[idx + 1:idx + 4]

            # 检查单调性
            is_monotonic_before = check_monotonicity(before_vals)
            is_monotonic_after = check_monotonicity(after_vals)

            # 如果前后段都单调且单调性一致，当前点可能是漂移值
            if is_monotonic_before and is_monotonic_after and \
                    get_monotonicity_direction(before_vals) == get_monotonicity_direction(after_vals):

                # 检查当前点是否为异常值
                # 计算前后段的统计量
                all_valid = np.concatenate([before_vals[valid_mask[idx - 3:idx]],
                                            after_vals[valid_mask[idx + 1:idx + 4]]])

                if len(all_valid) >= 3:
                    mean_val = np.mean(all_valid)
                    std_val = np.std(all_valid)

                    # 如果当前值偏离均值超过2个标准差，且前后单调性一致，则认为是漂移
                    if abs(current_val - mean_val) > 2 * std_val:
                        # 准备插值数据：排除当前异常点
                        start_idx = idx - 3
                        end_idx = idx + 4
                        range_ttc_values = ttc_values[start_idx:end_idx]
                        range_valid_mask = valid_mask[start_idx:end_idx]

                        # 确保数组长度一致
                        if len(range_ttc_values) != len(range_valid_mask):
                            continue

                        # 找到有效索引（排除当前异常点）
                        valid_interp_indices = []
                        valid_interp_values = []

                        for local_idx in range(len(range_ttc_values)):
                            global_idx = start_idx + local_idx
                            if range_valid_mask[local_idx] and global_idx != idx:  # 排除当前异常点
                                valid_interp_indices.append(global_idx)
                                valid_interp_values.append(range_ttc_values[local_idx])

                        if len(valid_interp_values) >= 3:
                            # 使用样条插值计算修正值
                            corrected_value = calculate_spline_interpolation(
                                np.array(valid_interp_indices), np.array(valid_interp_values), idx
                            )

                            if corrected_value is not None:
                                # 创建漂移行记录
                                drift_row = vid_data.iloc[[idx]].copy()
                                drift_row[f'{col_name}_before'] = drift_row[col_name].iloc[0]
                                drift_row[f'{col_name}_after'] = corrected_value
                                drift_ttc_rows = pd.concat([drift_ttc_rows, drift_row])

    if not drift_ttc_rows.empty:
        drift_ttc_rows[f"{col_name}_after"] = drift_ttc_rows[f"{col_name}_after"].round(2)

        drift_ttc_rows["data_type"] = "TTC单点漂移-修改"
        drift_ttc_rows["modify_before"] = drift_ttc_rows.apply(
            lambda x: f"TTC={x[f'{col_name}_before']:.2f} s（跳变异常）", axis=1
        )
        drift_ttc_rows["modify_after"] = drift_ttc_rows.apply(
            lambda x: f"修正为样条插值{x[f'{col_name}_after']:.2f} s", axis=1
        )
        trace_list.append(drift_ttc_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])

        # 应用修正
        for idx, row in drift_ttc_rows.iterrows():
            mask = (df['ID'] == row['ID']) & (df['Frame'] == row['Frame'])
            df.loc[mask, col_name] = row[f'{col_name}_after']

    print(f"  - TTC单点漂移修改行：{len(drift_ttc_rows)}")

    # 3. 持续异常车辆删除
    df[f"{col_name}_valid"] = (df[col_name] == 0) | (df[col_name] >= TTC_DEL_THRESH)
    ttc_valid_ratio = df.groupby("ID")[f"{col_name}_valid"].mean()
    invalid_ttc_ids = ttc_valid_ratio[ttc_valid_ratio < VALID_RATIO_THRESH].index
    delete_ttc_veh_rows = df[df["ID"].isin(invalid_ttc_ids)].copy()

    if not delete_ttc_veh_rows.empty:
        delete_ttc_veh_rows["data_type"] = "TTC持续异常-删除车辆"
        delete_ttc_veh_rows["modify_before"] = delete_ttc_veh_rows.apply(
            lambda x: f"TTC={x[col_name]:.2f} s（有效占比<90%）", axis=1
        )
        delete_ttc_veh_rows["modify_after"] = None
        trace_list.append(delete_ttc_veh_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                               "Length", "Width", "X", "Y", "Velocity", "Acceleration", col_name]])
        df = df[~df["ID"].isin(invalid_ttc_ids)].reset_index(drop=True)
    print(f"  - 删除TTC持续异常车辆行：{len(delete_ttc_veh_rows)}")

    # 清理辅助列
    aux_cols = [col for col in df.columns if col.startswith(f"{col_name}_")]
    df = df.drop(columns=aux_cols)
    print(f"步骤8后行数：{len(df)}（TTC处理完成）")
    return df


traffic_flows_west = clean_ttc_with_monotonicity(traffic_flows_west, trace_records)
gc.collect()

# ---------------------- 步骤9：Velocity单位转换（km/h → m/s） ----------------------
print("\n📏 开始转换Velocity单位：km/h → m/s")
# 使用Decimal提高精度，避免浮点转换错误
from decimal import Decimal, ROUND_HALF_UP
def precise_kmh_to_ms(kmh_value):
    """精确地将km/h转换为m/s，避免浮点精度问题"""
    if pd.isna(kmh_value):
        return kmh_value
    # 使用Decimal进行高精度计算
    result = Decimal(str(kmh_value)) / Decimal('3.6')
    # 四舍五入到2位小数
    return float(result.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

traffic_flows_west["Velocity"] = traffic_flows_west["Velocity"].apply(precise_kmh_to_ms)
print("✅ Velocity单位转换完成，km/h → m/s,已保留两位小数（使用高精度计算）")

# ---------------------- 步骤10：合并溯源记录并写入CSV ----------------------
if trace_records:
    trace_df = pd.concat(trace_records, ignore_index=True)
    trace_df = trace_df.sort_values(by=["data_type", "ID", "Frame"]).reset_index(drop=True)
    trace_df.to_csv(trace_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 数据溯源记录已写入：{trace_csv_path}")
    print(f"   溯源记录总行数：{len(trace_df)}")
    print("\n📊 溯源记录统计：")
    print(trace_df["data_type"].value_counts())
else:
    trace_df = pd.DataFrame()
    print("\n⚠️ 无任何修改/删除的行，未生成溯源文件")

# ---------------------- 步骤11：保存处理后的数据 ----------------------
traffic_flows_west = traffic_flows_west.reset_index(drop=True)
traffic_flows_west.to_pickle(processed_path)
traffic_flows_west.to_csv(processed_path_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 处理后数据已保存至：\n   {processed_path}\n   {processed_path_csv}")
print(f"   最终数据行数：{len(traffic_flows_west)}")


# ---------------------- 步骤12：溯源记录分析 - 按要求输出删除信息 ----------------------
def analyze_trace_records_v2(trace_df):
    print("\n===== 溯源记录分析结果 - 车辆删除统计 =====")

    # 1. 校验输入
    if trace_df.empty:
        print("❌ 无溯源记录，无需分析")
        return

    # 2. 定义关键类型
    delete_veh_type = [t for t in trace_df["data_type"].unique() if "删除车辆" in t]  # 持续异常全删类型
    delete_row_type = [t for t in trace_df["data_type"].unique() if "删除行" in t and "删除车辆" not in t]  # 部分删除类型

    # 3. 提取持续异常全删的车辆ID（仅输出ID）
    delete_veh_ids = trace_df[trace_df["data_type"].isin(delete_veh_type)]["ID"].unique()
    if len(delete_veh_ids) > 0:
        print(f"\n1. 因持续偏移被全部删除的车辆ID列表：")
        ids = sorted(delete_veh_ids)
        for i in range(0, len(ids), 10):
            group = ids[i:i + 10]
            print(" ".join([f"车辆ID：{id}" for id in group]))
    else:
        print(f"\n1. 无因持续偏移被全部删除的车辆ID")

    # 4. 提取部分删除的车辆ID + 统计删除行数
    if len(delete_row_type) > 0:
        delete_row_detail = trace_df[trace_df["data_type"].isin(delete_row_type)].groupby("ID").agg(
            删除行数=("Frame", "count")
        ).reset_index()
        print(f"\n2. 部分删除的车辆ID及对应删除行数：")
        # 先收集所有行的字符串
        lines = []
        for _, row in delete_row_detail.iterrows():
            lines.append(f"   车辆ID {row['ID']}：删除 {row['删除行数']} 行")

        # 每10个一组，拼成一行输出
        for i in range(0, len(lines), 5):
            print(" ".join(lines[i:i + 5]))
    else:
        print(f"\n2. 无部分删除的车辆ID")


# 执行步骤12：按新要求分析溯源记录
analyze_trace_records_v2(trace_df)

# ---------------------- 内存清理 ----------------------
del traffic_flows_west, trace_records, trace_df
gc.collect()