import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import gc

save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
east_pkl_path = os.path.join(save_dir, "traffic_flows_east.pkl")
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")
traffic_flows_east=pd.read_pickle(east_pkl_path)
traffic_flows_west=pd.read_pickle(west_pkl_path)

# 限速120km/h
MAX_VELOCITY = 144
VEL_MIN = 0
VEL_MAX = MAX_VELOCITY
# 坐标合理范围计算分位数（过滤0.1%极端值）
QUANTILE = 0.9999
# 有效数据占比阈值（低于该值判定为持续偏离/超出，删除车辆）
VALID_RATIO_THRESH = 0.9
X_MIN = 0  # 示例：X轴最小值
X_MAX = 325  # 示例：X轴最大值
Y_MIN = 20  # 示例：Y轴最小值
Y_MAX = 40  # 示例：Y轴最大值
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
# 溯源记录文件路径
trace_csv_path = os.path.join(save_dir, "traffic_flows_west_modify_trace.csv")

# ---------------------- 初始化溯源记录容器 ----------------------
# 用于收集所有修改/删除的行
trace_records = []

# ---------------------- 步骤1：删除Length<2且Width<1的行（记录删除行） ----------------------
print(f"原始数据行数：{len(traffic_flows_west)}")

# 1. 筛选要删除的行（长宽不符合）
delete_size_rows = traffic_flows_west[
    (traffic_flows_west["Length"] < 2) & (traffic_flows_west["Width"] < 1)
    ].copy()
# 2. 给删除行加标识
delete_size_rows["data_type"] = "长宽不符合-删除"
delete_size_rows["modify_before"] = delete_size_rows.apply(
    lambda x: f"Length={x['Length']}, Width={x['Width']}", axis=1
)
delete_size_rows["modify_after"] = None  # 删除无修改后值
# 3. 加入溯源记录
trace_records.append(delete_size_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                       "Length", "Width", "X", "Y", "Velocity"]])

# 4. 执行删除
traffic_flows_west = traffic_flows_west[
    ~((traffic_flows_west["Length"] < 2) & (traffic_flows_west["Width"] < 1))
].reset_index(drop=True)

print(f"步骤1后行数：{len(traffic_flows_west)}（删除长宽不符合行：{len(delete_size_rows)}）")
del delete_size_rows
gc.collect()

# ---------------------- 步骤2：速度预处理（核心优化：单位km/h + 严格筛选 + 删负速度） ----------------
def clean_velocity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print(f"\n速度合理范围：[{VEL_MIN}, {VEL_MAX}] km/h（<0直接删除）")

    # 子步骤1：删除速度<0的行，记录溯源
    delete_vel_neg_rows = df[df["Velocity"] < VEL_MIN].copy()
    if not delete_vel_neg_rows.empty:
        delete_vel_neg_rows["data_type"] = "速度<0-直接删除"
        delete_vel_neg_rows["modify_before"] = delete_vel_neg_rows.apply(
            lambda x: f"Velocity={x['Velocity']:.2f} km/h（<{VEL_MIN}）", axis=1
        )
        delete_vel_neg_rows["modify_after"] = None
        trace_list.append(delete_vel_neg_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                               "Length", "Width", "X", "Y", "Velocity"]])
        # 执行删除
        df = df[df["Velocity"] >= VEL_MIN].reset_index(drop=True)
    print(f"  删速度<0行：{len(delete_vel_neg_rows)}，剩余行数：{len(df)}")

    # 子步骤2：标记单点漂移（严格判断：当前超出0-120，前后帧都在0-120内）
    df["Velocity_prev"] = df.groupby("ID")["Velocity"].shift(1)  # 前一帧
    df["Velocity_next"] = df.groupby("ID")["Velocity"].shift(-1)  # 后一帧

    # 定义：当前帧超出合理范围
    vel_out_of_range = ~df["Velocity"].between(VEL_MIN, VEL_MAX)
    # 前一帧在合理范围（且非空）
    vel_prev_in_range = df["Velocity_prev"].between(VEL_MIN, VEL_MAX) & df["Velocity_prev"].notna()
    # 后一帧在合理范围（且非空）
    vel_next_in_range = df["Velocity_next"].between(VEL_MIN, VEL_MAX) & df["Velocity_next"].notna()
    # 单点漂移：仅当前超出，前后都在范围内（且有值）
    df["Velocity_single_drift"] = vel_out_of_range & vel_prev_in_range & vel_next_in_range

    # 子步骤3：处理单点漂移（插值修改，仅记录有实际变化的行）
    drift_vel_rows = df[df["Velocity_single_drift"]].copy()
    if not drift_vel_rows.empty:
        # 保存修改前值
        drift_vel_rows["Velocity_before"] = drift_vel_rows["Velocity"]
        # 按ID分组线性插值（仅修改漂移行）
        drift_vel_rows["Velocity_after"] = drift_vel_rows.groupby("ID")["Velocity"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        # 过滤：仅保留修改后值有变化的行（避免-0.1改-0.1这类无意义记录）
        drift_vel_rows = drift_vel_rows[
            abs(drift_vel_rows["Velocity_before"] - drift_vel_rows["Velocity_after"]) > 1e-6]

        if not drift_vel_rows.empty:
            # 标记溯源信息
            drift_vel_rows["data_type"] = "速度单点漂移-插值修改"
            drift_vel_rows["modify_before"] = drift_vel_rows.apply(
                lambda x: f"Velocity={x['Velocity_before']:.2f} km/h（超出{VEL_MIN}-{VEL_MAX}）", axis=1
            )
            drift_vel_rows["modify_after"] = drift_vel_rows.apply(
                lambda x: f"Velocity={x['Velocity_after']:.2f} km/h（插值修正）", axis=1
            )
            trace_list.append(drift_vel_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                              "Length", "Width", "X", "Y", "Velocity"]])
            # 覆盖原df的漂移行速度值
            df.loc[drift_vel_rows.index, "Velocity"] = drift_vel_rows["Velocity_after"].values
    print(f"  速度单点漂移需修改行：{len(drift_vel_rows)}")

    # 子步骤4：判断持续偏离（有效速度行占比<90%）
    df["Velocity_valid"] = df["Velocity"].between(VEL_MIN, VEL_MAX)
    id_vel_valid_ratio = df.groupby("ID")["Velocity_valid"].mean()
    invalid_vel_ids = id_vel_valid_ratio[id_vel_valid_ratio < VALID_RATIO_THRESH].index
    # 记录并删除持续偏离的车辆
    delete_vel_cont_rows = df[df["ID"].isin(invalid_vel_ids)].copy()
    if not delete_vel_cont_rows.empty:
        delete_vel_cont_rows["data_type"] = "速度持续偏离-删除车辆"
        delete_vel_cont_rows["modify_before"] = delete_vel_cont_rows.apply(
            lambda x: f"Velocity={x['Velocity']:.2f} km/h（车辆有效速度占比<90%）", axis=1
        )
        delete_vel_cont_rows["modify_after"] = None
        trace_list.append(delete_vel_cont_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                                "Length", "Width", "X", "Y", "Velocity"]])
        # 执行删除
        df = df[~df["ID"].isin(invalid_vel_ids)].reset_index(drop=True)
    print(f"  删速度持续偏离车辆行：{len(delete_vel_cont_rows)}，剩余行数：{len(df)}")

    # 清理辅助列（仅删除存在的列）
    vel_aux_cols = ["Velocity_prev", "Velocity_next", "Velocity_single_drift", "Velocity_valid"]
    existing_vel_cols = [col for col in vel_aux_cols if col in df.columns]
    df = df.drop(columns=existing_vel_cols)

    return df


# 执行速度清理
traffic_flows_west = clean_velocity(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤3：坐标预处理（核心优化：手动指定范围 + 严格漂移判断） ----------------------
def clean_xy_coords(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    print(f"\n坐标合理范围：X[{X_MIN}, {X_MAX}]，Y[{Y_MIN}, {Y_MAX}]（手动指定）")

    # 子步骤1：标记单点漂移（严格判断：当前超出手动范围，前后帧都在范围内）
    for coord in ["X", "Y"]:
        df[f"{coord}_prev"] = df.groupby("ID")[coord].shift(1)  # 前一帧
        df[f"{coord}_next"] = df.groupby("ID")[coord].shift(-1)  # 后一帧

        # 定义：当前帧超出手动范围
        coord_out_of_range = ~df[coord].between(X_MIN if coord == "X" else Y_MIN, X_MAX if coord == "X" else Y_MAX)
        # 前一帧在范围（且非空）
        coord_prev_in_range = df[f"{coord}_prev"].between(X_MIN if coord == "X" else Y_MIN,
                                                          X_MAX if coord == "X" else Y_MAX) & df[
                                  f"{coord}_prev"].notna()
        # 后一帧在范围（且非空）
        coord_next_in_range = df[f"{coord}_next"].between(X_MIN if coord == "X" else Y_MIN,
                                                          X_MAX if coord == "X" else Y_MAX) & df[
                                  f"{coord}_next"].notna()
        # 单点漂移标记
        df[f"{coord}_single_drift"] = coord_out_of_range & coord_prev_in_range & coord_next_in_range

    # 子步骤2：处理单点漂移（插值修改，仅记录有变化的行）
    drift_xy_rows = pd.DataFrame()
    for coord in ["X", "Y"]:
        coord_drift = df[df[f"{coord}_single_drift"]].copy()
        if not coord_drift.empty:
            coord_drift[f"{coord}_before"] = coord_drift[coord]
            # 插值计算修改后值
            coord_drift[f"{coord}_after"] = coord_drift.groupby("ID")[coord].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
            # 过滤：仅保留值有变化的行
            coord_drift = coord_drift[abs(coord_drift[f"{coord}_before"] - coord_drift[f"{coord}_after"]) > 1e-6]
            drift_xy_rows = pd.concat([drift_xy_rows, coord_drift])

    # 记录坐标漂移修改行（去重）
    if not drift_xy_rows.empty:
        drift_xy_rows = drift_xy_rows.drop_duplicates(subset=["ID", "Frame"])
        drift_xy_rows["data_type"] = "坐标单点漂移-插值修改"
        drift_xy_rows["modify_before"] = drift_xy_rows.apply(
            lambda x: f"X={x['X_before']:.2f}, Y={x['Y_before']:.2f}（超出手动范围）", axis=1
        )
        drift_xy_rows["modify_after"] = drift_xy_rows.apply(
            lambda x: f"X={x['X_after']:.2f}, Y={x['Y_after']:.2f}（插值修正）", axis=1
        )
        trace_list.append(drift_xy_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                         "Length", "Width", "X", "Y", "Velocity"]])
        # 覆盖原df的漂移行坐标值
        for coord in ["X", "Y"]:
            if f"{coord}_after" in drift_xy_rows.columns:
                df.loc[drift_xy_rows.index, coord] = drift_xy_rows[f"{coord}_after"].values
    print(f"  坐标单点漂移需修改行：{len(drift_xy_rows)}")

    # 子步骤3：判断持续偏离（有效坐标行占比<90%）
    df["XY_valid"] = df["X"].between(X_MIN, X_MAX) & df["Y"].between(Y_MIN, Y_MAX)
    id_xy_valid_ratio = df.groupby("ID")["XY_valid"].mean()
    invalid_xy_ids = id_xy_valid_ratio[id_xy_valid_ratio < VALID_RATIO_THRESH].index
    # 记录并删除持续偏离的车辆
    delete_xy_cont_rows = df[df["ID"].isin(invalid_xy_ids)].copy()
    if not delete_xy_cont_rows.empty:
        delete_xy_cont_rows["data_type"] = "坐标持续偏离-删除车辆"
        delete_xy_cont_rows["modify_before"] = delete_xy_cont_rows.apply(
            lambda x: f"X={x['X']:.2f}, Y={x['Y']:.2f}（车辆有效坐标占比<90%）", axis=1
        )
        delete_xy_cont_rows["modify_after"] = None
        trace_list.append(delete_xy_cont_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                               "Length", "Width", "X", "Y", "Velocity"]])
        # 执行删除
        df = df[~df["ID"].isin(invalid_xy_ids)].reset_index(drop=True)
    print(f"  删坐标持续偏离车辆行：{len(delete_xy_cont_rows)}，剩余行数：{len(df)}")

    # 清理辅助列（仅删除存在的列）
    xy_aux_cols = [col for col in df.columns if
                   col.endswith(("_prev", "_next", "_single_drift", "_before", "_after", "XY_valid"))]
    df = df.drop(columns=xy_aux_cols)

    return df


# 执行坐标清理
traffic_flows_west = clean_xy_coords(traffic_flows_west, trace_records)
gc.collect()

# ---------------------- 步骤4：合并溯源记录并写入CSV ----------------------
if trace_records:
    trace_df = pd.concat(trace_records, ignore_index=True)
    trace_df = trace_df.sort_values(by=["data_type", "ID", "Frame"]).reset_index(drop=True)
    # 写入CSV（中文不乱码）
    trace_df.to_csv(trace_csv_path, index=False, encoding="gbk")
    print(f"\n✅ 溯源记录已写入：{trace_csv_path}")
    print(f"   溯源记录总行数：{len(trace_df)}")
    print("\n📊 溯源记录统计：")
    print(trace_df["data_type"].value_counts())
else:
    print("\n⚠️ 无任何修改/删除的行，未生成溯源文件")

# ---------------------- 步骤5：保存处理后的数据 + 内存清理 ----------------------
processed_path = os.path.join(save_dir, "traffic_flows_west_processed.pkl")
traffic_flows_west.to_pickle(processed_path)
print(f"\n✅ 处理后数据已保存至：{processed_path}")
print(f"   最终数据行数：{len(traffic_flows_west)}")

# 强制清理内存
del traffic_flows_west, trace_records
if 'trace_df' in locals():
    del trace_df
gc.collect()