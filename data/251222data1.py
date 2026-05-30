import pandas as pd
import numpy as np
import gc
import os

save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
east_pkl_path = os.path.join(save_dir, "traffic_flows_east.pkl")
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")
traffic_flows_east=pd.read_pickle(east_pkl_path)
traffic_flows_west=pd.read_pickle(west_pkl_path)

# ---------------------- 基础配置 ----------------------
MAX_VELOCITY = 144 # 120km/h 转 m/s
QUANTILE = 0.999
VALID_RATIO_THRESH = 0.9
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
trace_csv_path = os.path.join(save_dir, "traffic_flows_west_modify_trace1.csv")

# ---------------------- 初始化溯源记录容器 ----------------------
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


# ---------------------- 步骤2：X/Y坐标预处理（记录漂移修改+持续超出删除） ----------------------
def clean_xy_coords(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    x_min, x_max = df["X"].quantile(1 - QUANTILE), df["X"].quantile(QUANTILE)
    y_min, y_max = df["Y"].quantile(1 - QUANTILE), df["Y"].quantile(QUANTILE)
    print(f"X合理范围：[{x_min:.2f}, {x_max:.2f}]，Y合理范围：[{y_min:.2f}, {y_max:.2f}]")

    # 标记单点漂移 + 记录修改前值
    drift_xy_rows = pd.DataFrame()
    for coord in ["X", "Y"]:
        df[f"{coord}_prev"] = df.groupby("ID")[coord].shift(1)
        df[f"{coord}_next"] = df.groupby("ID")[coord].shift(-1)

        out_of_range = ~df[coord].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        prev_in_range = df[f"{coord}_prev"].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        next_in_range = df[f"{coord}_next"].between(x_min if coord == "X" else y_min, x_max if coord == "X" else y_max)
        df[f"{coord}_single_drift"] = out_of_range & prev_in_range & next_in_range

        # 收集当前坐标的漂移行（修改前）
        coord_drift = df[df[f"{coord}_single_drift"]].copy()
        coord_drift[f"{coord}_before"] = coord_drift[coord]
        # 先插值计算修改后值
        coord_drift[f"{coord}_after"] = coord_drift.groupby("ID")[coord].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        drift_xy_rows = pd.concat([drift_xy_rows, coord_drift])

    # 记录坐标漂移修改行
    if not drift_xy_rows.empty:
        drift_xy_rows = drift_xy_rows.drop_duplicates(subset=["ID", "Frame"])  # 去重（X/Y都漂移的行）
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
        # 加入溯源记录
        trace_list.append(drift_xy_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                         "Length", "Width", "X", "Y", "Velocity"]])

    # 执行X/Y插值（覆盖原值）
    for coord in ["X", "Y"]:
        df[coord] = df.groupby("ID")[coord].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    # 记录坐标持续超出删除行
    df["XY_valid"] = df["X"].between(x_min, x_max) & df["Y"].between(y_min, y_max)
    id_xy_valid_ratio = df.groupby("ID")["XY_valid"].mean()
    invalid_xy_ids = id_xy_valid_ratio[id_xy_valid_ratio < VALID_RATIO_THRESH].index
    delete_xy_rows = df[df["ID"].isin(invalid_xy_ids)].copy()

    if not delete_xy_rows.empty:
        delete_xy_rows["data_type"] = "坐标持续超出-删除"
        delete_xy_rows["modify_before"] = delete_xy_rows.apply(
            lambda x: f"X={x['X']:.2f}, Y={x['Y']:.2f}（超出范围X[{x_min:.2f},{x_max:.2f}], Y[{y_min:.2f},{y_max:.2f}]）",
            axis=1
        )
        delete_xy_rows["modify_after"] = None
        trace_list.append(delete_xy_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity"]])

    # 删除持续超出的车辆
    df = df[~df["ID"].isin(invalid_xy_ids)].reset_index(drop=True)

    # ---------------------- 修复：只删除存在的辅助列 ----------------------
    # 定义要删除的辅助列列表
    xy_aux_cols = [col for col in df.columns if
                   col.endswith(("_prev", "_next", "_single_drift", "_before", "_after", "XY_valid"))]
    # 只删除存在的列，避免KeyError
    df = df.drop(columns=xy_aux_cols)

    print(f"步骤2后行数：{len(df)}（删除坐标持续超出行：{len(delete_xy_rows)}，坐标漂移修改行：{len(drift_xy_rows)}）")
    del drift_xy_rows, delete_xy_rows
    return df


# 执行坐标清理 + 记录溯源
traffic_flows_west = clean_xy_coords(traffic_flows_west, trace_records)
gc.collect()


# ---------------------- 步骤3：Velocity预处理（记录漂移修改+持续偏离删除） ----------------------
def clean_velocity(df, trace_list):
    df = df.sort_values(by=["ID", "Frame"]).reset_index(drop=True)
    vel_min, vel_max = 0, MAX_VELOCITY
    print(f"Velocity合理范围：[{vel_min:.2f}, {vel_max:.2f}] m/s（对应0~120km/h）")

    # 标记单点漂移 + 记录修改前值
    df["Velocity_prev"] = df.groupby("ID")["Velocity"].shift(1)
    df["Velocity_next"] = df.groupby("ID")["Velocity"].shift(-1)

    vel_out_of_range = ~df["Velocity"].between(vel_min, vel_max)
    vel_prev_in_range = df["Velocity_prev"].between(vel_min, vel_max)
    vel_next_in_range = df["Velocity_next"].between(vel_min, vel_max)
    df["Velocity_single_drift"] = vel_out_of_range & vel_prev_in_range & vel_next_in_range

    # 收集速度漂移行（修改前+修改后）
    drift_vel_rows = df[df["Velocity_single_drift"]].copy()
    if not drift_vel_rows.empty:  # 只有存在漂移行时才创建before/after列
        drift_vel_rows["Velocity_before"] = drift_vel_rows["Velocity"]
        # 插值计算修改后值
        drift_vel_rows["Velocity_after"] = drift_vel_rows.groupby("ID")["Velocity"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

        # 记录速度漂移修改行
        drift_vel_rows["data_type"] = "速度单点漂移-修改"
        drift_vel_rows["modify_before"] = drift_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity_before']:.2f} m/s", axis=1
        )
        drift_vel_rows["modify_after"] = drift_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity_after']:.2f} m/s", axis=1
        )
        trace_list.append(drift_vel_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                          "Length", "Width", "X", "Y", "Velocity"]])

    # 执行速度插值（覆盖原值）
    df["Velocity"] = df.groupby("ID")["Velocity"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # 记录速度持续偏离删除行
    df["Velocity_valid"] = df["Velocity"].between(vel_min, vel_max)
    id_vel_valid_ratio = df.groupby("ID")["Velocity_valid"].mean()
    invalid_vel_ids = id_vel_valid_ratio[id_vel_valid_ratio < VALID_RATIO_THRESH].index
    delete_vel_rows = df[df["ID"].isin(invalid_vel_ids)].copy()

    if not delete_vel_rows.empty:
        delete_vel_rows["data_type"] = "速度持续偏离-删除"
        delete_vel_rows["modify_before"] = delete_vel_rows.apply(
            lambda x: f"Velocity={x['Velocity']:.2f} m/s（超出范围0~{vel_max:.2f}）", axis=1
        )
        delete_vel_rows["modify_after"] = None
        trace_list.append(delete_vel_rows[["ID", "Frame", "data_type", "modify_before", "modify_after",
                                           "Length", "Width", "X", "Y", "Velocity"]])

    # 删除持续偏离的车辆
    df = df[~df["ID"].isin(invalid_vel_ids)].reset_index(drop=True)

    # ---------------------- 修复：只删除存在的辅助列 ----------------------
    # 定义要删除的列列表
    vel_aux_cols = ["Velocity_prev", "Velocity_next", "Velocity_single_drift",
                    "Velocity_before", "Velocity_after", "Velocity_valid"]
    # 筛选出df中实际存在的列
    existing_vel_cols = [col for col in vel_aux_cols if col in df.columns]
    # 只删除存在的列
    df = df.drop(columns=existing_vel_cols)

    print(f"步骤3后行数：{len(df)}（删除速度持续偏离行：{len(delete_vel_rows)}，速度漂移修改行：{len(drift_vel_rows)}）")
    del drift_vel_rows, delete_vel_rows
    return df


# 执行速度清理 + 记录溯源
traffic_flows_west = clean_velocity(traffic_flows_west, trace_records)
gc.collect()

# ---------------------- 步骤4：合并溯源记录并写入CSV ----------------------
# 合并所有溯源记录
if trace_records:
    trace_df = pd.concat(trace_records, ignore_index=True)
    # 按数据类型+ID+Frame排序，方便查看
    trace_df = trace_df.sort_values(by=["data_type", "ID", "Frame"]).reset_index(drop=True)
    # 写入CSV（中文不乱码）
    trace_df.to_csv(trace_csv_path, index=False, encoding="gbk")
    print(f"\n✅ 数据溯源记录已写入：{trace_csv_path}")
    print(f"   溯源记录总行数：{len(trace_df)}")
    # 输出各类型数据统计
    print("\n📊 溯源记录统计：")
    print(trace_df["data_type"].value_counts())
else:
    print("\n⚠️ 无任何修改/删除的行，未生成溯源文件")

# ---------------------- 步骤5：保存处理后的数据 + 最终内存清理 ----------------------
processed_path = os.path.join(save_dir, "traffic_flows_west_processed1.pkl")
traffic_flows_west.to_pickle(processed_path)
print(f"\n✅ 处理后数据已保存至：{processed_path}")
print(f"   最终数据行数：{len(traffic_flows_west)}")

# 强制清理内存
del traffic_flows_west, trace_records
if 'trace_df' in locals():  # 避免trace_df未定义时报错
    del trace_df
gc.collect()