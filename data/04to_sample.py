#实现数据上下文的可视化；将车辆轨迹数据中“周围车辆的 ID”转换为了“周围车辆与自车的欧几里得距离”，
# 从而为后续的模型训练（比如你之前训练的 LSTM 模型）提供更有意义的物理特征。
#读取traffic_flows_east/west,traffic_flows_complete0的数据，输出traffic_flows_sample
import pandas as pd
import numpy as np
import os
import time

# 样本化处理六个表头
# LeftBehindID	LeftSideID	LeftFrontID	BehindID	EgoVehicleID	FrontID	RightBehindID	RightSideID	RightFrontID
# 数据路径↓
# save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
# files = [
#     ("Raw", "traffic_flows_west.pkl"),  # 源数据库
#     ("Complete", "traffic_flows_complete0.pkl"),  # 补全数据库
#     ("sample", "traffic_flows_sample.pkl")]  # 样本化数据库

# 数据路径东向↓
save_dir = r"E:\0little\read\CQSkyEyedata5\location5e"
files = [
    ("Raw", "traffic_flows_east.pkl"),  # 源数据库
    ("Complete", "traffic_flows_complete0.pkl"),  # 补全数据库
    ("sample", "traffic_flows_sample.pkl")]  # 样本化数据库

def convert_ids_to_distances(complete_df, raw_df):
    """
    将ID转换为距离（修复行数问题）
    """
    print("开始进行ID到距离的转换...")

    # 修复：确保索引连续，避免行数问题
    complete_df = complete_df.reset_index(drop=True).copy()

    # 定义需要转换的列和对应的新的距离列名
    id_columns = ['LeftBehindID', 'LeftSideID', 'LeftFrontID', 'BehindID',
                  'RightBehindID', 'RightSideID', 'RightFrontID']
    dist_columns = ['LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist',
                    'RB_Dist', 'RS_Dist', 'RF_Dist']

    # 创建新DataFrame存储结果（确保索引连续）
    result_df = complete_df.copy()

    # 为新距离列初始化（使用0而不是NaN）
    for col in dist_columns:
        result_df[col] = 0.0  # 初始化为0

    # 按行处理（使用位置索引）
    total_rows = len(result_df)
    for idx in range(total_rows):
        if idx % 1000 == 0:  # 显示进度
            print(f"处理进度: {idx}/{total_rows} ({idx / total_rows * 100:.1f}%)")

        current_row = result_df.iloc[idx]
        current_time = current_row['time']  # 使用正确的列名 'time'
        current_ego_id = int(current_row['ID'])  # 使用正确的列名 'ID'
        current_x = current_row['X']  # 当前车辆X坐标
        current_y = current_row['Y']  # 当前车辆Y坐标

        # 为每一列ID计算距离
        for id_col, dist_col in zip(id_columns, dist_columns):
            vehicle_id = current_row.get(id_col, 0)

            # 如果ID为0或NaN，距离设为0
            if pd.isna(vehicle_id) or vehicle_id == 0:
                result_df.at[idx, dist_col] = 0.0
                continue

            vehicle_id = int(vehicle_id)

            # 在原始数据中查找对应时间和ID的车辆坐标
            time_mask = raw_df['time'] == current_time
            same_time_data = raw_df[time_mask]

            # 在同一时间点中查找目标车辆ID
            target_vehicle = same_time_data[same_time_data['ID'] == vehicle_id]

            if len(target_vehicle) > 0:
                # 取第一个匹配项
                target_x = target_vehicle.iloc[0]['X']
                target_y = target_vehicle.iloc[0]['Y']

                # 计算欧几里得距离
                distance = np.sqrt((current_x - target_x) ** 2 + (current_y - target_y) ** 2)

                # 保留3位小数
                result_df.at[idx, dist_col] = round(distance, 3)
            else:
                # 如果找不到对应车辆，距离设为0
                result_df.at[idx, dist_col] = 0.0

    print("ID到距离转换完成！")
    return result_df


def main():
    start_time = time.time()
    print(f"程序开始运行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # 构建文件路径
    raw_path = os.path.join(save_dir, files[0][1])  # traffic_flows_west.pkl
    complete_path = os.path.join(save_dir, files[1][1])  # traffic_flows_complete0.pkl
    sample_pkl_path = os.path.join(save_dir, files[2][1])  # traffic_flows_sample.pkl
    sample_csv_path = os.path.join(save_dir, "traffic_flows_sample.csv")  # CSV格式

    print("正在读取原始数据...")
    try:
        raw_df = pd.read_pickle(raw_path)
        print(f"原始数据形状: {raw_df.shape}")
    except Exception as e:
        print(f"读取原始数据失败: {e}")
        return

    print("正在读取补全数据...")
    try:
        complete_df = pd.read_pickle(complete_path)
        print(f"补全数据形状: {complete_df.shape}")
    except Exception as e:
        print(f"读取补全数据失败: {e}")
        return

    # 检查必要的列是否存在
    required_cols = ['LeftBehindID', 'LeftSideID', 'LeftFrontID', 'BehindID',
                     'RightBehindID', 'RightSideID', 'RightFrontID',
                     'ID', 'X', 'Y', 'time']

    missing_cols = [col for col in required_cols if col not in complete_df.columns]
    if missing_cols:
        print(f"警告：以下列在补全数据中不存在: {missing_cols}")
        print(f"现有列名: {list(complete_df.columns)}")
        return

    print("开始转换过程...")
    # 执行ID到距离的转换（修复行数问题）
    sampled_df = convert_ids_to_distances(complete_df, raw_df)

    # 确保行数匹配
    if len(sampled_df) != len(complete_df):
        print(f"警告：行数不一致！原始数据行数: {len(complete_df)}, 转换后行数: {len(sampled_df)}")
        print("正在修复行数...")
        # 修复：确保行数一致（取原始行数）
        sampled_df = sampled_df.iloc[:len(complete_df)].copy()
        print(f"行数已修复为: {len(sampled_df)}")

    # 删除原始的ID列
    id_columns_to_remove = ['LeftBehindID', 'LeftSideID', 'LeftFrontID', 'BehindID',
                            'EgoVehicleID', 'FrontID', 'RightBehindID', 'RightSideID', 'RightFrontID']

    # 删除Original_TTC列
    additional_columns_to_remove = ['Original_TTC']

    # 合并所有要删除的列
    all_columns_to_remove = id_columns_to_remove + additional_columns_to_remove

    # 只删除存在的列
    existing_columns_to_remove = [col for col in all_columns_to_remove if col in sampled_df.columns]

    if existing_columns_to_remove:
        print(f"将要删除的列: {existing_columns_to_remove}")
        columns_to_keep = [col for col in sampled_df.columns if col not in existing_columns_to_remove]
        sampled_df = sampled_df[columns_to_keep]
        print(f"删除列后数据形状: {sampled_df.shape}")
    else:
        print("没有需要删除的列")

    print(f"最终数据形状: {sampled_df.shape}")
    print(f"最终保留的列: {list(sampled_df.columns)}")
    print(f"新增的距离列: {['LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RS_Dist', 'RF_Dist']}")

    # 保存处理后的数据
    print("正在保存样本化数据...")

    # 保存为pickle格式
    try:
        sampled_df.to_pickle(sample_pkl_path)
        print(f"Pickle文件已保存至: {sample_pkl_path}")
    except Exception as e:
        print(f"保存Pickle文件失败: {e}")
        return

    # 保存为CSV格式
    try:
        sampled_df.to_csv(sample_csv_path, index=False)
        print(f"CSV文件已保存至: {sample_csv_path}")
    except Exception as e:
        print(f"保存CSV文件失败: {e}")
        return

    # 显示一些统计信息
    print("\n距离特征统计信息:")
    dist_cols = ['LB_Dist', 'LS_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RS_Dist', 'RF_Dist']
    for col in dist_cols:
        if col in sampled_df.columns:
            # 计算非零值的数量
            non_zero_count = (sampled_df[col] != 0).sum()
            zero_count = (sampled_df[col] == 0).sum()
            mean_dist = sampled_df[col].mean()
            min_dist = sampled_df[col].min()
            max_dist = sampled_df[col].max()

            print(f"{col}: 总数 {len(sampled_df)}, 零值 {zero_count}, 非零值 {non_zero_count}, "
                  f"平均距离 {mean_dist:.3f}, 最小值 {min_dist:.3f}, 最大值 {max_dist:.3f}")

    # 显示处理后的数据预览
    print("\n处理后数据预览（前5行）:")
    print(sampled_df.head())

    # 检查是否有空值
    null_counts = sampled_df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n警告：存在空值的列及数量:")
        print(null_counts[null_counts > 0])
    else:
        print("\n检查完成：数据中无空值")

    # 计算并显示运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"\n程序结束运行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")

    print("\n样本化处理完成！")


if __name__ == "__main__":
    main()