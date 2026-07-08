import pandas as pd
import numpy as np
import gc

def complete_by_direction(df_clean_1, df_clean_2, df_raw_1, df_raw_2, label_1='1', label_2='2'):
    """
    对两个方向的清洗后数据进行时序补全（基于Frame连续性），
    优先从原始数据中恢复缺失帧，否则使用线性插值。

    参数:
        df_clean_1, df_clean_2: 清洗后的DataFrame（已排序）
        df_raw_1, df_raw_2: 对应的原始DataFrame（包含可能被清洗掉的帧）
        label_1, label_2: 方向标签，用于日志输出

    返回:
        df_complete_1, df_complete_2: 补全后的DataFrame
    """
    def process_direction(df_clean, df_raw, label):
        print(f"\n--- 正在处理方向 {label} 数据补全 (当前行数: {len(df_clean)}) ---")

        # 连续性检查
        def check_continuity(df):
            df_sorted = df.sort_values(['ID', 'Frame'])
            gaps = df_sorted.groupby('ID')['Frame'].diff()
            missing_frames = gaps[gaps > 1]
            return len(missing_frames), int(missing_frames.sum() - len(missing_frames)) if len(missing_frames) > 0 else 0

        gap_count, total_missing = check_continuity(df_clean)
        print(f"  [初检] 发现 {gap_count} 处断点，累计缺失 {total_missing} 帧")

        if gap_count == 0:
            print(f"  ✅ 方向 {label} 数据完整，无需补全")
            return df_clean

        all_filled_rows = []
        # 确保原始数据中存在 Time 列（假设已从 time 改为 Time）
        # 如果原始数据中没有 Time，则基于帧序号估算（0.04s间隔）
        has_time = 'Time' in df_clean.columns and 'Time' in df_raw.columns

        for vehicle_id, group in df_clean.groupby('ID'):
            group = group.sort_values('Frame')
            frames = group['Frame'].values
            times = group['Time'].values if has_time else None

            for i in range(len(frames) - 1):
                if frames[i+1] - frames[i] > 1:
                    prev_row = group.iloc[i]
                    next_row = group.iloc[i+1]

                    # 补全中间的所有帧
                    for m_frame in range(int(frames[i] + 1), int(frames[i+1])):
                        # 计算时间（优先使用插值，否则基于帧间隔0.04s）
                        if has_time:
                            ratio = (m_frame - frames[i]) / (frames[i+1] - frames[i])
                            m_time = round(prev_row['Time'] + ratio * (next_row['Time'] - prev_row['Time']), 2)
                        else:
                            m_time = round(prev_row['time'] + (m_frame - frames[i]) * 0.04, 2)  # 兼容旧列名

                        # 尝试从原始数据中找回
                        source_row = df_raw[(df_raw['ID'] == vehicle_id) & (df_raw['Frame'] == m_frame)]
                        if not source_row.empty:
                            new_row = source_row.iloc[0].to_dict()
                            all_filled_rows.append(new_row)
                        else:
                            # 线性插值补全
                            ratio = (m_frame - frames[i]) / (frames[i+1] - frames[i])
                            new_row = prev_row.to_dict()
                            new_row['Frame'] = m_frame
                            new_row['Time'] = m_time  # 使用插值时间
                            # 补全关键运动学列
                            for col in ['X', 'Y', 'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc', 'lat_Acc']:
                                if col in prev_row and col in next_row:
                                    new_row[col] = round(prev_row[col] + ratio * (next_row[col] - prev_row[col]), 2)
                            all_filled_rows.append(new_row)

        if all_filled_rows:
            df_filled = pd.DataFrame(all_filled_rows)
            # 统一列名格式（去除空格和点）
            df_filled.columns = df_filled.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")
            df_complete = pd.concat([df_clean, df_filled], ignore_index=True)
            df_complete = df_complete.sort_values(['ID', 'Frame']).reset_index(drop=True)
        else:
            df_complete = df_clean

        final_gap_count, final_total_missing = check_continuity(df_complete)
        if final_gap_count == 0:
            print(f"  ✅ 方向 {label} 补全成功，复检通过")
        else:
            print(f"  ⚠️ 方向 {label} 补全后仍残余 {final_gap_count} 处断点")

        return df_complete

    df_complete_1 = process_direction(df_clean_1, df_raw_1, label_1)
    df_complete_2 = process_direction(df_clean_2, df_raw_2, label_2)
    gc.collect()
    return df_complete_1, df_complete_2