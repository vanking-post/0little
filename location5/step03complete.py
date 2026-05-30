import pandas as pd
import numpy as np
import gc


def data_complete(df_east_clean, df_west_clean, df_east_raw, df_west_raw):
    """
    对清洗后的东西向数据进行连续性检验、补全及复检
    """

    def process_direction(df_clean, df_raw, direction_label):
        print(f"\n--- 正在处理 {direction_label} 向数据补全 (当前行数: {len(df_clean)}) ---")

        # 1. 连续性初检
        def check_continuity(df):
            df_sorted = df.sort_values(['ID', 'Frame'])
            gaps = df_sorted.groupby('ID')['Frame'].diff()
            missing_frames = gaps[gaps > 1]
            return len(missing_frames), int(missing_frames.sum() - len(missing_frames)) if len(
                missing_frames) > 0 else 0

        gap_count, total_missing = check_continuity(df_clean)
        print(f"  [初检] 发现 {gap_count} 处断点，累计缺失 {total_missing} 帧")

        if gap_count == 0:
            print(f"  ✅ {direction_label} 向数据完整，无需补全")
            return df_clean

        # 2. 执行补全逻辑 (集成 buquan0 和 buquan1)
        # 为了提升速度，我们先识别出所有需要补全的 (ID, Frame)
        all_filled_rows = []

        for vehicle_id, group in df_clean.groupby('ID'):
            group = group.sort_values('Frame')
            frames = group['Frame'].values

            for i in range(len(frames) - 1):
                if frames[i + 1] - frames[i] > 1:
                    prev_row = group.iloc[i]
                    next_row = group.iloc[i + 1]

                    # 准备补全这些帧
                    for m_frame in range(int(frames[i] + 1), int(frames[i + 1])):
                        m_time = round(prev_row['time'] + (m_frame - frames[i]) * 0.04, 2)

                        # --- buquan0: 尝试从原始数据找回 ---
                        source_row = df_raw[(df_raw['ID'] == vehicle_id) & (df_raw['Frame'] == m_frame)]

                        if not source_row.empty:
                            new_row = source_row.iloc[0].to_dict()
                            # 补全可能在清洗阶段被删掉但在原始数据中存在的列
                            # (确保列名与 df_clean 一致)
                            all_filled_rows.append(new_row)
                        else:
                            # --- buquan1: 线性插值补全 ---
                            ratio = (m_frame - frames[i]) / (frames[i + 1] - frames[i])
                            new_row = prev_row.to_dict()
                            new_row['Frame'] = m_frame
                            new_row['time'] = m_time
                            # 坐标与运动学指标插值
                            for col in ['X', 'Y', 'Velocity', 'Acceleration', 'long_Vel', 'lat_Vel', 'long_Acc',
                                        'lat_Acc']:
                                if col in prev_row:
                                    new_row[col] = round(prev_row[col] + ratio * (next_row[col] - prev_row[col]), 2)
                            all_filled_rows.append(new_row)

        # 3. 合并并重新排序
        if all_filled_rows:
            df_filled = pd.DataFrame(all_filled_rows)
            # 统一列名格式
            df_filled.columns = df_filled.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")
            df_complete = pd.concat([df_clean, df_filled], ignore_index=True)
            df_complete = df_complete.sort_values(['ID', 'Frame']).reset_index(drop=True)
        else:
            df_complete = df_clean

        # 4. 补全后复检
        final_gap_count, final_total_missing = check_continuity(df_complete)
        if final_gap_count == 0:
            print(f"  ✅ {direction_label} 向补全成功，复检通过")
        else:
            print(f"  ⚠️ {direction_label} 向补全后仍残余 {final_gap_count} 处断点")

        return df_complete

    # 分别处理东西向
    df_east_final = process_direction(df_east_clean, df_east_raw, "东")
    df_west_final = process_direction(df_west_clean, df_west_raw, "西")

    gc.collect()
    return df_east_final, df_west_final