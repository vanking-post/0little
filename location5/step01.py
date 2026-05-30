# step01.py
import pandas as pd
import numpy as np
import os
import gc
def split_traffic_data(pkl_path):
    """
    读取原始数据并根据车道拆分为东西向两个 DataFrame
    """
    east_lane_ids = [0, 1, 2, 3]
    west_lane_ids = [5, 6, 7, 8]
    K = 9

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ 找不到文件：{pkl_path}")

    print(f"正在从内存读取并拆分数据: {pkl_path}")

    # 1. 加载数据
    if pkl_path.endswith(".pkl"):
        traffic_flows = pd.read_pickle(pkl_path)
    else:
        traffic_flows = pd.read_csv(pkl_path, index_col=False, low_memory=False, encoding="gbk")

    # 2. 筛选与清洗列
    # 自东向西
    df_east = traffic_flows[traffic_flows.iloc[:, K].isin(east_lane_ids)].copy()
    df_east = df_east.drop(columns=[col for col in ['Direction', 'Traveled Dist.'] if col in df_east.columns])

    # 自西向东
    df_west = traffic_flows[traffic_flows.iloc[:, K].isin(west_lane_ids)].copy()
    df_west = df_west.drop(columns=[col for col in ['Direction', 'Traveled Dist.'] if col in df_west.columns])

    # 3. 列名标准化（替换空格和点）
    for df in [df_east, df_west]:
        df.columns = df.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")

    print(f"✅ 拆分完成：东向 {len(df_east)} 行，西向 {len(df_west)} 行")

    # --- 核心：返回处理好的对象，而不是存盘 ---
    return df_east, df_west


# 如果你想单独运行这个脚本测试，可以保留下面这段
# if __name__ == "__main__":
#     path = r"E:\0little\read\CQSkyEyedata5\location5\5_trajectory.pkl"
#     e_df, w_df = split_traffic_data(path)