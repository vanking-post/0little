#将数据根据所处车道的区别进行分类
#分成东向和西向的两个文件
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import gc

csv_path = r"E:\0little\read\CQSkyEyedata5\location5\5_trajectory.csv"
pkl_path = r"E:\0little\read\CQSkyEyedata5\location5\5_trajectory.pkl"
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
#traffic_flows_data=pd.read_csv(csv_path,index_col=0,low_memory=False,encoding="utf-8")
#traffic_flows_data.to_pickle(pkl_path)
traffic_flows=pd.read_pickle(pkl_path)

print(len(traffic_flows))

east_lane_ids = [0, 1, 2, 3]
west_lane_ids = [5, 6, 7, 8]
K=9
traffic_flows_east=[]
traffic_flows_west =[]

def load_traffic_data(path):
    """读取交通流数据（兼容PKL/CSV格式）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到文件：{path}")

    # 识别文件格式并读取
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, index_col=False, low_memory=False, encoding="gbk")  # 中文用gbk
    else:
        raise ValueError("❌ 仅支持.pkl和.csv格式")

    # 验证K列是否存在（避免索引越界）
    if K >= len(df.columns):
        raise IndexError(f"❌ 列索引K={K}超出范围！数据仅有{len(df.columns)}列（索引0~{len(df.columns) - 1}）")

    # 获取LaneID列名（方便后续打印，可选）
    lane_col_name = df.columns[K]
    print(f"✅ 原始数据读取成功！共{len(df)}行，LaneID列名：{lane_col_name}")
    return df


# 读取数据到traffic_flows
traffic_flows = load_traffic_data(pkl_path)
original_rows = len(traffic_flows)  # 记录原始行数

# 筛选自东向西（LaneID=0/1/2/3）的数据，清除direction、Traveldist列数据
traffic_flows_east = traffic_flows[traffic_flows.iloc[:, K].isin(east_lane_ids)]
traffic_flows_east = traffic_flows_east.drop(columns=['Direction','Traveled Dist.'])
# 筛选自西向东（LaneID=5/6/7/8）的数据
traffic_flows_west = traffic_flows[traffic_flows.iloc[:, K].isin(west_lane_ids)]
traffic_flows_west = traffic_flows_west.drop(columns=['Direction','Traveled Dist.'])
#去掉空格、点替换为下划线
traffic_flows_west.columns = traffic_flows_west.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")
traffic_flows_east.columns = traffic_flows_east.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")
print("删除、替换空格与点后列名：", traffic_flows_east.columns.tolist())
# 输出两个DataFrame的行数
east_rows = len(traffic_flows_east)
west_rows = len(traffic_flows_west)
print(f"\n📊 拆分结果：")
print(f"自东向西（LaneID=0/1/2/3）数据行数：{east_rows}")
print(f"自西向东（LaneID=5/6/7/8）数据行数：{west_rows}")

# 验证行数是否匹配（注意：如果有LaneID=4/9等其他值，总和会小于原始行数）
total_split_rows = east_rows + west_rows
if total_split_rows == original_rows:
    print(f"✅ 行数验证通过！拆分后总行数({total_split_rows}) = 原始行数({original_rows})")
else:
    print(f"⚠️ 行数验证未通过！拆分后总行数({total_split_rows}) ≠ 原始行数({original_rows})")
    print(f"   差异行数：{original_rows - total_split_rows}（可能包含LaneID=4/9等未分类数据）")


#新建路径并保存东西向的交通流
east_pkl_path = os.path.join(save_dir, "traffic_flows_east.pkl")
west_pkl_path = os.path.join(save_dir, "traffic_flows_west.pkl")
east_csv_path = os.path.join(save_dir, "traffic_flows_east.csv")
west_csv_path = os.path.join(save_dir, "traffic_flows_west.csv")
# 保存自东向西数据
traffic_flows_east.to_pickle(east_pkl_path)
print(f"\n✅ 自东向西数据已保存：{east_pkl_path}")
# 保存自西向东数据
traffic_flows_west.to_pickle(west_pkl_path)
print(f"✅ 自西向东数据已保存：{west_pkl_path}")
#保存csv格式文件
traffic_flows_east.to_csv(
    east_csv_path,
    index=False,       # 不保存行索引（避免冗余列）
    encoding="gbk",    # 中文不乱码（Excel打开CSV必选）
    chunksize=100000   # 分块保存，避免内存溢出（百万行必备）
)
print(f"\n✅ 自东向西CSV已保存：{east_csv_path}")
traffic_flows_west.to_csv(
    west_csv_path,
    index=False,
    encoding="gbk",
    chunksize=100000
)
print(f"✅ 自西向东CSV已保存：{west_csv_path}")
del traffic_flows_east,traffic_flows_west,traffic_flows
gc.collect()