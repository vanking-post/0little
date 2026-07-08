# step01.py
import pandas as pd
def split_by_direction(file_path):
    """
    读取 Excel 文件，根据 Direction 列的值（0 和 1）拆分为两个 DataFrame

    参数:
        file_path (str): Excel 文件路径
    返回:
        df_dir0, df_dir1: Direction == 0 和 Direction == 1 的 DataFrame
    """
    # 读取文件（兼容 utf-8-sig 和 gbk）
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')

    # 检查 Direction 列是否存在
    if 'Direction' not in df.columns:
        raise KeyError("数据中未找到 'Direction' 列")

    # 修复 XLSX 浮点精度误差: Frame→整数, Time→2位小数
    df['Frame'] = df['Frame'].round().astype(int)
    if 'Time' in df.columns:
        df['Time'] = df['Time'].round(2)

    # 分流
    df_dir2 = df[df['Direction'] == 2].copy()
    df_dir1 = df[df['Direction'] == 1].copy()
    for df in [df_dir2, df_dir1]:
        df.columns = df.columns.str.replace(r"[\.\s]+", "_", regex=True).str.strip().str.strip("_")
    print(f"Direction=2 的行数: {len(df_dir2)}")
    print(f"Direction=1 的行数: {len(df_dir1)}")

    return df_dir1,df_dir2
# 使用示例（单独运行时测试）
if __name__ == "__main__":
    file_path_1 = r"E:\0little\read\location1\location1\1-1_trajectory.csv"
    file_path_2 = r"E:\0little\read\location1\location1\1-2_trajectory.csv"
    df1_1, df1_2 = split_by_direction(file_path_1)
    df2_1, df2_2 = split_by_direction(file_path_2)