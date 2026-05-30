#读取traffic_flows_guiyi数据，输出traffic_flows_sampling
#从该数据guiyi中截取出可供模型训练使用的样本数据，截取的思路为变道点前100到50帧的行驶数据
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

# --- 阈值参数控制 ---
THRESHOLD_LEFT_RIGHT_CHANGE = 100
THRESHOLD_SECONDARY_FIRST_CHANGE = 100
THRESHOLD_SECONDARY_SECOND_CHANGE = 50
THRESHOLD_RAMP_FIRST_CHANGE = 100
THRESHOLD_RAMP_SUBSEQUENT_CHANGE = 50

# --- 数据路径 ---
save_dir = r"E:\0little\read\CQSkyEyedata5\location5"
file_path = os.path.join(save_dir, "traffic_flows_guiyi.pkl")

# 读取数据
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    df = pd.read_pickle(file_path)
    # 删除指定列
    df = df.drop(columns=[col for col in ['Class', 'Length', 'Width'] if col in df.columns])
    df['ID'] = df['ID'].astype(int)
    df['LaneID'] = df['LaneID'].astype(int)

    # 排序
    df_sorted = df.sort_values(['ID', 'time']).reset_index(drop=True)

    # --- 容器初始化 ---
    ramp_categories = {k: [] for k in ['匝道8汇入7并直行', '匝道8汇入7后汇入6', '匝道8汇入7后汇入6后汇入5',
                                       '匝道8直接汇入6', '匝道8直接汇入5', '匝道8汇入6后汇入5']}
    maneuver_counts = defaultdict(int)

    # 来源统计计数器 (核心新增)
    source_sample_counts = {
        '左变道': {'普通变道': 0, '二次变道': 0, '匝道驶入后续变道': 0},  # 将'匝道驶入变道'更名为更准确的'匝道驶入后续变道'
        '右变道': {'普通变道': 0, '二次变道': 0}
    }
    sampling_data = []

    print("=== 开始处理：分类、过滤与采样合并执行 ===")

    # 使用 groupby 优化遍历
    for vehicle_id, group in df_sorted.groupby('ID'):
        lanes = group['LaneID'].values
        if len(lanes) < 2: continue

        # 快速去重算法获取车道序列及变道点索引
        change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
        unique_seq = [lanes[0]] + lanes[change_idx].tolist()

        # 检查 unique_seq 长度 ---
        if len(unique_seq) < 2:
            # 如果是主路车(5,6,7)且没变道，归类为跟驰
            if unique_seq[0] in [5, 6, 7]:
                maneuver_counts['跟驰'] += 1
                # 执行跟驰采样逻辑...
                if len(group) >= 50:
                    s = random.randint(0, len(group) - 50)
                    sample = group.iloc[s:s + 50].copy()
                    sample['Label'] = '跟驰'
                    sampling_data.append(sample)
            # 如果是匝道车(8)且没变道，直接跳过
            continue

        # 1. 匝道驶入逻辑 (起始车道为8)
        if unique_seq[0] == 8:
            category = None
            if unique_seq[1] == 7:
                if len(unique_seq) == 2:
                    category = '匝道8汇入7并直行'
                elif len(unique_seq) > 2 and unique_seq[2] == 6:
                    category = '匝道8汇入7后汇入6后汇入5' if (
                            len(unique_seq) > 3 and unique_seq[3] == 5) else '匝道8汇入7后汇入6'
                else:
                    category = '匝道8汇入7并直行'
            elif unique_seq[1] == 6:
                category = '匝道8汇入6后汇入5' if (len(unique_seq) > 2 and unique_seq[2] == 5) else '匝道8直接汇入6'
            elif unique_seq[1] == 5:
                category = '匝道8直接汇入5'

            if category:
                ramp_categories[category].append(vehicle_id)

                # --- 修改后的匝道车辆采样逻辑 ---
                # 匝道车辆的第一变道(change_idx[0])是汇入主路(如8->7)，不作为样本。
                # 寻找进入主路后的后续变道(如7->6)，通常在 change_idx[1] 发生。
                if len(change_idx) > 1:
                    second_idx = change_idx[1]
                    # 确保第二次变道发生前，车辆在主路上(如车道7)保持了足够长的时间（符合 THRESHOLD_RAMP_SUBSEQUENT_CHANGE）
                    # 这里的距离是相对于第一次变道点的。
                    if (second_idx - change_idx[0]) >= THRESHOLD_RAMP_SUBSEQUENT_CHANGE and second_idx >= 100:
                        sample = group.iloc[second_idx - 100: second_idx - 50].copy()
                        sample['Label'] = '左变道'
                        sampling_data.append(sample)
                        source_sample_counts['左变道']['匝道驶入后续变道'] += 1

        # 2. 主路分类逻辑 (5, 6, 7)
        elif unique_seq[0] in [5, 6, 7]:
            if len(unique_seq) < 2:
                behavior = '跟驰'
            else:
                # 判定方向：增加=R/右, 减小=L/左
                changes = ['R' if unique_seq[i] > unique_seq[i - 1] else 'L' for i in range(1, len(unique_seq))]
                if len(changes) == 1:
                    behavior = '左变道' if changes[0] == 'L' else '右变道'
                else:
                    mapping = {('L', 'L'): '左变道后左变道', ('L', 'R'): '左变道后右变道',
                               ('R', 'L'): '右变道后左变道', ('R', 'R'): '右变道后右变道'}
                    behavior = mapping.get((changes[0], changes[-1]), '跟驰')

            maneuver_counts[behavior] += 1

            # 3. 采样与过滤逻辑
            first_idx = change_idx[0] if len(change_idx) > 0 else 0

            # 跟驰采样
            if behavior == '跟驰' and len(group) >= 50:
                s = random.randint(0, len(group) - 50)
                sample = group.iloc[s:s + 50].copy()
                sample['Label'] = '跟驰'
                sampling_data.append(sample)

            # 普通变道采样
            elif behavior in ['左变道', '右变道'] and first_idx >= THRESHOLD_LEFT_RIGHT_CHANGE:
                if first_idx >= 100:
                    sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                    sample['Label'] = behavior
                    sampling_data.append(sample)
                    source_sample_counts[behavior]['普通变道'] += 1

            # 二次变道采样 (取第一次变道前)
            elif '后' in behavior and first_idx >= THRESHOLD_SECONDARY_FIRST_CHANGE:
                label = '左变道' if behavior.startswith('左变道') else '右变道'
                if first_idx >= 100:
                    sample = group.iloc[first_idx - 100: first_idx - 50].copy()
                    sample['Label'] = label
                    sampling_data.append(sample)
                    source_sample_counts[label]['二次变道'] += 1

    # --- 最终统计输出 ---
    print(f"\n=== 1. 基础分类统计 ===")
    print(f"匝道汇入车辆: {sum(len(v) for v in ramp_categories.values())} 辆")
    print(f"主路行驶车辆行为分布: {dict(maneuver_counts)}")

    print(f"\n=== 2. 采样数据标签分布与来源分析 ===")
    if sampling_data:
        final_df = pd.concat(sampling_data, ignore_index=True)
        # 清理不必要的特征列
        to_drop = ['time', 'X', 'Y', 'LaneID', 'Dist_to_right_edge_marking', 'Dist_to_left_marking',
                   'Dist_to_right_marking']
        final_df = final_df.drop(columns=[c for c in to_drop if c in final_df.columns])

        # 标签分布打印
        counts = final_df['Label'].value_counts()
        print("-" * 50)
        for label, row_count in counts.items():
            vehicle_count = row_count // 50
            if label == '跟驰':
                print(f"  * {label}: {row_count} 行 (约 {vehicle_count} 辆车)")
            else:
                print(f"  * {label}: {row_count} 行 ({vehicle_count} 辆车)")
                # 打印详细来源
                for src, s_count in source_sample_counts[label].items():
                    if s_count > 0:
                        print(f"      - 来自{src}: {s_count} 辆")
        print("-" * 50)

        # 保存
        final_df.to_pickle(os.path.join(save_dir, "traffic_flows_sampling.pkl"))
        final_df.to_csv(os.path.join(save_dir, "traffic_flows_sampling.csv"), index=False)
        print(f"数据已保存至: {save_dir}")
    else:
        print("未提取到任何满足阈值条件的采样样本。")

    print(f"\n处理完成！")