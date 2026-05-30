import pandas as pd
import numpy as np
import random
from collections import defaultdict
import gc
import matplotlib as plt


THRESHOLD_RAMP_SUBSEQUENT = 50  # 西向匝道后二次变道最小间隔
# ==================== 西向处理函数（保留原逻辑） ====================
def process_west(df):
    print("\n--- 正在执行step06 west 向样本提取与标签分类 ---")
    sampling_data = []
    source_counts = {
        '左变道': defaultdict(int),
        '右变道': defaultdict(int),
        '跟驰': defaultdict(int)
    }
    main_lanes = [5, 6, 7]
    ramp_lane = 8

    def get_dir(cur, prev):
        return 'L' if cur < prev else 'R'  # 西向：车道减小为左

    for vehicle_id, group in df.groupby('ID'):
        group = group.sort_values('time')
        lanes = group['LaneID'].values
        if len(lanes) < 2:
            continue

        change_idx = np.where(lanes[:-1] != lanes[1:])[0] + 1
        unique_seq = [lanes[0]] + lanes[change_idx].tolist()

        # ---- 场景A：匝道汇入车辆 (起始车道为8) ----
        if unique_seq[0] == ramp_lane:
            if len(change_idx) > 1:  # 至少两次变道（8->7 + 后续）
                second_idx = change_idx[1]
                # 第二次变道必须满足间隔≥50且帧号≥100
                if (second_idx - change_idx[0]) >= THRESHOLD_RAMP_SUBSEQUENT and second_idx >= 100:
                    sample = group.iloc[second_idx - 100:second_idx - 50].copy()
                    sample['Label'] = '左变道'
                    sampling_data.append(sample)
                    source_counts['左变道']['匝道驶入后续变道'] += 1
            continue  # 匝道车辆处理完毕，跳过主路逻辑

        # ---- 场景B：纯主路车辆 ----
        if len(unique_seq) < 2:
            behavior = '跟驰'
            src_tag = '常规跟驰'
        else:
            # 判定多次变道的总体方向
            changes = [get_dir(unique_seq[i], unique_seq[i - 1]) for i in range(1, len(unique_seq))]
            if len(changes) == 1:
                behavior = '左变道' if changes[0] == 'L' else '右变道'
                src_tag = '普通变道'
            else:
                mapping = {('L', 'L'): '左变道', ('L', 'R'): '左变道',
                           ('R', 'L'): '右变道', ('R', 'R'): '右变道'}
                behavior = mapping.get((changes[0], changes[-1]), '跟驰')
                src_tag = '二次变道'

        first_idx = change_idx[0] if len(change_idx) > 0 else 0

        if behavior == '跟驰' and len(group) >= 50:
            s = random.randint(0, len(group) - 50)
            sample = group.iloc[s:s + 50].copy()
            sample['Label'] = '跟驰'
            sampling_data.append(sample)
            source_counts['跟驰'][src_tag] += 1
        elif behavior in ['左变道', '右变道'] and first_idx >= 100:
            sample = group.iloc[first_idx - 100:first_idx - 50].copy()
            sample['Label'] = behavior
            sampling_data.append(sample)
            source_counts[behavior][src_tag] += 1

    # 合并输出
    if sampling_data:
        res_df = pd.concat(sampling_data, ignore_index=True)
        to_drop = ['Dist_to_right_edge_marking',
                   'Dist_to_left_marking', 'Dist_to_right_marking']
        res_df = res_df.drop(columns=[c for c in to_drop if c in res_df.columns])

        print(f"✅ west 向样本提取完成：提取了 {len(res_df) // 50} 辆车的样本片段")
        for label, sub_src in source_counts.items():
            total = sum(sub_src.values())
            if total > 0:
                print(f"   - {label}: {total} 辆 (来源: {dict(sub_src)})")
        return res_df
    else:
        print("⚠️ west 向未提取到符合条件的样本")
        return pd.DataFrame()

# df_west_sampling = process_west(df_west_sample)
#
# gc.collect()

