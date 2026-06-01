"""
批量处理 5 个 Location 交通流数据（location1-5）
XLSX/CSV → 流水线 → 合并左右变道 → 安全编码 → 保存
"""
import pandas as pd
import numpy as np
import os
import ast
import gc
import sys
import importlib.util
import time

# 确保能导入同目录下的 step 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step01 import split_by_direction
from step02clean import clean_by_direction
from step03complete import complete_by_direction
from step04smooth import smooth_by_direction
from step05_sample import compute_features_with_mttc
from step06change import extract_lane_change_samples, extract_following_samples
from model import encode_safety_categories

# ==================== 配置 ====================
LANE_COEFFS_FILE = r"E:\0little\lane_coeffs.xlsx"
BASE_READ = r"E:\0little\read"
BASE_OUT = r"E:\0little"

# 制动反应时间（秒），影响 RSD、F_ERSD、B_ERSD
REACTION_TIME = 2.0          # 用于 step05（同车道前车 RSD）
LOC1_REACTION_TIME = REACTION_TIME   # 用于 location1-4 的 step06（F_ERSD、B_ERSD）
LOC5_REACTION_TIME = REACTION_TIME     # 用于 location5 的 step06（F_ERSD、B_ERSD）

# 样本截取窗口参数
PRE_FRAMES = 100             # 变道/跟驰样本：变化点前推的帧数
SAMPLE_FRAMES = 75           # 截取的样本帧数（25 帧 = 1 秒）

LOCATIONS = {
    'location1': {
        'dir': os.path.join(BASE_READ, 'location1'),
        'files': ['1-1_trajectory', '1-2_trajectory'],
    },
    'location2': {
        'dir': os.path.join(BASE_READ, 'location2'),
        'files': ['2-1_trajectory', '2-2_trajectory', '2-3_trajectory', '2-4_trajectory'],
    },
    'location3_part1': {
        'dir': os.path.join(BASE_READ, 'location3_part1'),
        'files': ['3-1_trajectory', '3-2_trajectory', '3-3_trajectory', '3-4_trajectory'],
    },
    'location4_part1': {
        'dir': os.path.join(BASE_READ, 'location4_part1'),
        'files': ['4-1_trajectory', '4-2_trajectory', '4-3_trajectory', '4-4_trajectory', '4-5_trajectory'],
    },
}


# ==================== 车道线加载 ====================
def load_lane_coeffs():
    """从 lane_coeffs.xlsx 加载车道线系数，返回 {where: [8条曲线]}"""
    df = pd.read_excel(LANE_COEFFS_FILE)
    coeffs_map = {}
    for _, row in df.iterrows():
        where = str(row['where'])
        raw = row['lane_coeffs']
        # 解析字符串形式的列表："[a5,...,a0],\n    [a5,...,a0],\n    ..."
        wrapped = '[' + raw.replace('\n', '') + ']'
        parsed = ast.literal_eval(wrapped)
        coeffs_map[where] = parsed  # 8 条曲线: 前4=dir1, 后4=dir2
    return coeffs_map


# ==================== XLSX → CSV ====================
def convert_xlsx_to_csv(xlsx_path, csv_path):
    """将 XLSX 转为 CSV（utf-8-sig），修复浮点精度，若 CSV 已存在则跳过"""
    if os.path.exists(csv_path):
        print(f"    CSV 已存在，跳过转换: {os.path.basename(csv_path)}")
        return
    print(f"    转换 {os.path.basename(xlsx_path)} → CSV ...")
    df = pd.read_excel(xlsx_path)
    # 修复 XLSX 浮点精度误差: Frame→整数, Time→2位小数
    df['Frame'] = df['Frame'].round().astype(int)
    if 'Time' in df.columns:
        df['Time'] = df['Time'].round(2)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    完成: {len(df)} 行 → {os.path.basename(csv_path)}")


# ==================== 单文件流水线 ====================
def process_single_file(csv_path, out_dir=None):
    """
    对单个 CSV 文件运行完整流水线，返回 (df_left, df_right, df_traj) 或 (None, None, None)
    df_traj: 变道车辆的精简完整轨迹 (ID, Frame, X, Y, Direction)
    """
    fname = os.path.basename(csv_path)
    prefix = fname.replace('_trajectory.csv', '')

    try:
        # Step 1: 按方向分流
        df_dir1, df_dir2 = split_by_direction(csv_path)
        if len(df_dir1) == 0 or len(df_dir2) == 0:
            print(f"    [!] {prefix}: 某方向数据为空，跳过")
            return None, None, None, None

        # Step 2: 清洗
        df_clean_1, df_clean_2 = clean_by_direction(df_dir1, df_dir2)
        if len(df_clean_1) == 0 or len(df_clean_2) == 0:
            print(f"    [!] {prefix}: 清洗后数据为空，跳过")
            return None, None, None, None

        # Step 3: 补全
        df_comp_1, df_comp_2 = complete_by_direction(df_clean_1, df_clean_2,
                                                      df_dir1, df_dir2, '1', '2')

        # Step 4: 平滑
        df_smooth_1, df_smooth_2 = smooth_by_direction(df_comp_1, df_comp_2,
                                                        label_1='1', label_2='2')

        # ★ 保存平滑后的完整轨迹（含邻车ID），供可视化使用
        df_smooth_full = None
        if df_smooth_1 is not None and len(df_smooth_1) > 0:
            df_smooth_1 = df_smooth_1.copy()
            df_smooth_1['Direction'] = 1
            df_smooth_1['Source'] = prefix
        if df_smooth_2 is not None and len(df_smooth_2) > 0:
            df_smooth_2 = df_smooth_2.copy()
            df_smooth_2['Direction'] = 2
            df_smooth_2['Source'] = prefix
        smooth_parts = [d for d in [df_smooth_1, df_smooth_2] if d is not None and len(d) > 0]
        if smooth_parts:
            df_smooth_full = pd.concat(smooth_parts, ignore_index=True)

        # Step 5: 特征重构（df_sample = 全部车辆的完整轨迹）
        df_sample_1, df_sample_2 = compute_features_with_mttc(df_smooth_1, df_smooth_2, '1', '2',
                                                               reaction_time=REACTION_TIME)

        # Step 6: 提取变道样本 (offset=5, tolerance=1.5)
        df_left, df_right = extract_lane_change_samples(df_sample_1, df_sample_2,
                                                         df_smooth_1, df_smooth_2, 5, 1.5,
                                                         reaction_time=LOC1_REACTION_TIME,
                                                         pre_frames=PRE_FRAMES, sample_frames=SAMPLE_FRAMES)

        # 添加来源标识（子文件前缀，如 2-1、3-2），便于追溯场景和匹配车道线
        if df_left is not None and len(df_left) > 0:
            df_left['Source'] = prefix
        if df_right is not None and len(df_right) > 0:
            df_right['Source'] = prefix

        # 保存变道车辆的精简完整轨迹 (用于可视化画蓝点)
        lc_ids = set()
        for d in [df_left, df_right]:
            if d is not None and len(d) > 0:
                lc_ids.update(d['ID'].unique())
        traj_parts = []
        for traj_df, d_label in [(df_sample_1, 1), (df_sample_2, 2)]:
            sub = traj_df[traj_df['ID'].isin(lc_ids)][['ID', 'Frame', 'X', 'Y']].copy()
            sub['Direction'] = d_label
            sub['Source'] = prefix
            traj_parts.append(sub)
        df_traj = pd.concat(traj_parts, ignore_index=True) if traj_parts else pd.DataFrame()

        # 提取跟驰样本
        df_following = extract_following_samples(df_sample_1, df_sample_2,
                                                  df_smooth_1, df_smooth_2,
                                                  pre_frames=PRE_FRAMES, sample_frames=SAMPLE_FRAMES)
        if df_following is not None and len(df_following) > 0:
            df_following['Source'] = prefix
            df_following = encode_safety_categories(df_following)
            fout = os.path.join(out_dir, 'traffic_following_change.csv')
            if os.path.exists(fout):
                existing = pd.read_csv(fout)
                df_following = pd.concat([existing, df_following], ignore_index=True)
            df_following.to_csv(fout, index=False, encoding='utf-8-sig')
            print(f"    跟驰: {len(df_following)} 行, {df_following['ID'].nunique()} 辆车")

        gc.collect()
        return df_left, df_right, df_traj, df_smooth_full

    except Exception as e:
        print(f"    [!] {prefix}: 处理失败 - {e}")
        return None, None, None, None


# ==================== Phase 4: 可视化（合并后按 Source 匹配车道线） ====================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _get_direction(y_values, threshold=25):
    return 1 if np.median(y_values) < threshold else 2


def _get_threshold(df_coeffs, source):
    """从车道线 a0 值自动推算方向判定阈值（方向1最大a0 + 方向2最小a0）/2"""
    sub = df_coeffs[df_coeffs['where'] == source]
    dir1_a0 = sub[sub['direction'] == 1]['a0']
    dir2_a0 = sub[sub['direction'] == 2]['a0']
    if len(dir1_a0) > 0 and len(dir2_a0) > 0:
        return (dir1_a0.max() + dir2_a0.min()) / 2
    return 25  # fallback


def visualize_location(out_dir):
    """对单个 location 的合并数据生成 5×2 轨迹图（左右变道各 10 辆）"""
    left_path = os.path.join(out_dir, 'traffic_left_change.csv')
    right_path = os.path.join(out_dir, 'traffic_right_change.csv')
    traj_path = os.path.join(out_dir, 'trajectory_full.csv')
    coeffs_path = os.path.join(out_dir, 'lane_coeffs.csv')

    if not os.path.exists(coeffs_path):
        print(f"  [!] 无车道线文件，跳过可视化")
        return

    df_coeffs = pd.read_csv(coeffs_path)
    df_traj = pd.read_csv(traj_path) if os.path.exists(traj_path) else pd.DataFrame()
    vis_dir = os.path.join(out_dir, 'vis_lane_change')
    os.makedirs(vis_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for side_label, side_file in [('left', left_path), ('right', right_path)]:
        if not os.path.exists(side_file):
            continue
        df_samples = pd.read_csv(side_file)
        veh_ids = df_samples['ID'].unique()
        n_select = min(10, len(veh_ids))
        if n_select == 0:
            continue

        selected = rng.choice(veh_ids, n_select, replace=False)

        # === 计算该 behavior 全局 X 范围（所有选中车辆） ===
        x_min_all, x_max_all = float('inf'), float('-inf')
        y_min_all, y_max_all = float('inf'), float('-inf')
        for vid in selected:
            veh_sample = df_samples[df_samples['ID'] == vid]
            src = veh_sample['Source'].iloc[0]
            veh_full = df_traj[(df_traj['ID'] == vid) & (df_traj['Source'] == src)] if len(df_traj) > 0 else pd.DataFrame()
            all_pts = pd.concat([veh_full[['X', 'Y']], veh_sample[['X', 'Y']]], ignore_index=True)
            if len(all_pts) > 0:
                x_min_all = min(x_min_all, all_pts['X'].min())
                x_max_all = max(x_max_all, all_pts['X'].max())
                y_min_all = min(y_min_all, all_pts['Y'].min())
                y_max_all = max(y_max_all, all_pts['Y'].max())
        # 加 5m 边距
        x_pad, y_pad = 5, 2
        x_lim = (x_min_all - x_pad, x_max_all + x_pad)
        y_lim = (y_max_all + y_pad, y_min_all - y_pad)  # Y 逆序

        fig, axes = plt.subplots(5, 2, figsize=(24, 13.5))
        axes = axes.flatten()
        behavior_label = '左变道' if side_label == 'left' else '右变道'

        for idx, vid in enumerate(selected):
            ax = axes[idx]
            veh_sample = df_samples[df_samples['ID'] == vid]
            if veh_sample.empty:
                ax.axis('off')
                continue

            src = veh_sample['Source'].iloc[0]
            # 按 Source 精确筛选轨迹（避免不同子文件同名 ID 混淆）
            veh_full = df_traj[(df_traj['ID'] == vid) & (df_traj['Source'] == src)] if len(df_traj) > 0 else pd.DataFrame()
            # 回退：若无轨迹，用样本数据作为背景
            use_sample_as_bg = veh_full.empty
            if use_sample_as_bg:
                veh_full = veh_sample

            # 确定方向：从车道线系数推算该 Source 的方向分界阈值
            y_thresh = _get_threshold(df_coeffs, src)
            if len(veh_sample['Y'].dropna()) > 0:
                direction = _get_direction(veh_sample['Y'].values, y_thresh)
            else:
                direction = _get_direction(veh_full['Y'].values, y_thresh)

            # 匹配车道线系数
            coeffs_match = df_coeffs[(df_coeffs['where'] == src) & (df_coeffs['direction'] == direction)]
            curves = []
            for _, cr in coeffs_match.iterrows():
                curves.append([cr['a5'], cr['a4'], cr['a3'], cr['a2'], cr['a1'], cr['a0']])

            # --- 绘图 ---
            if use_sample_as_bg:
                # 回退：无完整轨迹，样本放大为蓝色表示"仅有数据"
                ax.scatter(veh_full['X'], veh_full['Y'], c='blue', s=5, alpha=0.6, label='样本(无完整)')
            else:
                ax.scatter(veh_full['X'], veh_full['Y'], c='blue', s=1, alpha=0.5, label='完整轨迹')
                ax.scatter(veh_sample['X'], veh_sample['Y'], c='red', s=3, alpha=0.8, label='样本片段')

            # 车道线（在数据范围内绘制，避免多项式发散）
            x_vals = np.linspace(x_lim[0], x_lim[1], 400)
            for coeffs in curves:
                y_vals = np.polyval(coeffs, x_vals)
                ax.plot(x_vals, y_vals, color='white', linewidth=2.5, alpha=0.95)
                ax.plot(x_vals, y_vals, color='black', linewidth=0.8, linestyle='--', alpha=0.7)

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            note = ' (无轨迹)' if use_sample_as_bg else ''
            ax.set_title(f'ID:{int(vid)} {src}{note}', fontsize=9)
            ax.set_xlabel('X (m)', fontsize=7)
            ax.set_ylabel('Y (m)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='upper right')

        # 隐藏多余子图
        for idx in range(n_select, 10):
            axes[idx].axis('off')

        plt.suptitle(f'{os.path.basename(out_dir)} {behavior_label}轨迹 (随机{n_select}辆)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'{behavior_label}_trajectory.png')
        plt.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close()
        side_cn = '左变道' if side_label == 'left' else '右变道'
        print(f"    {side_cn}: {n_select} 辆车 → {save_path}")


# ==================== Location5 单独处理 ====================
def process_location5():
    """
    处理 location5：仅 step06（变道提取）使用 location5 自身模块，
    step01~step05 与 location1-4 共用同一套流水线。
    差异点：原始列名为 time（小写），需在 CSV 转换时统一为 Time。
    """
    print(f"\n{'=' * 60}")
    print("处理 location5")
    print(f"{'=' * 60}")

    loc_name = 'location5'
    src_dir = os.path.join(BASE_READ, 'location5')
    out_dir = os.path.join(BASE_OUT, loc_name)
    os.makedirs(out_dir, exist_ok=True)

    xlsx_path = os.path.join(src_dir, '5_trajectory.xlsx')
    csv_path = os.path.join(src_dir, '5_trajectory.csv')

    # ---------- Phase 1: XLSX → CSV（合并两个 sheet） ----------
    print("\n[Phase 1] XLSX → CSV 转换...")
    if not os.path.exists(csv_path):
        print(f"    转换 5_trajectory.xlsx (part1 + part2) → CSV ...")
        xls = pd.ExcelFile(xlsx_path)
        parts = []
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name)
            # 丢弃第一列乱码列
            first_col = df_sheet.columns[0]
            df_sheet.drop(columns=[first_col], inplace=True)
            # 修复浮点精度
            df_sheet['Frame'] = df_sheet['Frame'].round().astype(int)
            if 'time' in df_sheet.columns:
                df_sheet['time'] = df_sheet['time'].round(2)
            parts.append(df_sheet)

        # 直接合并两个 sheet（Frame 有重叠是正常的，两个 sheet 记录同一时间段的不同车辆）
        df = pd.concat(parts, ignore_index=True)
        # 边界 ID 重复行去重（如 ID 6348 在两个 sheet 中同时出现）
        n_before = len(df)
        df.drop_duplicates(subset=['ID', 'Frame'], keep='first', inplace=True)
        if n_before - len(df) > 0:
            print(f"    去重 {n_before - len(df)} 行 (边界 ID 重叠)")
        df.sort_values(['ID', 'Frame'], inplace=True)
        # 统一列名 time → Time
        if 'time' in df.columns and 'Time' not in df.columns:
            df.rename(columns={'time': 'Time'}, inplace=True)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"    完成: {len(df)} 行, {df['ID'].nunique()} 辆 → {os.path.basename(csv_path)}")
    else:
        print(f"    CSV 已存在，跳过转换: {os.path.basename(csv_path)}")

    # ---------- Phase 2: 共用流水线（step01~step05） ----------
    print("\n[Phase 2] 流水线处理 (step01~step05 共用)...")

    # Step 1: 按 Direction 分流（location5 也有 Direction=1,2）
    print("\n  --- Step 1: 按 Direction 分流 ---")
    df_dir1, df_dir2 = split_by_direction(csv_path)
    if len(df_dir1) == 0 or len(df_dir2) == 0:
        print(f"    [!] 某方向数据为空，跳过")
        return

    # Step 2: 清洗
    print("\n  --- Step 2: 清洗 ---")
    df_clean_1, df_clean_2 = clean_by_direction(df_dir1, df_dir2,
                                                  label_1='1', label_2='2')
    if len(df_clean_1) == 0 or len(df_clean_2) == 0:
        print(f"    [!] 清洗后数据为空，跳过")
        return

    # Step 3: 补全
    print("\n  --- Step 3: 补全 ---")
    df_comp_1, df_comp_2 = complete_by_direction(df_clean_1, df_clean_2,
                                                  df_dir1, df_dir2, '1', '2')

    # Step 4: 平滑（保存含邻车 ID 的完整轨迹）
    print("\n  --- Step 4: 平滑 ---")
    df_smooth_1, df_smooth_2 = smooth_by_direction(df_comp_1, df_comp_2,
                                                    label_1='1', label_2='2')

    smooth_parts = []
    for d, label in [(df_smooth_1, 1), (df_smooth_2, 2)]:
        if d is not None and len(d) > 0:
            d = d.copy()
            d['Direction'] = label
            d['Source'] = 'loc5'
            smooth_parts.append(d)
    df_smooth_full = pd.concat(smooth_parts, ignore_index=True) if smooth_parts else pd.DataFrame()

    # Step 5: 特征重构
    print("\n  --- Step 5: 特征重构 ---")
    df_sample_1, df_sample_2 = compute_features_with_mttc(df_smooth_1, df_smooth_2, '1', '2',
                                                           reaction_time=REACTION_TIME)

    # ---------- 按 LaneID 重组为东西向（供 step06 使用） ----------
    print("\n  --- 按 LaneID 重组为东西向 ---")
    # 利用已有的 df_smooth_full（含 Direction/Source）按 LaneID 拆分
    df_smooth_e = df_smooth_full[df_smooth_full['LaneID'].isin([0, 1, 2, 3])].copy()
    df_smooth_w = df_smooth_full[df_smooth_full['LaneID'].isin([5, 6, 7, 8])].copy()

    # 样本数据同样合并后按 LaneID 拆分
    sample_parts = []
    for d, label in [(df_sample_1, 1), (df_sample_2, 2)]:
        if d is not None and len(d) > 0:
            d = d.copy()
            d['Direction'] = label
            sample_parts.append(d)
    df_sample_all = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    df_sample_e = df_sample_all[df_sample_all['LaneID'].isin([0, 1, 2, 3])].copy()
    df_sample_w = df_sample_all[df_sample_all['LaneID'].isin([5, 6, 7, 8])].copy()

    print(f"    东向LaneID 0-3: 样本 {len(df_sample_e)} 行, 平滑 {len(df_smooth_e)} 行")
    print(f"    西向LaneID 5-8: 样本 {len(df_sample_w)} 行, 平滑 {len(df_smooth_w)} 行")

    # 若某方向无数据则跳过
    if df_sample_e.empty or df_sample_w.empty or df_smooth_e.empty or df_smooth_w.empty:
        print("    [!] 某方向无数据，跳过 location5 step06")
        df_left, df_right = pd.DataFrame(), pd.DataFrame()
    else:
        # ★ 关键：location5 的 step06 内部使用 time（小写），需要重命名
        for d in [df_sample_e, df_sample_w, df_smooth_e, df_smooth_w]:
            if 'Time' in d.columns and 'time' not in d.columns:
                d.rename(columns={'Time': 'time'}, inplace=True)

        # ---------- Step 6: 变道提取（使用 location5 专用模块） ----------
        loc5_dir = r"E:\0little\location5"
        spec = importlib.util.spec_from_file_location("_loc5_step06",
                                                        os.path.join(loc5_dir, 'step06change.py'))
        step06_l5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step06_l5)

        print("\n  --- Step 6: 变道提取 (location5 模块) ---")
        df_left, df_right = step06_l5.extract_lane_change_samples(
            df_sample_e, df_sample_w, df_smooth_e, df_smooth_w,
            reaction_time=LOC5_REACTION_TIME,
            pre_frames=PRE_FRAMES, sample_frames=SAMPLE_FRAMES)

        # 返回后统一列名 time → Time
        for d in [df_left, df_right]:
            if d is not None and not d.empty and 'time' in d.columns and 'Time' not in d.columns:
                d.rename(columns={'time': 'Time'}, inplace=True)

        # ---------- 跟驰样本提取 ----------
        print("\n  --- Step 6b: 跟驰样本提取 (location5 模块) ---")
        df_following = step06_l5.extract_following_samples(
            df_sample_e, df_sample_w, df_smooth_e, df_smooth_w,
            pre_frames=PRE_FRAMES, sample_frames=SAMPLE_FRAMES)
        if df_following is not None and len(df_following) > 0:
            if 'time' in df_following.columns and 'Time' not in df_following.columns:
                df_following.rename(columns={'time': 'Time'}, inplace=True)
            df_following['Source'] = 'loc5'
            df_following = encode_safety_categories(df_following)
            fout = os.path.join(out_dir, 'traffic_following_change.csv')
            if os.path.exists(fout):
                existing = pd.read_csv(fout)
                df_following = pd.concat([existing, df_following], ignore_index=True)
            df_following.to_csv(fout, index=False, encoding='utf-8-sig')
            print(f"    跟驰: {len(df_following)} 行, {df_following['ID'].nunique()} 辆车")

    # ---------- Phase 3: 编码 + 保存 ----------
    print(f"\n[Phase 3] 编码 & 保存...")

    for df_lr, side in [(df_left, 'left'), (df_right, 'right')]:
        if df_lr is None or len(df_lr) == 0:
            print(f"  {side}: 无数据，跳过")
            continue
        df_lr = df_lr.copy()
        df_lr['Source'] = 'loc5'

        # 根据 LaneID 推算 Direction（0-3 → 1, 5-8 → 2）
        def _lane_to_dir_simple(series):
            m = series.mode().iloc[0]
            return 1 if m <= 3 else 2
        dir_map = df_lr.groupby('ID')['LaneID'].agg(_lane_to_dir_simple)
        df_lr['Direction'] = df_lr['ID'].map(dir_map)

        df_lr = encode_safety_categories(df_lr)

        out_path = os.path.join(out_dir, f'traffic_{side}_change.csv')
        df_lr.to_csv(out_path, index=False, encoding='utf-8-sig')
        n_veh = df_lr['ID'].nunique()
        print(f"  {side}_change: {len(df_lr)} 行, {n_veh} 辆车 → {out_path}")

    # 平滑完整轨迹
    if len(df_smooth_full) > 0:
        sp = os.path.join(out_dir, 'trajectory_full_smoothed.csv')
        df_smooth_full.to_csv(sp, index=False, encoding='utf-8-sig')
        print(f"  平滑完整轨迹已保存: {len(df_smooth_full)} 行 → {sp}")

    # 精简轨迹
    lc_ids = set()
    for d in [df_left, df_right]:
        if d is not None and len(d) > 0:
            lc_ids.update(d['ID'].unique())
    traj_parts = []
    for d, label in [(df_sample_e, 1), (df_sample_w, 2)]:
        if d is None or len(d) == 0:
            continue
        sub = d[d['ID'].isin(lc_ids)][['ID', 'Frame', 'X', 'Y']].copy()
        if len(sub) > 0:
            sub['Direction'] = label
            sub['Source'] = 'loc5'
            traj_parts.append(sub)
    if traj_parts:
        tp = os.path.join(out_dir, 'trajectory_full.csv')
        pd.concat(traj_parts, ignore_index=True).to_csv(tp, index=False, encoding='utf-8-sig')
        print(f"  精简轨迹已保存: {len(pd.concat(traj_parts))} 行 → {tp}")

    # location5 无车道线数据，跳过可视化
    # print(f"\n  location5 完成（无可视化: 无车道线系数）")


# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("批量交通流处理 — 5 个 Location × 16 个文件")
    print("=" * 60)

    # 加载车道线
    print("\n加载车道线系数...")
    coeffs_map = load_lane_coeffs()
    print(f"  已加载 {len(coeffs_map)} 个场景的车道线")

    # 逐一处理各 location
    for loc_name, loc_cfg in LOCATIONS.items():
        loc_dir = loc_cfg['loc_dir'] = loc_cfg['dir']
        files = loc_cfg['files']
        out_dir = os.path.join(BASE_OUT, loc_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"处理 {loc_name}: {len(files)} 个文件")
        print(f"  输入: {loc_dir}")
        print(f"  输出: {out_dir}")
        print(f"{'=' * 60}")

        # Phase 1: XLSX → CSV（或直接读取已有 CSV）
        print("\n[Phase 1] XLSX → CSV 转换...")
        csv_paths = []
        for fbase in files:
            xlsx_path = os.path.join(loc_dir, fbase + '.xlsx')
            csv_path = os.path.join(loc_dir, fbase + '.csv')
            if os.path.exists(csv_path):
                print(f"    CSV 已存在，直接使用: {os.path.basename(csv_path)}")
                csv_paths.append((fbase, csv_path))
            elif os.path.exists(xlsx_path):
                convert_xlsx_to_csv(xlsx_path, csv_path)
                csv_paths.append((fbase, csv_path))
            else:
                print(f"    [!] 文件不存在: {xlsx_path} 或 {csv_path}")
                continue
        print(f"  共 {len(csv_paths)} 个 CSV 就绪")

        # Phase 2: 逐文件走流水线
        print("\n[Phase 2] 流水线处理...")
        all_left, all_right, all_traj, all_smooth = [], [], [], []
        total_left_veh, total_right_veh = 0, 0

        for fbase, csv_path in csv_paths:
            print(f"\n  --- {fbase} ---")
            df_left, df_right, df_traj, df_smooth_full = process_single_file(csv_path, out_dir)
            if df_left is not None and len(df_left) > 0:
                n_left = df_left['ID'].nunique()
                all_left.append(df_left)
                total_left_veh += n_left
                print(f"    左变道: {len(df_left)} 行, {n_left} 辆车")
            if df_right is not None and len(df_right) > 0:
                n_right = df_right['ID'].nunique()
                all_right.append(df_right)
                total_right_veh += n_right
                print(f"    右变道: {len(df_right)} 行, {n_right} 辆车")
            if df_traj is not None and len(df_traj) > 0:
                all_traj.append(df_traj)
            if df_smooth_full is not None and len(df_smooth_full) > 0:
                all_smooth.append(df_smooth_full)

        # Phase 3: 合并 + 编码 + 保存
        print(f"\n[Phase 3] 合并 & 编码 & 保存...")

        for df_list, side_label in [(all_left, 'left'), (all_right, 'right')]:
            if not df_list:
                print(f"  {side_label}: 无数据，跳过")
                continue

            df_merged = pd.concat(df_list, ignore_index=True)
            df_merged = encode_safety_categories(df_merged)

            out_path = os.path.join(out_dir, f'traffic_{side_label}_change.csv')
            df_merged.to_csv(out_path, index=False, encoding='utf-8-sig')
            n_veh = df_merged['ID'].nunique()
            print(f"  {side_label}_change: {len(df_merged)} 行, {n_veh} 辆车 → {out_path}")

        print(f"\n  {loc_name} 完成: 左变道 {total_left_veh} 辆, 右变道 {total_right_veh} 辆")

        # 保存精简轨迹（供可视化使用）
        if all_traj:
            traj_merged = pd.concat(all_traj, ignore_index=True)
            traj_path = os.path.join(out_dir, 'trajectory_full.csv')
            traj_merged.to_csv(traj_path, index=False, encoding='utf-8-sig')
            print(f"  精简轨迹已保存: {len(traj_merged)} 行 → {traj_path}")

        # 保存含邻车ID的平滑完整轨迹（供查询后车等周边车辆轨迹）
        if all_smooth:
            smooth_merged = pd.concat(all_smooth, ignore_index=True)
            smooth_path = os.path.join(out_dir, 'trajectory_full_smoothed.csv')
            smooth_merged.to_csv(smooth_path, index=False, encoding='utf-8-sig')
            print(f"  平滑完整轨迹已保存: {len(smooth_merged)} 行 → {smooth_path}")

        # 保存该 location 的车道线映射（供可视化使用）
        loc_coeffs = {}
        for fbase in files:
            if fbase.replace('_trajectory', '') in coeffs_map:
                loc_coeffs[fbase.replace('_trajectory', '')] = coeffs_map[fbase.replace('_trajectory', '')]
        coeffs_out = os.path.join(out_dir, 'lane_coeffs.csv')
        coeffs_rows = []
        for where, curves in loc_coeffs.items():
            for i, c in enumerate(curves):
                coeffs_rows.append({'where': where, 'curve_idx': i,
                                    'direction': 1 if i < len(curves)//2 else 2,
                                    'a5': c[0], 'a4': c[1], 'a3': c[2], 'a2': c[3], 'a1': c[4], 'a0': c[5]})
        if coeffs_rows:
            pd.DataFrame(coeffs_rows).to_csv(coeffs_out, index=False, encoding='utf-8-sig')
            print(f"  车道线已保存: {coeffs_out}")

    # ==================== Location5 单独处理 ====================
    process_location5()

    # Phase 4: 可视化（对所有有轨迹数据的 location）
    print(f"\n{'=' * 60}")
    print("[Phase 4] 轨迹可视化")
    print(f"{'=' * 60}")
    for loc_name in LOCATIONS:
        out_dir = os.path.join(BASE_OUT, loc_name)
        print(f"\n  处理 {loc_name} ...")
        visualize_location(out_dir)

    print(f"\n{'=' * 60}")
    print("全部处理完成!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
