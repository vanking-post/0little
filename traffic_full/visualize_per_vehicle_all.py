"""
单车辆多维可视化（全量） — 每张图 3×2=6 面板，读取各 location 原始目录全部 1,496 辆车
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os

from safety_scoring import worst_cat
from safety_scoring_exp import risk_score as exp_risk_score, risk_label as exp_risk_label

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
SEED = 39
N_PER_SIDE = 5
DPI = 80
FIGSIZE = (26, 13)

LOCS = {
    'location1': {
        'sample_dir': 'E:/0little/location1',
        'sample_prefix': 'traffic',
        'traj_path': 'E:/0little/location1/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location1/lane_coeffs.csv',
    },
    'location2': {
        'sample_dir': 'E:/0little/location2',
        'sample_prefix': 'traffic',
        'traj_path': 'E:/0little/location2/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location2/lane_coeffs.csv',
    },
    'location3': {
        'sample_dir': 'E:/0little/location3',
        'sample_prefix': 'traffic',
        'traj_path': 'E:/0little/location3/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location3/lane_coeffs.csv',
    },
    'location4': {
        'sample_dir': 'E:/0little/location4',
        'sample_prefix': 'traffic',
        'traj_path': 'E:/0little/location4/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location4/lane_coeffs.csv',
    },
    'location5': {
        'sample_dir': 'E:/0little/location5',
        'sample_prefix': 'traffic',
        'traj_path': 'E:/0little/location5/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location5/lane_coeffs.csv',
    },
}

OUT_DIR = 'E:/0little/traffic_full/vis_all'
os.makedirs(OUT_DIR, exist_ok=True)


# ==================== 辅助函数 ====================
def load_sample(loc_name, side):
    cfg = LOCS[loc_name]
    fp = os.path.join(cfg['sample_dir'], f'{cfg["sample_prefix"]}_{side}_change.csv')
    df = pd.read_csv(fp)
    if 'time' in df.columns and 'Time' not in df.columns:
        df = df.rename(columns={'time': 'Time'})
    return df


def _ptsize(length_series):
    """车长(m) → 散点大小(points²)，车长4~16m → s≈12~48"""
    return (length_series * 3).clip(lower=6, upper=60)


def compute_threshold(coeffs_df, source):
    sub = coeffs_df[coeffs_df['where'] == source]
    d1 = sub[sub['direction'] == 1]['a0']
    d2 = sub[sub['direction'] == 2]['a0']
    if len(d1) > 0 and len(d2) > 0:
        return (d1.max() + d2.min()) / 2
    return 25


def get_direction(y_vals, threshold):
    return 1 if np.median(y_vals) < threshold else 2


# ==================== 各面板绘制 ====================
def panel_xy_trajectory(ax, veh_traj, veh_sample, curves, x_range,
                        smooth_df=None, side='left', src=None, loc_name=''):
    use_fallback = veh_traj.empty
    sample_frames = veh_sample['Frame'].values
    ego_id = veh_sample['ID'].iloc[0]

    if use_fallback:
        ax.scatter(veh_sample['X'], veh_sample['Y'], c='#3498db', s=5, alpha=0.7,
                   label='样本(无完整轨迹)')
    else:
        ax.scatter(veh_traj['X'], veh_traj['Y'], c='#3498db', s=0.5, alpha=0.4,
                   label='完整轨迹')
        ax.scatter(veh_sample['X'], veh_sample['Y'], c='#e74c3c',
                   s=_ptsize(veh_sample['Length']), alpha=0.85, label='变道样本')
        # 每隔8帧画本车矩形（保证首尾帧有矩形）
        rect_idx = list(range(0, len(veh_sample), 8))
        if rect_idx[-1] != len(veh_sample) - 1:
            rect_idx.append(len(veh_sample) - 1)
        for _, r in veh_sample.iloc[rect_idx].iterrows():
            rect = Rectangle((r['X'] - r['Length'] / 2, r['Y'] - r['Width'] / 2),
                             r['Length'], r['Width'],
                             facecolor='#e74c3c', edgecolor='white', linewidth=0.5, alpha=0.55)
            ax.add_patch(rect)

    # === 邻车轨迹（散点大小按车长缩放） ===
    if smooth_df is not None and len(smooth_df) > 0:
        ego_smooth = smooth_df[(smooth_df['ID'] == ego_id) & (smooth_df['Source'] == src)
                               & (smooth_df['Frame'].isin(sample_frames))]
        if len(ego_smooth) > 0:

            def _plot_nbr(id_col, color, prefix):
                ids = ego_smooth[id_col].values
                ids = ids[(ids > 0) & (ids != ego_id)]
                if len(ids) == 0:
                    return
                nid = int(np.median(ids))
                traj = smooth_df[(smooth_df['ID'] == nid) & (smooth_df['Source'] == src)
                                 & (smooth_df['Frame'].isin(sample_frames))].copy()
                if len(traj) == 0:
                    return
                # 小圆点显示完整路径
                ax.scatter(traj['X'], traj['Y'], c=color, s=3, alpha=0.35)
                # 每隔8帧画矩形显示车身尺寸（保证首尾帧）
                rect_idx = list(range(0, len(traj), 8))
                if rect_idx[-1] != len(traj) - 1:
                    rect_idx.append(len(traj) - 1)
                for _, r in traj.iloc[rect_idx].iterrows():
                    rect = Rectangle((r['X'] - r['Length'] / 2, r['Y'] - r['Width'] / 2),
                                     r['Length'], r['Width'],
                                     facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.55)
                    ax.add_patch(rect)
                # 图例仅用一个代理
                ax.scatter([], [], c=color, s=30, label=f'{prefix}(ID:{nid})')

            _plot_nbr('LeftBehindID' if side == 'left' else 'RightBehindID', '#8e44ad', '目标车道后车')
            _plot_nbr('LeftFrontID' if side == 'left' else 'RightFrontID', '#e84393', '目标车道前车')
            _plot_nbr('LeftSideID' if side == 'left' else 'RightSideID', '#1a5276', '目标车道邻车')
            _plot_nbr('FrontID', '#2ecc71', '前车')
            _plot_nbr('BehindID', '#e67e22', '车道后车')

    # 车道线
    x_vals = np.linspace(x_range[0], x_range[1], 300)
    for c in curves:
        y_vals = np.polyval(c, x_vals)
        ax.plot(x_vals, y_vals, color='white', linewidth=2.5, alpha=0.95)
        ax.plot(x_vals, y_vals, color='black', linewidth=0.6, linestyle='--', alpha=0.6)

    ax.set_xlim(x_range)
    ax.set_ylim(ax.get_ylim()[::-1])

    y_median = np.median(veh_sample['Y'])
    if y_median > 25:
        ax.annotate('行驶方向 →', xy=(0.5, 0.03), xycoords='axes fraction',
                    fontsize=12, fontweight='bold', color='#27ae60',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    else:
        ax.annotate('← 行驶方向', xy=(0.5, 0.03), xycoords='axes fraction',
                    fontsize=12, fontweight='bold', color='#27ae60',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='upper right', markerscale=0.8)
    ax.set_title('XY 轨迹', fontsize=13, fontweight='bold')


def panel_acceleration(ax, veh_sample):
    frames = veh_sample['Frame'].values
    ax.plot(frames, veh_sample['long_Acc'].values, color='#2c3e50', linewidth=1.2, label='纵向加速度')
    ax.plot(frames, veh_sample['lat_Acc'].values, color='#e74c3c', linewidth=1.2, label='横向加速度')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Acc (m/s^2)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.set_title('纵/横向加速度', fontsize=13, fontweight='bold')


def panel_surrounding(ax, veh_sample, side='left'):
    frames = veh_sample['Frame'].values

    # 前车距 — 绿色（同车道前车）
    fd = veh_sample['Following_dist'].values
    if fd.max() > 0:
        ax.plot(frames, fd, color='#27ae60', linewidth=3.0, marker='o', markersize=2.5, label='前车距')

    # 后车距 — 橙色（同车道后车）
    bd = veh_sample['B_Dist'].values
    if bd.max() > 0:
        ax.plot(frames, bd, color='#e67e22', linewidth=3.0, marker='o', markersize=2.5, label='后车距')

    # 目标车道前距 — 粉色（按变道方向选 LF_Dist 或 RF_Dist）
    targ_front_col = 'LF_Dist' if side == 'left' else 'RF_Dist'
    tfd = veh_sample[targ_front_col].values
    if tfd.max() > 0:
        ax.plot(frames, tfd, color='#e84393', linewidth=2.0, marker='o', markersize=2.0, label='目标车道前距')

    # 目标车道后距 — 紫色（按变道方向选 LB_Dist 或 RB_Dist）
    targ_rear_col = 'LB_Dist' if side == 'left' else 'RB_Dist'
    trd = veh_sample[targ_rear_col].values
    if trd.max() > 0:
        ax.plot(frames, trd, color='#8e44ad', linewidth=2.0, marker='o', markersize=2.0, label='目标车道后距')

    # 目标车道侧距 — 深蓝色（按变道方向选 LS_Dist 或 RS_Dist）
    targ_side_col = 'LS_Dist' if side == 'left' else 'RS_Dist'
    tsd = veh_sample[targ_side_col].values
    if tsd.max() > 0:
        ax.plot(frames, tsd, color='#1a5276', linewidth=2.0, marker='o', markersize=2.0, label='目标车道侧距')

    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.tick_params(labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title('周边车距时序', fontsize=13, fontweight='bold')


def panel_safety_metrics(ax, veh_sample):
    frames = veh_sample['Frame'].values
    mttc = veh_sample['mTTC'].values
    thw = veh_sample['Time_Headway'].values
    f_ettc = veh_sample['F_ETTC'].values if 'F_ETTC' in veh_sample.columns else None

    # mTTC — 橙色
    mask_m = (mttc > 0) & np.isfinite(mttc) & (mttc < 20)
    if mask_m.any():
        ax.plot(frames[mask_m], mttc[mask_m], color='#d35400', linewidth=1.2, label='mTTC')

    # THW — 紫色
    mask_h = thw > 0
    if mask_h.any():
        ax.plot(frames[mask_h], thw[mask_h], color='#8e44ad', linewidth=1.2, label='THW')

    # F_ETTC — 亮粉色（仅绘制有限的合理范围，避免极高值压扁纵轴）
    if f_ettc is not None:
        mask_f = (f_ettc > 0) & np.isfinite(f_ettc) & (f_ettc < 20)
        if mask_f.any():
            ax.plot(frames[mask_f], f_ettc[mask_f], color='#ff4757', linewidth=1.2,
                    marker='o', markersize=2, alpha=0.8, label='F_ETTC')

    # PET — 绿色横线
    pet_val = veh_sample['PET'].iloc[0]
    if np.isfinite(pet_val) and pet_val > 0:
        x_start, x_end = frames.min(), frames.max()
        ax.plot([x_start, x_end], [pet_val, pet_val], color='#27ae60', linewidth=1.5,
                linestyle='-', alpha=0.8, label=f'PET={pet_val:.2f}s')
        ax.text(x_end + (x_end - x_start) * 0.01, pet_val, f'{pet_val:.2f}s',
                fontsize=9, color='#27ae60', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))

    # OL_PET — 亮青色横线
    ol_pet_val = veh_sample['OL_PET'].iloc[0]
    if np.isfinite(ol_pet_val) and ol_pet_val > 0:
        x_start = frames.min()
        ax.plot([x_start, frames.max()], [ol_pet_val, ol_pet_val], color='#00d4aa', linewidth=1.5,
                linestyle='-', alpha=0.8, label=f'OL_PET={ol_pet_val:.2f}s')

    ax.axhline(y=2, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5, label='危险(2s)')
    ax.axhline(y=5, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.5, label='谨慎(5s)')
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('安全指标时序', fontsize=13, fontweight='bold')


def panel_velocity_components(ax, veh_sample):
    frames = veh_sample['Frame'].values
    long_vel = veh_sample['long_Vel'].values * 3.6  # m/s → km/h
    lat_vel = veh_sample['lat_Vel'].values          # 保留 m/s

    ax2 = ax.twinx()
    line1, = ax.plot(frames, long_vel, color='#2c3e50', linewidth=1.5, label='纵向速度')
    line2, = ax2.plot(frames, lat_vel, color='#e74c3c', linewidth=1.5, label='横向速度')

    # 左轴 — 纵向速度按数据范围缩放（km/h）
    l_min, l_max = long_vel.min(), long_vel.max()
    l_margin = max((l_max - l_min) * 0.1, 5)
    ax.set_ylim(l_min - l_margin, l_max + l_margin)

    # 右轴 — 横向速度按数据范围缩放（m/s）
    r_min, r_max = lat_vel.min(), lat_vel.max()
    r_margin = max((r_max - r_min) * 0.1, 0.3)
    ax2.set_ylim(r_min - r_margin, r_max + r_margin)

    ax.axhline(y=0, color='#2c3e50', linewidth=0.4, linestyle=':', alpha=0.4)
    ax2.axhline(y=0, color='#e74c3c', linewidth=0.4, linestyle=':', alpha=0.4)
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('纵向速度 (km/h)', fontsize=11, color='#2c3e50')
    ax2.set_ylabel('横向速度 (m/s)', fontsize=11, color='#e74c3c')
    ax.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)
    lines = [line1, line2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=11)
    ax.set_title('纵/横向速度', fontsize=13, fontweight='bold')


# worst_cat — safety_scoring 模块; risk_score/risk_label — safety_scoring_exp 模块


def panel_risk_summary(ax, veh_sample, loc_name=''):
    ax.axis('off')
    cats = {
        'F_ETTC': worst_cat(veh_sample['F_ETTC_cat']) if 'F_ETTC_cat' in veh_sample.columns else 'safe',
        'PET':    worst_cat(veh_sample['PET_cat']),
        'mTTC':   worst_cat(veh_sample['mTTC_cat']),
        'THW':    worst_cat(veh_sample['THW_cat']),
        'OL_PET': worst_cat(veh_sample['OL_PET_cat']),
    }
    row0 = veh_sample.iloc[0]
    has_front = bool(row0.get('has_front_vehicle', False))
    has_rear = bool(row0.get('has_rear_vehicle', False))
    score = exp_risk_score(veh_sample, v0_kmh=80 if loc_name=='location5' else 100)
    overall, overall_color = exp_risk_label(score, 'lane_change')

    col_labels = ['指标', '分类', '含义']
    cell_text = []
    cell_colors = []
    for label, cat_val in cats.items():
        if cat_val == 'dangerous':
            meaning, bg = '<2s', '#fddede'
        elif cat_val == 'cautious':
            meaning, bg = '2-5s', '#fef5e7'
        elif cat_val == 'safe':
            meaning, bg = '≥5s', '#d5f5e3'
        elif cat_val in ('no_leader', 'no_follower'):
            meaning, bg = '无目标', '#eaecee'
        else:
            meaning, bg = 'N/A', '#ecf0f1'
        cell_text.append([label, cat_val, meaning])
        cell_colors.append([bg, bg, bg])

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors,
                     cellLoc='center', loc='upper center', colWidths=[0.22, 0.28, 0.2],
                     colColours=['#34495e', '#34495e', '#34495e'])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for j in range(3):
        table[0, j].set_text_props(color='white', fontweight='bold')

    status_text = (f"前车: {'有' if has_front else '无'}    "
                   f"后车: {'有' if has_rear else '无'}\n"
                   f"综合风险: {overall}")
    ax.text(0.5, 0.12, status_text, transform=ax.transAxes, fontsize=13,
            ha='center', va='center', fontweight='bold', color=overall_color)
    ax.set_title('风险指标汇总', fontsize=13, fontweight='bold', y=0.98)


# ==================== 主绘制函数 ====================
def plot_one_vehicle(fig, axes, vid, veh_sample, veh_traj, curves, x_range,
                     smooth_df=None, side='left', loc_name=''):
    a = axes.flatten()
    src = veh_sample['Source'].iloc[0]
    panel_xy_trajectory(a[0], veh_traj, veh_sample, curves, x_range,
                        smooth_df, side, src, loc_name)
    panel_acceleration(a[1], veh_sample)
    panel_surrounding(a[2], veh_sample, side)
    panel_safety_metrics(a[3], veh_sample)
    panel_velocity_components(a[4], veh_sample)
    panel_risk_summary(a[5], veh_sample, loc_name)


# ==================== 主流程 ====================
def main():
    from safety_scoring_exp import risk_score, risk_label

    rng = np.random.default_rng(SEED)
    total_figs = 0

    # 全量标签统计
    label_counts = {'高风险': 0, '中风险': 0, '低风险': 0}
    label_per_loc = {}
    for loc_name in LOCS:
        label_per_loc[loc_name] = {'高风险': 0, '中风险': 0, '低风险': 0}
        for side in ['left', 'right']:
            sample_df = load_sample(loc_name, side)
            for (vid, src), grp in sample_df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                risk = risk_score(grp, v0_kmh=80 if loc_name == 'location5' else 100)
                tag, _ = risk_label(risk, 'lane_change')
                label_counts[tag] += 1
                label_per_loc[loc_name][tag] += 1

    print(f"\n{'=' * 50}")
    print("全量标签统计")
    print(f"{'=' * 50}")
    total_n = sum(label_counts.values())
    print(f"  {'高风险':>6s}: {label_counts['高风险']:4d}  ({label_counts['高风险']/total_n*100:.1f}%)")
    print(f"  {'中风险':>6s}: {label_counts['中风险']:4d}  ({label_counts['中风险']/total_n*100:.1f}%)")
    print(f"  {'低风险':>6s}: {label_counts['低风险']:4d}  ({label_counts['低风险']/total_n*100:.1f}%)")
    print(f"  {'合计':>4s}: {total_n:4d}")
    print(f"\n  {'Location':<15s} {'高风险':>6s} {'中风险':>6s} {'低风险':>6s}")
    print(f"  {'-'*39}")
    for loc_name in LOCS:
        cnt = label_per_loc[loc_name]
        print(f"  {loc_name:<15s} {cnt['高风险']:>6d} {cnt['中风险']:>6d} {cnt['低风险']:>6d}")

    for loc_name in LOCS:
        cfg = LOCS[loc_name]
        print(f"\n{'=' * 50}")
        print(f"处理 {loc_name}")

        traj_df = pd.read_csv(cfg['traj_path'])
        coeffs_df = pd.read_csv(cfg['coeffs_path'])

        smooth_path = cfg['traj_path'].replace('trajectory_full.csv',
                                                'trajectory_full_smoothed.csv')
        if os.path.exists(smooth_path):
            smooth_df = pd.read_csv(smooth_path)
            print(f"  已加载平滑轨迹: {os.path.basename(smooth_path)}")
        else:
            smooth_df = None
            print(f"  [!] 未找到平滑轨迹文件，跳过邻车绘制")

        for side in ['left', 'right']:
            side_cn = '左变道' if side == 'left' else '右变道'
            sample_df = load_sample(loc_name, side)
            # 按 (ID, Source) 成对选择，避免同一 ID 跨 Source 被合并
            pair_df = sample_df[['ID', 'Source']].drop_duplicates()
            veh_pairs = list(pair_df.itertuples(index=False, name=None))
            n_select = min(N_PER_SIDE, len(veh_pairs))

            if n_select == 0:
                print(f"  {side_cn}: 无数据，跳过")
                continue

            selected_pairs = [veh_pairs[i] for i in
                              rng.choice(len(veh_pairs), n_select, replace=False)]
            print(f"  {side_cn}: {n_select}/{len(veh_pairs)} 辆车")

            for vid, src in selected_pairs:
                veh_sample = sample_df[(sample_df['ID'] == vid) & (sample_df['Source'] == src)]
                if veh_sample.empty:
                    continue

                direction = int(veh_sample['Direction'].iloc[0])

                veh_traj = traj_df[(traj_df['ID'] == vid) & (traj_df['Source'] == src)]

                coeffs_match = coeffs_df[(coeffs_df['where'] == src) &
                                         (coeffs_df['direction'] == direction)]
                curves = []
                for _, cr in coeffs_match.iterrows():
                    curves.append([cr['a5'], cr['a4'], cr['a3'], cr['a2'], cr['a1'], cr['a0']])

                all_x = pd.concat([veh_traj['X'], veh_sample['X']], ignore_index=True)
                x_pad = 5
                x_range = (all_x.min() - x_pad, all_x.max() + x_pad)

                fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)
                plot_one_vehicle(fig, axes, vid, veh_sample, veh_traj, curves, x_range,
                                 smooth_df, side, loc_name)

                fig.suptitle(f'{src} {side_cn} ID:{int(vid)} 多维分析',
                             fontsize=18, fontweight='bold', y=0.995)
                fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.05,
                                    hspace=0.35, wspace=0.3)

                out_name = f'{loc_name}_{side}_{int(vid)}_{src}.png'
                out_path = os.path.join(OUT_DIR, out_name)
                fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                total_figs += 1

    print(f"\n{'=' * 50}")
    print(f"全部完成! 共 {total_figs} 张图 → {OUT_DIR}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
