"""
跟驰车辆多维可视化 — 每张图 3×2=6 面板
从 visualize_per_vehicle_all.py 分离的跟驰专用版
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'traffic_full'))
from safety_scoring import worst_cat
from safety_scoring_exp import risk_score as exp_risk_score, risk_label as exp_risk_label

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
SEED = 39
N_PER_SOURCE = 5       # 每 source 选的车辆数（与变道 N_PER_SIDE 对齐）
DPI = 80
FIGSIZE = (26, 13)

LOCS = {
    'location1': {
        'sample_dir': 'E:/0little/location1',
        'traj_path': 'E:/0little/location1/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location1/lane_coeffs.csv',
    },
    'location2': {
        'sample_dir': 'E:/0little/location2',
        'traj_path': 'E:/0little/location2/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location2/lane_coeffs.csv',
    },
    'location3': {
        'sample_dir': 'E:/0little/location3',
        'traj_path': 'E:/0little/location3/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location3/lane_coeffs.csv',
    },
    'location4': {
        'sample_dir': 'E:/0little/location4',
        'traj_path': 'E:/0little/location4/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location4/lane_coeffs.csv',
    },
    'location5': {
        'sample_dir': 'E:/0little/location5',
        'traj_path': 'E:/0little/location5/trajectory_full.csv',
        'coeffs_path': 'E:/0little/location5/lane_coeffs.csv',
    },
}

OUT_DIR = os.path.join('E:/0little/data_statistics', 'visualize_per_vehicle_all_output', 'following')
os.makedirs(OUT_DIR, exist_ok=True)


# ==================== 辅助函数 ====================
def _ptsize(length_series):
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
                   s=_ptsize(veh_sample['Length']), alpha=0.85, label='跟驰样本')
        rect_idx = list(range(0, len(veh_sample), 8))
        if rect_idx[-1] != len(veh_sample) - 1:
            rect_idx.append(len(veh_sample) - 1)
        for _, r in veh_sample.iloc[rect_idx].iterrows():
            rect = Rectangle((r['X'] - r['Length'] / 2, r['Y'] - r['Width'] / 2),
                             r['Length'], r['Width'],
                             facecolor='#e74c3c', edgecolor='white', linewidth=0.5, alpha=0.55)
            ax.add_patch(rect)

    if smooth_df is not None and len(smooth_df) > 0:
        ego_smooth = smooth_df[(smooth_df['ID'] == ego_id) & (smooth_df['Source'] == src)
                               & (smooth_df['Frame'].isin(sample_frames))]
        if len(ego_smooth) > 0:
            def _plot_nbr(id_col, color, label_prefix):
                """绘制该位置的主导邻车（最频繁出现的 ID）"""
                ids = ego_smooth[id_col].values
                ids = ids[(ids > 0) & (ids != ego_id)]
                if len(ids) == 0:
                    return
                # 取最频繁出现的邻车 ID（主导车辆）
                nid = int(pd.Series(ids).mode().iloc[0])
                # 限制同 Source（不同 Source 是不同时段，不混用）
                traj = smooth_df[(smooth_df['ID'] == nid) & (smooth_df['Source'] == src)
                                 & (smooth_df['Frame'].isin(sample_frames))].copy()
                if len(traj) == 0:
                    return
                # 小圆点显示完整路径
                ax.scatter(traj['X'], traj['Y'], c=color, s=3, alpha=0.35)
                # 每隔 8 帧绘制车身矩形
                rect_idx = list(range(0, len(traj), 8))
                if rect_idx[-1] != len(traj) - 1:
                    rect_idx.append(len(traj) - 1)
                for _, r in traj.iloc[rect_idx].iterrows():
                    rect = Rectangle((r['X'] - r['Length'] / 2, r['Y'] - r['Width'] / 2),
                                     r['Length'], r['Width'],
                                     facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.55)
                    ax.add_patch(rect)
                ax.scatter([], [], c=color, s=30, label=f'{label_prefix}(ID:{nid})')

            _plot_nbr('LeftFrontID', '#e84393', '左前')
            _plot_nbr('LeftSideID', '#1a5276', '左侧')
            _plot_nbr('LeftBehindID', '#8e44ad', '左后')
            _plot_nbr('FrontID', '#2ecc71', '前车')
            _plot_nbr('RightFrontID', '#fd79a8', '右前')
            _plot_nbr('RightSideID', '#00b894', '右侧')
            _plot_nbr('RightBehindID', '#6c5ce7', '右后')
            _plot_nbr('BehindID', '#e67e22', '后车')

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


def panel_surrounding(ax, veh_sample, side='following'):
    """绘制 8 方向邻车距离时序（配色与 XY 轨迹面板一致）"""
    frames = veh_sample['Frame'].values
    # 8 方向配置：(列名, 颜色, 图例标签)
    nbr_config = [
        ('Following_dist', '#2ecc71',  '前车'),
        ('B_Dist',         '#e67e22',  '后车'),
        ('LF_Dist',        '#e84393',  '左前'),
        ('LS_Dist',        '#1a5276',  '左侧'),
        ('LB_Dist',        '#8e44ad',  '左后'),
        ('RF_Dist',        '#fd79a8',  '右前'),
        ('RS_Dist',        '#00b894',  '右侧'),
        ('RB_Dist',        '#6c5ce7',  '右后'),
    ]
    for col, color, label in nbr_config:
        if col not in veh_sample.columns:
            continue
        vals = veh_sample[col].values.astype(float)
        # 邻车不存在时距离=0/inf，设为 NaN 避免突变为 0 的误导线段
        vals = np.where((vals > 0) & np.isfinite(vals), vals, np.nan)
        if np.all(np.isnan(vals)):
            continue
        ax.plot(frames, vals, color=color, linewidth=1.5, marker='o', markersize=1.5,
                alpha=0.8, label=label)

    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Distance (m)', fontsize=11)
    ax.tick_params(labelsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.set_title('周边车距时序', fontsize=13, fontweight='bold')


def panel_safety_metrics(ax, veh_sample):
    frames = veh_sample['Frame'].values
    mttc = veh_sample['mTTC'].values
    thw = veh_sample['Time_Headway'].values
    f_ettc = veh_sample['F_ETTC'].values if 'F_ETTC' in veh_sample.columns else None
    mask_m = (mttc > 0) & np.isfinite(mttc) & (mttc < 20)
    if mask_m.any():
        ax.plot(frames[mask_m], mttc[mask_m], color='#d35400', linewidth=1.2, label='mTTC')
    mask_h = thw > 0
    if mask_h.any():
        ax.plot(frames[mask_h], thw[mask_h], color='#8e44ad', linewidth=1.2, label='THW')
    if f_ettc is not None:
        mask_f = (f_ettc > 0) & np.isfinite(f_ettc) & (f_ettc < 20)
        if mask_f.any():
            ax.plot(frames[mask_f], f_ettc[mask_f], color='#ff4757', linewidth=1.2,
                    marker='o', markersize=2, alpha=0.8, label='F_ETTC')
    pet_val = veh_sample['PET'].iloc[0]
    if np.isfinite(pet_val) and pet_val > 0:
        x_start, x_end = frames.min(), frames.max()
        ax.plot([x_start, x_end], [pet_val, pet_val], color='#27ae60', linewidth=1.5,
                linestyle='-', alpha=0.8, label=f'PET={pet_val:.2f}s')
        ax.text(x_end + (x_end - x_start) * 0.01, pet_val, f'{pet_val:.2f}s',
                fontsize=9, color='#27ae60', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    ol_pet_val = veh_sample['OL_PET'].iloc[0]
    if np.isfinite(ol_pet_val) and ol_pet_val > 0:
        ax.plot([frames.min(), frames.max()], [ol_pet_val, ol_pet_val], color='#00d4aa', linewidth=1.5,
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
    long_vel = veh_sample['long_Vel'].values * 3.6
    lat_vel = veh_sample['lat_Vel'].values
    ax2 = ax.twinx()
    line1, = ax.plot(frames, long_vel, color='#2c3e50', linewidth=1.5, label='纵向速度')
    line2, = ax2.plot(frames, lat_vel, color='#e74c3c', linewidth=1.5, label='横向速度')
    l_min, l_max = long_vel.min(), long_vel.max()
    l_margin = max((l_max - l_min) * 0.1, 5)
    ax.set_ylim(l_min - l_margin, l_max + l_margin)
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
    ax.legend([line1, line2], [l.get_label() for l in [line1, line2]], fontsize=11)
    ax.set_title('纵/横向速度', fontsize=13, fontweight='bold')


def panel_risk_summary(ax, veh_sample, loc_name='', scenario='following'):
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
    score = exp_risk_score(veh_sample, v0_kmh=80 if loc_name == 'location5' else 100)
    overall, overall_color = exp_risk_label(score, scenario)

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
                     smooth_df=None, side='following', loc_name='', scenario='following'):
    a = axes.flatten()
    src = veh_sample['Source'].iloc[0]
    panel_xy_trajectory(a[0], veh_traj, veh_sample, curves, x_range,
                        smooth_df, side, src, loc_name)
    panel_acceleration(a[1], veh_sample)
    panel_surrounding(a[2], veh_sample, side)
    panel_safety_metrics(a[3], veh_sample)
    panel_velocity_components(a[4], veh_sample)
    panel_risk_summary(a[5], veh_sample, loc_name, scenario)


# ==================== 主流程 ====================
def main():
    rng = np.random.default_rng(SEED)
    total_figs = 0

    for loc_name in LOCS:
        cfg = LOCS[loc_name]
        f_fp = os.path.join(cfg['sample_dir'], 'traffic_following_change.csv')
        if not os.path.exists(f_fp):
            print(f"  {loc_name}: 无跟驰数据，跳过")
            continue

        sample_df = pd.read_csv(f_fp, low_memory=False)
        n_before = len(sample_df)
        sample_df = sample_df.drop_duplicates(subset=['ID', 'Source', 'Frame'])
        if len(sample_df) < n_before:
            print(f"  {loc_name}: 去重 {n_before - len(sample_df)} 行")

        traj_df = pd.read_csv(cfg['traj_path'].replace('trajectory_full.csv',
                                     'trajectory_full_smoothed.csv'), low_memory=False)
        traj_df = traj_df.drop_duplicates(subset=['ID', 'Source', 'Frame'])
        smooth_df = traj_df  # 用于邻车检索（含 FrontID/BehindID 等列）
        coeffs_df = pd.read_csv(cfg['coeffs_path'])

        # 按 Source 分组，每 Source 选 N_PER_SOURCE 辆
        selected_pairs = []
        for src_val in sample_df['Source'].unique():
            avail = sample_df[sample_df['Source'] == src_val][['ID', 'Source']].drop_duplicates()
            avail_pairs = list(avail.itertuples(index=False, name=None))
            n = min(N_PER_SOURCE, len(avail_pairs))
            if n == 0:
                continue
            chosen = [avail_pairs[i] for i in rng.choice(len(avail_pairs), n, replace=False)]
            selected_pairs.extend(chosen)

        n_select = len(selected_pairs)
        if n_select == 0:
            continue
        print(f"  {loc_name}: {n_select} 辆跟驰 ({sample_df['Source'].nunique()} sources)")

        for vid, src in selected_pairs:
            veh_sample = sample_df[(sample_df['ID'] == vid) & (sample_df['Source'] == src)]
            if veh_sample.empty:
                continue
            direction = int(veh_sample['Direction'].iloc[0])
            veh_traj = traj_df[(traj_df['ID'] == vid) & (traj_df['Source'] == src)]
            coeffs_match = coeffs_df[(coeffs_df['where'] == src) &
                                     (coeffs_df['direction'] == direction)]
            curves = [[cr['a5'], cr['a4'], cr['a3'], cr['a2'], cr['a1'], cr['a0']]
                      for _, cr in coeffs_match.iterrows()]
            all_x = pd.concat([veh_traj['X'], veh_sample['X']], ignore_index=True)
            x_range = (all_x.min() - 5, all_x.max() + 5)

            fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)
            plot_one_vehicle(fig, axes, vid, veh_sample, veh_traj, curves, x_range,
                             smooth_df=smooth_df, side='following', loc_name=loc_name,
                             scenario='following')
            fig.suptitle(f'{src} 跟驰 ID:{int(vid)} 多维分析',
                         fontsize=18, fontweight='bold', y=0.995)
            fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.05,
                                hspace=0.35, wspace=0.3)

            out_name = f'{loc_name}_following_{int(vid)}_{src}.png'
            fig.savefig(os.path.join(OUT_DIR, out_name), dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            total_figs += 1

    print(f"\n全部完成! 共 {total_figs} 张图 → {OUT_DIR}")


if __name__ == '__main__':
    main()
