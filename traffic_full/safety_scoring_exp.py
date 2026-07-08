"""
安全评价评分 — 连续风险版 (exp(-x/k) + 加权)

与 safety_scoring.py 的区别:
  - 不再使用类别打分 (dangerous=2, cautious=1)，改用 exp(-值/k) 连续映射
  - 无对应车辆时贡献=0，不参与权重归一化
  - 暂不设定高风险/中风险/低风险阈值，仅输出连续分

用法:
    from safety_scoring_exp import risk_score, risk_label
"""

import numpy as np
import os
import pandas as pd

B = 1.3  # 速度修正参数

# ---------- 风险等级阈值 ----------
# 中风险 ≥ thresh_mid, 高风险 ≥ thresh_high
THRESH_LANE_CHANGE = {'mid': 0.40, 'high': 0.60}  # 变道车辆
THRESH_FOLLOWING   = {'mid': 0.20, 'high': 0.35}  # 跟驰车辆

# ---------- 指标配置（变道车辆）当前使用：专家经验权重 ----------
METRICS = [
    {'name': 'mTTC',         'col': 'mTTC',         'w': 0.25, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',          'col': 'Time_Headway',  'w': 0.2, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'PET',          'col': 'PET',           'w': 0.2, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
    {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.2, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.15, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
]
# ---------- 指标配置（变道车辆）当前使用：专家经验权重2 ----------
# METRICS = [
#     {'name': 'mTTC',         'col': 'mTTC',         'w': 0.25, 'k': 20.80,
#      'valid': lambda v: (v > 0)},
#     {'name': 'THW',          'col': 'Time_Headway',  'w': 0.2, 'k': 6.30,
#      'valid': lambda v: (v > 0)},
#     {'name': 'PET',          'col': 'PET',           'w': 0.2, 'k': 5.30,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
#     {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.2, 'k': 25.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.15, 'k': 7.3,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
# ]
# ---------- 指标配置（跟驰车辆）当前使用：专家经验权重 ----------
FOLLOWING_METRICS = [
    {'name': 'mTTC',    'col': 'mTTC',    'w': 0.4, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',     'col': 'Time_Headway', 'w': 0.3, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'B_mTTC',  'col': 'B_mTTC',  'w': 0.4, 'k': 12.0,
     'valid': lambda v: (v > 0)},
]

# ===== 备选权重：EWM 客观权重（取消注释使用） =====
# METRICS = [
#     {'name': 'mTTC',         'col': 'mTTC',         'w': 0.247, 'k': 12.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'THW',          'col': 'Time_Headway',  'w': 0.083, 'k': 6.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'PET',          'col': 'PET',           'w': 0.131, 'k': 12.0,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
#     {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.356, 'k': 12.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.183, 'k': 12.0,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
# ]
# METRICS = [
#     {'name': 'mTTC',         'col': 'mTTC',         'w': 0.247, 'k': 20.8,
#      'valid': lambda v: (v > 0)},
#     {'name': 'THW',          'col': 'Time_Headway',  'w': 0.083, 'k': 6.3,
#      'valid': lambda v: (v > 0)},
#     {'name': 'PET',          'col': 'PET',           'w': 0.131, 'k': 5.3,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
#     {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.356, 'k': 25.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.183, 'k': 7.3,
#      'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
# ]
#
# FOLLOWING_METRICS = [
#     {'name': 'mTTC',    'col': 'mTTC',    'w': 0.463, 'k': 12.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'THW',     'col': 'Time_Headway', 'w': 0.114, 'k': 6.0,
#      'valid': lambda v: (v > 0)},
#     {'name': 'B_mTTC',  'col': 'B_mTTC',  'w': 0.423, 'k': 12.0,
#      'valid': lambda v: (v > 0)},
# ]

def _frame_contrib(val, k):
    """单帧的风险贡献: exp(-val/k)，val 越小、贡献越接近 1"""
    return np.exp(-val / k)


def risk_score(grp, v0_kmh=100):
    """计算一辆车的连续风险分 (0~1+，越高越危险)

    参数:
        grp: 单辆车的 50 帧 DataFrame
        v0_kmh: 道路基准速度 (km/h)，location1-4→100, location5→80

    返回:
        float: 风险分（未归一化上限，通常 0~1 之间）
    """
    score = 0.0

    for m in METRICS:
        if m['col'] not in grp.columns:
            continue

        vals = grp[m['col']].values.astype(float)
        valid = m['valid'](vals)

        if not valid.any():
            # 无对应车辆 → 该指标不贡献
            continue

        # 取 50 帧中 exp(-val/k) 的最大值（等同 worst_cat 的取最差思路）
        contrib = np.nanmax(_frame_contrib(vals[valid], m['k']))
        score += m['w'] * contrib

    # 速度修正系数 K
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        k_speed = 1.0
    else:
        v0 = v0_kmh / 3.6
        k_speed = 1.0 if v85 <= v0 else (v85 / v0) ** B

    return score * k_speed


def following_risk_score(grp, v0_kmh=100):
    """计算跟驰车辆的连续风险分（与变道车辆同尺度，可直接对比）

    仅用前车(mTTC/THW) + 后车(B_mTTC)三项，
    权重已归一化到和为 1，乘速度修正后与 risk_score 对齐。
    """
    score = 0.0

    for m in FOLLOWING_METRICS:
        if m['col'] not in grp.columns:
            continue

        vals = grp[m['col']].values.astype(float)
        valid = m['valid'](vals)

        if not valid.any():
            continue

        contrib = np.nanmax(_frame_contrib(vals[valid], m['k']))
        score += m['w'] * contrib

    # 速度修正系数 K（同 risk_score）
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        k_speed = 1.0
    else:
        v0 = v0_kmh / 3.6
        k_speed = 1.0 if v85 <= v0 else (v85 / v0) ** B

    return score * k_speed


def risk_label(score, scenario='lane_change'):
    """将风险分转为中文标签

    参数:
        score: 连续风险分
        scenario: 'lane_change' 或 'following'，不同场景阈值不同

    阈值由题头 THRESH_LANE_CHANGE / THRESH_FOLLOWING 全局参数控制。
    """
    t = THRESH_LANE_CHANGE if scenario == 'lane_change' else THRESH_FOLLOWING

    if score >= t['high']:
        return '高风险', '#e74c3c'
    elif score >= t['mid']:
        return '中风险', '#f39c12'
    return '低风险', '#27ae60'


# ── K-means 风险等级聚类（替代阈值标签）──


def _collect_scores(data_type='lc'):
    """读取所有 CSV 并计算各车的风险评分"""
    locs = {f'location{i}': f'E:/0little/location{i}' for i in range(1, 6)}
    loc_v0 = {f'location{i}': 100 if i < 5 else 80 for i in range(1, 6)}

    scores = []
    func = risk_score if data_type == 'lc' else following_risk_score

    for loc_key, base_dir in locs.items():
        v0 = loc_v0[loc_key]
        if data_type == 'lc':
            files = [os.path.normpath(f'{base_dir}/traffic_left_change.csv'),
                     os.path.normpath(f'{base_dir}/traffic_right_change.csv')]
        else:
            files = [os.path.normpath(f'{base_dir}/traffic_following_change.csv')]

        for fp in files:
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            for (vid, src), grp in df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                scores.append(func(grp, v0))

    return np.array(scores)


def run_kmeans_threshold():
    """对风险评分运行 K-means (K=3) 聚类，输出新阈值并绘图"""
    from sklearn.cluster import KMeans
    from scipy.stats import gaussian_kde

    print('\n' + '=' * 72)
    print('  K-means 一维风险评分聚类 (K=3)')
    print('=' * 72)

    results = {}

    for data_type, scenario_name in [('lc', 'Lane Change'), ('fl', 'Following')]:
        scores = _collect_scores(data_type)
        n = len(scores)
        if n == 0:
            continue

        # K-means clustering
        X = scores.reshape(-1, 1)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Sort clusters by center value (low → mid → high)
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        centers_sorted = centers[order]
        label_map = {old: new for new, old in enumerate(order)}
        labels_mapped = np.array([label_map[l] for l in labels])

        # Statistics
        counts = [int(np.sum(labels_mapped == i)) for i in range(3)]
        thresholds = {
            'mid': (centers_sorted[0] + centers_sorted[1]) / 2,
            'high': (centers_sorted[1] + centers_sorted[2]) / 2,
        }
        cur = THRESH_LANE_CHANGE if data_type == 'lc' else THRESH_FOLLOWING

        # K-means 边界 → 重新统计
        labels_new = np.zeros(n, dtype=int)
        labels_new[scores >= thresholds['high']] = 2
        labels_new[(scores >= thresholds['mid']) & (scores < thresholds['high'])] = 1

        print(f'\n  >> {scenario_name} (n={n})')
        print(f'  {"":>12s} {"Low Risk":>10s} {"Mid Risk":>10s} {"High Risk":>10s}')
        print(f'  {"Cluster Center":>12s} {centers_sorted[0]:10.4f} '
              f'{centers_sorted[1]:10.4f} {centers_sorted[2]:10.4f}')
        print(f'  {"Count":>12s} {counts[0]:>10d} {counts[1]:>10d} {counts[2]:>10d}')
        print(f'  {"Proportion":>12s} {counts[0]/n*100:9.1f}% '
              f'{counts[1]/n*100:9.1f}% {counts[2]/n*100:9.1f}%')
        print(f'  K-means thresholds:  mid >= {thresholds["mid"]:.4f},  '
              f'high >= {thresholds["high"]:.4f}')
        print(f'  Current thresholds:  mid >= {cur["mid"]:.2f},  '
              f'high >= {cur["high"]:.2f}')

        new_pcts = [np.sum(labels_new == i) / n * 100 for i in range(3)]
        print(f'  K-means labels:  Low={int(np.sum(labels_new==0))}({new_pcts[0]:.1f}%)  '
              f'Mid={int(np.sum(labels_new==1))}({new_pcts[1]:.1f}%)  '
              f'High={int(np.sum(labels_new==2))}({new_pcts[2]:.1f}%)')

        results[data_type] = {
            'scores': scores, 'centers': centers_sorted,
            'counts': counts, 'thresholds': thresholds,
            'current': cur, 'scenario_name': scenario_name,
        }

    _plot_kmeans_thresholds(results)


def _plot_kmeans_thresholds(results):
    """绘制 K-means 聚类结果（跟驰含低分放大窗）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    n_types = len(results)
    fig, axes = plt.subplots(1, n_types, figsize=(n_types * 6, 4))
    if n_types == 1:
        axes = [axes]

    colors = ['#27ae60', '#f39c12', '#e74c3c']

    for ax_idx, (data_type, res) in enumerate(results.items()):
        scores = res['scores']
        centers = res['centers']
        thresh = res['thresholds']
        cur = res['current']
        name = res['scenario_name']

        ax = axes[ax_idx]

        zoom_xmax = 0.02
        main_scores = scores[scores > zoom_xmax]
        zoom_scores = scores[scores <= zoom_xmax]

        # Main histogram (exclude 0-0.02 region)
        ax.hist(main_scores, bins=50, color='gray', alpha=0.35, edgecolor='white', density=True)

        # KDE curve (on main_scores only)
        try:
            kde = gaussian_kde(main_scores)
            xs = np.linspace(main_scores.min(), main_scores.max(), 200)
            ax.plot(xs, kde(xs), 'k-', linewidth=1.5, alpha=0.7)
        except Exception:
            pass

        # Cluster centers (dashed colored lines)
        for i, c in enumerate(centers):
            ax.axvline(c, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f'C{i} center={c:.3f}')

        # K-means thresholds (navy dotted lines)
        ax.axvline(thresh['mid'], color='navy', linestyle=':', linewidth=1.5,
                   label=f'Kmeans mid={thresh["mid"]:.3f}')
        ax.axvline(thresh['high'], color='navy', linestyle=':', linewidth=1.5,
                   label=f'Kmeans high={thresh["high"]:.3f}')

        # Current thresholds (red dashed)
        ax.axvline(cur['mid'], color='red', linestyle='-', linewidth=1, alpha=0.6,
                   label=f'Current mid={cur["mid"]:.2f}')
        ax.axvline(cur['high'], color='red', linestyle='-', linewidth=1, alpha=0.6,
                   label=f'Current high={cur["high"]:.2f}')

        # ── 低分区域放大窗（0-0.02 百分比柱状图） ──
        if len(zoom_scores) > 0:
            ax_inset = ax.inset_axes([0.55, 0.18, 0.42, 0.32])

            bins_zoom = min(20, max(5, int(len(zoom_scores) / 50)))
            counts, edges = np.histogram(zoom_scores, bins=bins_zoom, range=(0, zoom_xmax))
            pcts = counts / len(scores) * 100  # 百分比
            centers_bins = (edges[:-1] + edges[1:]) / 2
            widths = np.diff(edges)

            ax_inset.bar(centers_bins, pcts, width=widths * 0.9,
                         color='gray', alpha=0.6, edgecolor='white')

            n_zoom = len(zoom_scores)
            ax_inset.text(0.95, 0.95, f'n={n_zoom}\n({n_zoom/len(scores)*100:.1f}%)',
                          transform=ax_inset.transAxes, va='top', ha='right',
                          fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_inset.set_xlim(0, zoom_xmax)
            ax_inset.set_xlabel('Risk Score', fontsize=7)
            ax_inset.set_ylabel('% of total', fontsize=7)
            ax_inset.tick_params(labelsize=6)
            ax_inset.grid(axis='y', alpha=0.3)
            ax_inset.set_xlabel('Risk Score (zoomed)', fontsize=7)
            ax_inset.set_ylabel('Density', fontsize=7)
            ax_inset.tick_params(labelsize=6)
            ax_inset.grid(axis='y', alpha=0.3)

        counts = res['counts']
        n = len(scores)
        info_text = (f'{name} (n={n})\n'
                     f'Low Risk:  {counts[0]} ({counts[0]/n*100:.0f}%)\n'
                     f'Mid Risk:  {counts[1]} ({counts[1]/n*100:.0f}%)\n'
                     f'High Risk: {counts[2]} ({counts[2]/n*100:.0f}%)')
        ax.text(0.97, 0.97, info_text, transform=ax.transAxes, va='top', ha='right',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        ax.set_xlabel('Risk Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name} — K-means (K=3) Thresholds', fontsize=12, fontweight='bold')
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'weight_sensitivity')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kmeans_thresholds.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {out_path}')


if __name__ == '__main__':
    run_kmeans_threshold()
