"""
安全指标分布分析 — 读取全部 1,494 辆样本的 TTC/mTTC/PET/OL_PET/THW 分布
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import sys

# 导入安全评分模块
sys.path.insert(0, 'E:/0little/traffic_full')
from safety_scoring import overall_risk, risk_label
from safety_scoring_exp import risk_score, following_risk_score, risk_label as risk_label_exp

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join('E:/0little/data_statistics', 'distributions_analysis_output')
os.makedirs(OUT_DIR, exist_ok=True)

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}

METRICS = {
    'TTC': {'label': 'TTC (s)', 'bins': 200, 'range': (0, 20), 'xlim': (0, 20), 'color': '#c0392b'},
    'mTTC': {'label': 'mTTC (s)', 'bins': 200, 'range': (0, 20), 'xlim': (0, 20), 'color': '#d35400'},
    'PET': {'label': 'PET (s)', 'bins': 200, 'range': (0, 10), 'xlim': (0, 10), 'color': '#e74c3c'},
    'OL_PET': {'label': 'OL_PET (s)', 'bins': 200, 'range': (0, 12.5), 'xlim': (0, 12.5), 'color': '#8e44ad'},
    'Time_Headway': {'label': 'THW (s)', 'bins': 100, 'range': (0, 8), 'xlim': (0, 8), 'color': '#2980b9'},
}


def load_all():
    """加载全部 10 个文件，返回合并的 DataFrame"""
    parts = []
    for loc_key in LOCS:
        for side in ['left', 'right']:
            fp = os.path.join(LOCS[loc_key], f'traffic_{side}_change.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            df['location'] = loc_key
            df['side'] = side
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def print_stats(df):
    """打印各指标统计摘要"""
    # 检测每车帧数
    sample_frames = int(df.groupby(['ID', 'Source']).size().mode().iloc[0])
    print('=' * 70)
    print(f'  样本总数: {df.groupby(["ID","Source"]).ngroups} 辆, '
          f'{len(df)} 行 (每车 {sample_frames} 帧)')
    print('=' * 70)
    for col in ['TTC', 'mTTC', 'PET', 'OL_PET', 'Time_Headway', 'ETTC', 'F_ETTC', 'B_ETTC', 'RSD', 'F_ERSD', 'B_ERSD']:
        if col not in df.columns:
            continue
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        valid = vals.dropna()
        nonzero = valid[valid > 0]
        print(f'\n{col}')
        print(f'  有效帧: {len(valid)}/{len(df)}')
        print(f'  非零帧: {len(nonzero)}/{len(valid)}')
        print(f'  均值:  {valid.mean():.2f}  中位数: {valid.median():.2f}')
        if len(nonzero) > 0:
            print(f'  非零均值: {nonzero.mean():.2f}  非零中位数: {nonzero.median():.2f}')
            print(f'  <2s 占比: {(nonzero < 2).mean() * 100:.1f}%')
        if 'OL_PET_cat' in df.columns:
            print(f'  OL_PET_cat 分布: {df["OL_PET_cat"].value_counts().to_dict()}')


def plot_distributions(df):
    """Fig 1: Individual metric histograms + KDE comparison"""
    cols_list = list(METRICS.items())

    # 5 individual metric histogram + fit
    for col_name, cfg in cols_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = df[col_name].replace([np.inf, -np.inf], np.nan)
        vals_plot = vals[(vals > 0) & (vals < cfg['range'][1])].dropna()
        ax.hist(vals_plot, bins=cfg['bins'], color=cfg['color'],
                alpha=0.65, edgecolor='white', density=False, linewidth=0.3)

        fit_candidates = []
        for dist in [stats.lognorm, stats.gamma, stats.expon]:
            try:
                params = dist.fit(vals_plot)
                sse = np.sum((np.histogram(vals_plot, bins=cfg['bins'], range=cfg['range'])[0] -
                              dist.pdf(np.linspace(0.01, cfg['range'][1], cfg['bins']), *params) *
                              len(vals_plot) * (cfg['range'][1] / cfg['bins'])) ** 2)
                fit_candidates.append((sse, dist, params, dist.name))
            except Exception:
                continue
        if fit_candidates:
            fit_candidates.sort(key=lambda x: x[0])
            _, best_dist, best_params, dist_name = fit_candidates[0]
            x_fit = np.linspace(0.01, cfg['range'][1], 500)
            y_fit = best_dist.pdf(x_fit, *best_params)
            scale = len(vals_plot) * (cfg['range'][1] / cfg['bins'])
            ax.plot(x_fit, y_fit * scale, color='#2c3e50', linewidth=2.5,
                    alpha=0.8, label=f'Fit: {dist_name}')
            ax.text(0.97, 0.5, f'Best fit: {dist_name}', transform=ax.transAxes,
                    fontsize=9, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            if len(fit_candidates) > 1:
                _, dist2, p2, dn2 = fit_candidates[1]
                y2 = dist2.pdf(x_fit, *p2) * scale
                ax.plot(x_fit, y2, color='gray', linewidth=1.5, linestyle='--',
                        alpha=0.6, label=f'Alt: {dn2}')

        ax.axvline(x=2, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.6, label='Danger (2s)')
        ax.axvline(x=5, color='#f39c12', linewidth=1.5, linestyle='--', alpha=0.6, label='Caution (5s)')
        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'Mean={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('Frames', fontsize=11)
        ax.set_title(f'{col_name} Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(cfg.get('xlim', (0, cfg['range'][1])))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'dist_{col_name}.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] dist_{col_name}.png')

    # KDE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    for idx, (col, cfg) in enumerate(METRICS.items()):
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals = vals[(vals > 0) & (vals < cfg['range'][1])].dropna()
        try:
            kde = stats.gaussian_kde(vals, bw_method='scott')
            x_kde = np.linspace(0.01, cfg['range'][1], 500)
            ax.plot(x_kde, kde(x_kde), color=cfg['color'], linewidth=2.0,
                    linestyle=line_styles[idx % len(line_styles)],
                    alpha=0.85, label=col)
        except Exception:
            pass
    ax.set_xlim(0, 15)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('KDE Comparison (5 Metrics)', fontsize=13, fontweight='bold')
    ax.axvline(x=2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_kde_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_kde_comparison.png')


def plot_boxplot(df):
    """Boxplot comparison of safety metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    data_list, labels = [], []
    for col in ['TTC', 'mTTC', 'PET', 'OL_PET', 'Time_Headway']:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals = vals[(vals > 0) & (vals < 20)]
        data_list.append(vals.dropna())
        labels.append(col)
    bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, widths=0.5,
                     medianprops={'color': 'black', 'linewidth': 2})
    colors = ['#c0392b', '#d35400', '#e74c3c', '#8e44ad', '#2980b9']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(y=2, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.6, label='Danger (2s)')
    ax.axhline(y=5, color='#f39c12', linewidth=1, linestyle='--', alpha=0.6, label='Caution (5s)')
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Safety Metrics Boxplot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_boxplot.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_boxplot.png')


def plot_cat_distribution(df):
    """OL_PET_cat distribution: pie (by frame) + bar (by vehicle)"""
    colors_cat = {'dangerous': '#e74c3c', 'cautious': '#f39c12', 'safe': '#27ae60',
                  'no_follower': '#95a5a6'}
    veh_cats = df.groupby(['ID', 'Source'])['OL_PET_cat'].first().value_counts()

    # Pie (by frame)
    fig, ax = plt.subplots(figsize=(8, 6))
    cat_counts = df['OL_PET_cat'].value_counts()
    c = [colors_cat.get(x, '#95a5a6') for x in cat_counts.index]
    wedges, texts, autotexts = ax.pie(
        cat_counts.values, labels=cat_counts.index, colors=c,
        autopct='%1.1f%%', startangle=90)
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title('OL_PET_cat Distribution (by Frame)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_ol_pet_pie.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_ol_pet_pie.png')

    # Bar (by vehicle)
    fig, ax = plt.subplots(figsize=(8, 6))
    c2 = [colors_cat.get(x, '#95a5a6') for x in veh_cats.index]
    bars = ax.bar(veh_cats.index, veh_cats.values, color=c2, width=0.5)
    for bar, v in zip(bars, veh_cats.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Category', fontsize=11)
    ax.set_ylabel('Vehicles', fontsize=11)
    ax.set_title(f'OL_PET_cat Distribution (by Vehicle, n={len(veh_cats)})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_ol_pet_bar.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_ol_pet_bar.png')


def plot_ettc_distributions(df):
    """ETTC/F_ETTC/B_ETTC individual histograms + KDE comparison"""
    ettc_cols = {
        'ETTC':    {'label': 'ETTC (s)',   'color': '#2980b9'},
        'F_ETTC':  {'label': 'F_ETTC (s)', 'color': '#27ae60'},
        'B_ETTC':  {'label': 'B_ETTC (s)', 'color': '#e67e22'},
    }

    for col, cfg in ettc_cols.items():
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals_plot = vals[(vals > 0) & (vals < 50)].dropna()
        ax.hist(vals_plot, bins=200, color=cfg['color'], alpha=0.65,
                edgecolor='white', density=False, linewidth=0.3)
        for dist in [stats.lognorm, stats.gamma]:
            try:
                params = dist.fit(vals_plot)
                x_fit = np.linspace(0.01, 50, 500)
                y_fit = dist.pdf(x_fit, *params)
                scale = len(vals_plot) * (50 / 200)
                ax.plot(x_fit, y_fit * scale, color='#2c3e50', linewidth=2,
                        alpha=0.7, label=f'Fit: {dist.name}')
                break
            except Exception:
                continue
        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'Mean={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('Frames', fontsize=11)
        ax.set_title(f'{col} Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 50)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'dist_{col}_hist.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] dist_{col}_hist.png')

    # KDE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, cfg in ettc_cols.items():
        if col not in df.columns:
            continue
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals = vals[(vals > 0) & (vals < 50)].dropna()
        if len(vals) < 5:
            continue
        try:
            kde = stats.gaussian_kde(vals, bw_method='scott')
            x_kde = np.linspace(0.01, 50, 500)
            ax.plot(x_kde, kde(x_kde), color=cfg['color'], linewidth=2,
                    alpha=0.85, label=cfg['label'])
        except Exception:
            continue
    ax.set_xlim(0, 50)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('ETTC / F_ETTC / B_ETTC KDE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_ettc_kde.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_ettc_kde.png')


def plot_rsd_distributions(df):
    """RSD/F_ERSD/B_ERSD individual histograms + KDE comparison"""
    rsd_cols = {
        'RSD':    {'label': 'RSD (m)',   'color': '#c0392b'},
        'F_ERSD': {'label': 'F_ERSD (m)', 'color': '#2980b9'},
        'B_ERSD': {'label': 'B_ERSD (m)', 'color': '#e67e22'},
    }

    for col, cfg in rsd_cols.items():
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals_plot = vals[(vals > 0) & (vals < 100)].dropna()
        ax.hist(vals_plot, bins=100, color=cfg['color'], alpha=0.65,
                edgecolor='white', density=False, linewidth=0.3)
        for dist in [stats.lognorm, stats.gamma]:
            try:
                params = dist.fit(vals_plot)
                x_fit = np.linspace(0.01, 100, 500)
                y_fit = dist.pdf(x_fit, *params)
                scale = len(vals_plot) * (100 / 100)
                ax.plot(x_fit, y_fit * scale, color='#2c3e50', linewidth=2,
                        alpha=0.7, label=f'Fit: {dist.name}')
                break
            except Exception:
                continue
        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'Mean={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('Frames', fontsize=11)
        ax.set_title(f'{col} Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'dist_{col}_hist.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] dist_{col}_hist.png')

    # KDE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, cfg in rsd_cols.items():
        if col not in df.columns:
            continue
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals = vals[(vals > 0) & (vals < 100)].dropna()
        if len(vals) < 5:
            continue
        try:
            kde = stats.gaussian_kde(vals, bw_method='scott')
            x_kde = np.linspace(0.01, 100, 500)
            ax.plot(x_kde, kde(x_kde), color=cfg['color'], linewidth=2,
                    alpha=0.85, label=cfg['label'])
        except Exception:
            continue
    ax.set_xlim(0, 100)
    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('RSD / F_ERSD / B_ERSD KDE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_rsd_kde.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] dist_rsd_kde.png')


def plot_risk_distribution_by_location(df):
    """图7: 变道 vs 跟驰 — 各场景风险等级对比（上下对称柱状图）"""
    import os
    print("\n计算各车辆风险等级（变道 + 跟驰）...")

    locations = ['location1', 'location2', 'location3', 'location4', 'location5']
    risk_levels = ['高风险', '中风险', '低风险']
    colors_lc = {'高风险': '#e74c3c', '中风险': '#f39c12', '低风险': '#27ae60'}
    colors_fl = {'高风险': '#c0392b', '中风险': '#d35400', '低风险': '#1e8449'}

    # ── 变道车辆 ──
    lc_scores = {loc: [] for loc in locations}
    lc_counts = {loc: {lvl: 0 for lvl in risk_levels} for loc in locations}
    for (vid, source, loc), grp in df.groupby(['ID', 'Source', 'location']):
        try:
            sc = risk_score(grp, v0_kmh=80 if loc == 'location5' else 100)
            lc_scores[loc].append(sc)
            lbl, _ = risk_label_exp(sc, 'lane_change')
            lc_counts[loc][lbl] += 1
        except:
            continue

    # ── 跟驰车辆 ──
    fl_scores = {loc: [] for loc in locations}
    fl_counts = {loc: {lvl: 0 for lvl in risk_levels} for loc in locations}
    for loc in locations:
        fp = os.path.normpath(os.path.join(LOCS[loc], 'traffic_following_change.csv'))
        if not os.path.exists(fp):
            continue
        df_f = pd.read_csv(fp, low_memory=False)
        for (vid, src), grp in df_f.groupby(['ID', 'Source']):
            try:
                sc = following_risk_score(grp, v0_kmh=80 if loc == 'location5' else 100)
                fl_scores[loc].append(sc)
                lbl, _ = risk_label_exp(sc, 'following')
                fl_counts[loc][lbl] += 1
            except:
                continue

    # ── 转为百分比（变道/跟驰样本量差异大，百分比可对比风险分布模式）──
    lc_pct = {loc: {} for loc in locations}
    fl_pct = {loc: {} for loc in locations}
    for loc in locations:
        lc_tot = sum(lc_counts[loc].values()) or 1
        fl_tot = sum(fl_counts[loc].values()) or 1
        for lvl in risk_levels:
            lc_pct[loc][lvl] = lc_counts[loc][lvl] / lc_tot * 100
            fl_pct[loc][lvl] = fl_counts[loc][lvl] / fl_tot * 100

    # ── 山脊密度图 (Ridgeline): 跟驰在上、变道在下 ──
    fig, ax = plt.subplots(figsize=(14, 9))

    x_grid = np.linspace(0, 1, 500)
    band_height = 1.2
    gap = 0.3
    n_locs = len(locations)
    total_h = n_locs * (band_height + gap)

    lc_color = '#e74c3c'
    fl_color = '#3498db'
    half_bh = band_height * 0.85 * 0.8

    for idx, loc in enumerate(locations):
        y_base = idx * (band_height + gap)

        # ── 跟驰 KDE (基线上方) ──
        fl_data = np.array(fl_scores[loc])
        if len(fl_data) > 3:
            fl_kde = stats.gaussian_kde(fl_data, bw_method='scott')
            fl_d = fl_kde(x_grid)
            fl_d_s = fl_d / fl_d.max() * half_bh
            ax.fill_between(x_grid, y_base, y_base + fl_d_s,
                           color=fl_color, alpha=0.65, linewidth=0)
            ax.plot(x_grid, y_base + fl_d_s, color=fl_color, linewidth=1.8, alpha=0.9)

        # ── 变道 KDE (基线下方) ──
        lc_data = np.array(lc_scores[loc])
        if len(lc_data) > 3:
            lc_kde = stats.gaussian_kde(lc_data, bw_method='scott')
            lc_d = lc_kde(x_grid)
            lc_d_s = lc_d / lc_d.max() * half_bh
            ax.fill_between(x_grid, y_base - lc_d_s, y_base,
                           color=lc_color, alpha=0.65, linewidth=0)
            ax.plot(x_grid, y_base - lc_d_s, color=lc_color, linewidth=1.8, alpha=0.9)

        # 场景标签
        ax.text(-0.04, y_base, f'Loc{idx+1}',
               ha='right', va='center', fontsize=12, fontweight='bold',
               transform=ax.transData)

        # 基线
        ax.axhline(y=y_base, color='gray', linewidth=0.5, alpha=0.3)

    # 阈值线
    x_min, x_max = -0.08, 1.08
    y_top = total_h + gap * 0.3
    y_bot = -gap * 0.3

    # 阈值线（已去除，保持图面简洁）

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-half_bh - gap * 0.8, total_h + gap * 0.5)

    # 双 x 轴：底部=变道，顶部=跟驰
    ax.set_xlabel('risk score', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax_top = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    ax_top.tick_params(axis='x', colors=fl_color, labelsize=10)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例（扩大 2 倍）
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=fl_color, alpha=0.6, label='Following'),
                       Patch(facecolor=lc_color, alpha=0.6, label='Lane Change')]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=22, framealpha=0.9,
              handlelength=2.0, handleheight=1.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'risk_by_location.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] risk_by_location.png')

    # Print summary
    print("\nRisk distribution by location:")
    print("-" * 70)
    for loc in locations:
        lc_t = sum(lc_counts[loc].values())
        fl_t = sum(fl_counts[loc].values())
        print(f"{loc}:")
        print(f"  Lane change({lc_t}): High={lc_counts[loc]['高风险']}({lc_counts[loc]['高风险']/max(lc_t,1)*100:.1f}%) "
              f"Mid={lc_counts[loc]['中风险']}({lc_counts[loc]['中风险']/max(lc_t,1)*100:.1f}%) "
              f"Low={lc_counts[loc]['低风险']}({lc_counts[loc]['低风险']/max(lc_t,1)*100:.1f}%)")
        print(f"  Following({fl_t}): High={fl_counts[loc]['高风险']}({fl_counts[loc]['高风险']/max(fl_t,1)*100:.1f}%) "
              f"Mid={fl_counts[loc]['中风险']}({fl_counts[loc]['中风险']/max(fl_t,1)*100:.1f}%) "
              f"Low={fl_counts[loc]['低风险']}({fl_counts[loc]['低风险']/max(fl_t,1)*100:.1f}%)")


def plot_overall_risk_distribution(df):
    """Left/right lane change risk distribution (pie + bar per side)"""
    print("\nComputing risk levels (by left/right lane change)...")

    vehicle_risks = []
    grouped = df.groupby(['ID', 'Source'])

    for (vid, source), grp in grouped:
        try:
            rs = overall_risk(grp, v0_kmh=100)
            label, color = risk_label(rs)
            side = grp['side'].iloc[0] if 'side' in grp.columns else 'unknown'
            vehicle_risks.append({
                'ID': vid, 'Source': source, 'side': side,
                'risk_code': rs, 'risk_label': label, 'color': color
            })
        except Exception:
            continue

    if not vehicle_risks:
        print("  [SKIP] No data")
        return

    risk_df = pd.DataFrame(vehicle_risks)
    colors_map = {'高风险': '#e74c3c', '中风险': '#f39c12', '低风险': '#27ae60'}
    labels_order = ['高风险', '中风险', '低风险']
    labels_en = ['High Risk', 'Medium Risk', 'Low Risk']

    for side_name, side_value in [('Left', 'left'), ('Right', 'right')]:
        side_data = risk_df[risk_df['side'] == side_value]
        if len(side_data) == 0:
            print(f"  [SKIP] {side_name} lane change: no data")
            continue

        risk_counts = side_data['risk_label'].value_counts()
        counts_ordered = [risk_counts.get(l, 0) for l in labels_order]
        colors_ordered = [colors_map[l] for l in labels_order]
        total_v = sum(counts_ordered)

        # Pie
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            counts_ordered, labels=labels_en, colors=colors_ordered,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight('bold')
        ax.set_title(f'{side_name} Lane Change Risk Levels (n={total_v})',
                    fontsize=14, fontweight='bold', pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'overall_risk_{side_value}_pie.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] overall_risk_{side_value}_pie.png')

        # Bar
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels_en, counts_ordered, color=colors_ordered,
                     width=0.5, alpha=0.85, edgecolor='white')
        for bar, count in zip(bars, counts_ordered):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vehicles', fontsize=12, fontweight='bold')
        ax.set_title(f'{side_name} Lane Change Vehicle Count', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'overall_risk_{side_value}_bar.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] overall_risk_{side_value}_bar.png')

        # Stats
        print(f"\n{side_name} Lane Change Risk Summary:")
        print("-" * 50)
        for level, cnt in zip(labels_en, counts_ordered):
            print(f"  {level}: {cnt} ({cnt/max(total_v,1)*100:.1f}%)")
        print(f"  Total: {total_v}")


def plot_risk_heatmap(df):
    """Risk heatmap per location (lane change + following)"""
    import os
    print("\nComputing risk labels...")

    lc_risk_map = {}
    for (vid, source, loc), grp in df.groupby(['ID', 'Source', 'location']):
        try:
            sc = risk_score(grp, v0_kmh=80 if loc == 'location5' else 100)
            lbl, _ = risk_label_exp(sc, 'lane_change')
            lc_risk_map[(vid, source, loc)] = 1 if lbl == '高风险' else 0
        except:
            continue

    df_lc = df.copy()
    df_lc['is_high_risk'] = df_lc.apply(
        lambda r: lc_risk_map.get((r['ID'], r['Source'], r['location']), 0), axis=1)

    df_fl_parts = []
    for loc_name in LOCS:
        fp = os.path.normpath(os.path.join(LOCS[loc_name], 'traffic_following_change.csv'))
        if not os.path.exists(fp):
            continue
        df_f = pd.read_csv(fp, low_memory=False)
        df_f['location'] = loc_name
        df_f['is_high_risk'] = 0
        for (vid, src), grp in df_f.groupby(['ID', 'Source']):
            try:
                sc = following_risk_score(grp, v0_kmh=80 if loc_name == 'location5' else 100)
                lbl, _ = risk_label_exp(sc, 'following')
                if lbl == '高风险':
                    df_f.loc[(df_f['ID'] == vid) & (df_f['Source'] == src), 'is_high_risk'] = 1
            except:
                continue
        df_fl_parts.append(df_f)

    df_fl = pd.concat(df_fl_parts, ignore_index=True) if df_fl_parts else pd.DataFrame()
    df_all = pd.concat([df_lc, df_fl], ignore_index=True)
    n_high_risk = df_all['is_high_risk'].sum()

    # Load lane coefficients from xlsx (fallback for locations without CSV)
    import ast, re
    xlsx_coeffs = {}
    xlsx_path = os.path.join('E:/0little', 'lane_coeffs.xlsx')
    if os.path.exists(xlsx_path):
        xl = pd.read_excel(xlsx_path, sheet_name='Sheet1')
        for _, row in xl.iterrows():
            src = row['where']
            matches = re.findall(r'\[[^\]]+\]', str(row['lane_coeffs']))
            if matches:
                xlsx_coeffs[src] = np.array([ast.literal_eval(m) for m in matches])

    # Individual heatmaps per location
    for loc_name in LOCS:
        loc_df = df_all[df_all['location'] == loc_name]
        if len(loc_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        x_all = loc_df['X'].values
        y_all = loc_df['Y'].values
        weights = loc_df['is_high_risk'].values

        x_min, x_max = np.percentile(x_all, 1), np.percentile(x_all, 99)
        x_pad = (x_max - x_min) * 0.05
        y_min, y_max = np.percentile(y_all, 1), np.percentile(y_all, 99)
        y_pad = (y_max - y_min) * 0.05

        bins = [80, 40]
        h, x_edges, y_edges = np.histogram2d(
            x_all, y_all, weights=weights,
            bins=bins, range=[[x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]]
        )
        h = h.T
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        im = ax.imshow(h, cmap='hot_r', interpolation='bilinear', alpha=0.85,
                       aspect='auto', extent=extent, origin='lower')

        # Try CSV first, then xlsx fallback
        coeffs_list = None
        coeffs_csv = os.path.join(LOCS[loc_name], 'lane_coeffs.csv')
        if os.path.exists(coeffs_csv):
            coeffs_df = pd.read_csv(coeffs_csv)
            first_src = coeffs_df['where'].iloc[0]
            src_df = coeffs_df[coeffs_df['where'] == first_src]
            coeffs_list = src_df[['a5','a4','a3','a2','a1','a0']].values
        else:
            # Fallback to xlsx
            loc_num = loc_name.replace('location', '')
            src_keys = sorted([k for k in xlsx_coeffs if k.startswith(f'{loc_num}-')])
            if src_keys:
                # Average across sources
                all_arr = np.array([xlsx_coeffs[k] for k in src_keys])
                # Each source: (8, 6); average all
                coeffs_list = all_arr.reshape(-1, 6)  # flatten all curves

        if coeffs_list is not None:
            x_line = np.linspace(x_min - x_pad, x_max + x_pad, 400)
            for coeff in coeffs_list:
                y_line = np.polyval(coeff, x_line)
                ax.plot(x_line, y_line, color='white', linewidth=2.5, alpha=0.9)
                ax.plot(x_line, y_line, color='#2c3e50', linewidth=0.8, linestyle='--', alpha=0.6)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_max + y_pad, y_min - y_pad)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'{loc_name} Risk Heatmap', fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label='High Risk Density')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'risk_heatmap_{loc_name}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [OK] risk_heatmap_{loc_name}.png')


def print_metric_percentiles(base_dir='E:/0little'):
    """读取全部变道+跟驰数据，计算五个指标 (mTTC/THW/PET/F_ETTC/OL_PET) 的 P15/P50/P85"""
    locs = [f'location{i}' for i in range(1, 6)]
    metric_cols = ['mTTC', 'Time_Headway', 'PET', 'F_ETTC', 'OL_PET']
    all_vals = {col: [] for col in metric_cols}

    print("\n" + "=" * 60)
    print("  Metric Percentiles (P15 / P50 / P85)")
    print("=" * 60)

    for loc in locs:
        for fname in ['traffic_left_change.csv', 'traffic_right_change.csv', 'traffic_following_change.csv']:
            fp = os.path.normpath(os.path.join(base_dir, loc, fname))
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp, low_memory=False)
            for col in metric_cols:
                if col in df.columns:
                    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                    vals = vals[vals > 0]
                    all_vals[col].extend(vals.tolist())

    print('  {:<20s} {:>8s} {:>8s} {:>8s} {:>8s}'.format('Metric', 'P15', 'P50', 'P85', 'Count'))
    print('  ' + '-' * 52)
    for col in metric_cols:
        arr = np.array(all_vals[col])
        if len(arr) == 0:
            continue
        p15, p50, p85 = np.percentile(arr, [15, 50, 85])
        print(f'  {col:<20s} {p15:>8.3f} {p50:>8.3f} {p85:>8.3f} {len(arr):>8d}')
    print("=" * 60)


def main():
    print("Loading all data...")
    df = load_all()
    n_veh = df.groupby(['ID', 'Source']).ngroups
    print(f"  Total: {len(df)} rows, {n_veh} vehicles")

    print("\nPrinting statistics summary...")
    print_stats(df)

    print("\nGenerating charts...")
    plot_distributions(df)
    plot_boxplot(df)
    plot_cat_distribution(df)
    plot_ettc_distributions(df)
    plot_rsd_distributions(df)
    plot_risk_distribution_by_location(df)
    plot_overall_risk_distribution(df)
    plot_risk_heatmap(df)

    print(f"\n✅ All done! Charts saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
