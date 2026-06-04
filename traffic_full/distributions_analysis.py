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

OUT_DIR = 'E:/0little/traffic_full/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}

METRICS = {
    'TTC': {'label': 'TTC (s)', 'bins': 200, 'range': (0, 20), 'color': '#c0392b'},
    'mTTC': {'label': 'mTTC (s)', 'bins': 200, 'range': (0, 20), 'color': '#d35400'},
    'PET': {'label': 'PET (s)', 'bins': 200, 'range': (0, 20), 'color': '#e74c3c'},
    'OL_PET': {'label': 'OL_PET (s)', 'bins': 200, 'range': (0, 20), 'color': '#8e44ad'},
    'Time_Headway': {'label': 'THW (s)', 'bins': 100, 'range': (0, 10), 'color': '#2980b9'},
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
    """图1: 五指标直方图对比 (2×3 布局)"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (col, cfg) in enumerate(METRICS.items()):
        ax = axes[idx]
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals_plot = vals[(vals > 0) & (vals < cfg['range'][1])].dropna()
        ax.hist(vals_plot, bins=cfg['bins'], color=cfg['color'],
                alpha=0.65, edgecolor='white', density=False, linewidth=0.3)

        # 拟合分布: 尝试 lognormal + gamma，取 SSE 最小的
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
            # 缩放到直方图尺度
            scale = len(vals_plot) * (cfg['range'][1] / cfg['bins'])
            ax.plot(x_fit, y_fit * scale, color='#2c3e50', linewidth=2.5,
                    alpha=0.8, label=f'拟合:{dist_name}')
            # 在右上角标注分布类型
            ax.text(0.97, 0.5, f'最佳拟合: {dist_name}', transform=ax.transAxes,
                    fontsize=9, ha='right', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            # 尝试第二种（如果存在）作为虚线
            if len(fit_candidates) > 1:
                _, dist2, p2, dn2 = fit_candidates[1]
                y2 = dist2.pdf(x_fit, *p2) * scale
                ax.plot(x_fit, y2, color='gray', linewidth=1.5, linestyle='--',
                        alpha=0.6, label=f'备选:{dn2}')

        # 2s/5s 阈值线
        ax.axvline(x=2, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.6, label='危险(2s)')
        ax.axvline(x=5, color='#f39c12', linewidth=1.5, linestyle='--', alpha=0.6, label='谨慎(5s)')

        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'均值={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')

        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('帧数', fontsize=11)
        ax.set_title(f'{col} 分布', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0, cfg['range'][1])

    # 第6个子图：五指标 KDE 密度对比（改用实线）
    ax = axes[5]
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
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('五指标密度对比 (KDE)', fontsize=13, fontweight='bold')
    ax.axvline(x=2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(x=5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    sample_frames = int(df.groupby(['ID', 'Source']).size().mode().iloc[0])
    n_veh = df.groupby(['ID', 'Source']).ngroups
    fig.suptitle(f'安全指标分布对比 ({n_veh} 辆 × {sample_frames} 帧)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_histograms.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ dist_histograms.png')


def plot_boxplot(df):
    """图2: 箱线图对比"""
    fig, ax = plt.subplots(figsize=(12, 6))
    data_list = []
    labels = []
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
    ax.axhline(y=2, color='#e74c3c', linewidth=1, linestyle='--', alpha=0.6, label='危险(2s)')
    ax.axhline(y=5, color='#f39c12', linewidth=1, linestyle='--', alpha=0.6, label='谨慎(5s)')
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('安全指标箱线图对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_boxplot.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ dist_boxplot.png')


def plot_cat_distribution(df):
    """图3: OL_PET_cat 分布饼图 + 柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 每辆车聚合一行（取第一个 OL_PET_cat 值）
    cats = df.groupby(['ID', 'Source'])['OL_PET_cat'].first().value_counts()
    colors_cat = {'dangerous': '#e74c3c', 'cautious': '#f39c12', 'safe': '#27ae60',
                  'no_follower': '#95a5a6'}

    # 饼图
    ax = axes[0]
    cat_counts = df['OL_PET_cat'].value_counts()
    c = [colors_cat.get(x, '#95a5a6') for x in cat_counts.index]
    wedges, texts, autotexts = ax.pie(
        cat_counts.values, labels=cat_counts.index, colors=c,
        autopct='%1.1f%%', startangle=90)
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title('OL_PET_cat 分布 (按帧)', fontsize=14, fontweight='bold')

    # 每辆车聚合柱状图
    ax = axes[1]
    veh_cats = df.groupby(['ID', 'Source'])['OL_PET_cat'].first().value_counts()
    c2 = [colors_cat.get(x, '#95a5a6') for x in veh_cats.index]
    bars = ax.bar(veh_cats.index, veh_cats.values, color=c2, width=0.5)
    for bar, v in zip(bars, veh_cats.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('类别', fontsize=11)
    ax.set_ylabel('车辆数', fontsize=11)
    ax.set_title('OL_PET_cat 分布 (按车)', fontsize=14, fontweight='bold')

    fig.suptitle(f'OL_PET_cat 标签分布 (n={len(cats)} 辆)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_ol_pet_cat.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ dist_ol_pet_cat.png')


def plot_ettc_distributions(df):
    """图5: ETTC/F_ETTC/B_ETTC 三指标分布 + KDE 对比"""
    ettc_cols = {
        'ETTC':    {'label': 'ETTC (s)',   'color': '#2980b9'},
        'F_ETTC':  {'label': 'F_ETTC (s)', 'color': '#27ae60'},
        'B_ETTC':  {'label': 'B_ETTC (s)', 'color': '#e67e22'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, cfg) in enumerate(ettc_cols.items()):
        ax = axes[idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        vals_plot = vals[(vals > 0) & (vals < 50)].dropna()
        ax.hist(vals_plot, bins=200, color=cfg['color'], alpha=0.65,
                edgecolor='white', density=False, linewidth=0.3)

        # 拟合分布
        for dist in [stats.lognorm, stats.gamma]:
            try:
                params = dist.fit(vals_plot)
                x_fit = np.linspace(0.01, 50, 500)
                y_fit = dist.pdf(x_fit, *params)
                scale = len(vals_plot) * (50 / 200)
                ax.plot(x_fit, y_fit * scale, color='#2c3e50', linewidth=2,
                        alpha=0.7, label=f'拟合:{dist.name}')
                break
            except Exception:
                continue

        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'均值={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('帧数', fontsize=11)
        ax.set_title(f'{col} 分布', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 50)

    # 第4个子图：三个指标的 KDE 密度对比
    ax = axes[3]
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
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('ETTC/F_ETTC/B_ETTC 密度对比 (KDE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('扩展碰撞时间分布对比', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_ettc.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ dist_ettc.png')


def plot_rsd_distributions(df):
    """图6: RSD/F_ERSD/B_ERSD 三指标分布 + KDE 对比"""
    rsd_cols = {
        'RSD':    {'label': 'RSD (m)',   'color': '#c0392b'},
        'F_ERSD': {'label': 'F_ERSD (m)', 'color': '#2980b9'},
        'B_ERSD': {'label': 'B_ERSD (m)', 'color': '#e67e22'},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (col, cfg) in enumerate(rsd_cols.items()):
        ax = axes[idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue
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
                        alpha=0.7, label=f'拟合:{dist.name}')
                break
            except Exception:
                continue

        mean_v = vals_plot.mean()
        ax.axvline(x=mean_v, color='#2c3e50', linewidth=1, linestyle=':', alpha=0.8)
        ax.text(mean_v, ax.get_ylim()[1] * 0.9, f'均值={mean_v:.2f}',
                fontsize=9, rotation=90, color='#2c3e50')
        ax.set_xlabel(cfg['label'], fontsize=11)
        ax.set_ylabel('帧数', fontsize=11)
        ax.set_title(f'{col} 分布', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

    # 第4个子图：KDE 密度对比
    ax = axes[3]
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
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('RSD/F_ERSD/B_ERSD 密度对比 (KDE)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    fig.suptitle('危险停车距离分布对比', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dist_rsd.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ dist_rsd.png')


def plot_risk_distribution_by_location(df):
    """图7: 变道 vs 跟驰 — 各场景风险等级对比（上下对称柱状图）"""
    import os
    print("\n计算各车辆风险等级（变道 + 跟驰）...")

    locations = ['location1', 'location2', 'location3', 'location4', 'location5']
    risk_levels = ['高风险', '中风险', '低风险']
    colors_lc = {'高风险': '#e74c3c', '中风险': '#f39c12', '低风险': '#27ae60'}
    colors_fl = {'高风险': '#c0392b', '中风险': '#d35400', '低风险': '#1e8449'}

    # ── 变道车辆 ──
    lc_counts = {loc: {lvl: 0 for lvl in risk_levels} for loc in locations}
    for (vid, source, loc), grp in df.groupby(['ID', 'Source', 'location']):
        try:
            sc = risk_score(grp, v0_kmh=80 if loc == 'location5' else 100)
            lbl, _ = risk_label_exp(sc, 'lane_change')
            lc_counts[loc][lbl] += 1
        except:
            continue

    # ── 跟驰车辆 ──
    fl_counts = {loc: {lvl: 0 for lvl in risk_levels} for loc in locations}
    for loc in locations:
        fp = os.path.normpath(os.path.join(LOCS[loc], 'traffic_following_change.csv'))
        if not os.path.exists(fp):
            continue
        df_f = pd.read_csv(fp, low_memory=False)
        for (vid, src), grp in df_f.groupby(['ID', 'Source']):
            try:
                sc = following_risk_score(grp, v0_kmh=80 if loc == 'location5' else 100)
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

    # ── 绘图：以 y=0 为中心，变道向上、跟驰向下 ──
    fig, ax = plt.subplots(figsize=(14, 7))
    x_pos = np.arange(len(locations))
    width = 0.25

    # 找最高百分比，用于对称轴缩放（让最高柱占 ~87% 空间）
    max_pct = max(
        max(lc_pct[loc][lvl] for loc in locations for lvl in risk_levels),
        max(fl_pct[loc][lvl] for loc in locations for lvl in risk_levels)
    )

    for idx, lvl in enumerate(risk_levels):
        lc_vals = [lc_pct[loc][lvl] for loc in locations]
        fl_vals = [fl_pct[loc][lvl] for loc in locations]

        # 上：变道
        bars = ax.bar(x_pos + idx * width, lc_vals, width,
                      label=f'变道-{lvl}', color=colors_lc[lvl], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, lc_vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=colors_lc[lvl])

        # 下：跟驰（取负值显示）
        bars = ax.bar(x_pos + idx * width, [-v for v in fl_vals], width,
                      label=f'跟驰-{lvl}', color=colors_fl[lvl], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, fl_vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 2,
                        f'{v:.0f}%', ha='center', va='top', fontsize=8, fontweight='bold',
                        color=colors_fl[lvl])

    # 对称百分比轴（两侧空间相等，最高比例柱占 ~87%）
    margin_pct = max_pct * 0.15
    ax.set_ylim(-(max_pct + margin_pct), max_pct + margin_pct)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel('场景位置', fontsize=12, fontweight='bold')
    ax.set_ylabel('跟驰 ↑ / 变道 ↓', fontsize=12, fontweight='bold')
    ax.set_title('变道 vs 跟驰 — 各场景风险等级占比对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    loc_labels = ['场景1', '场景2', '场景3', '场景4', '场景5']
    ax.set_xticklabels(loc_labels, fontsize=11)

    # 双图例：变道 + 跟驰
    handles_lc = [plt.Rectangle((0, 0), 1, 1, fc=colors_lc[lvl], alpha=0.85) for lvl in risk_levels]
    handles_fl = [plt.Rectangle((0, 0), 1, 1, fc=colors_fl[lvl], alpha=0.85) for lvl in risk_levels]
    leg1 = ax.legend(handles_lc, [f'变道-{lvl}' for lvl in risk_levels],
                     title='变道', loc='upper left', fontsize=9, title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles_fl, [f'跟驰-{lvl}' for lvl in risk_levels],
              title='跟驰', loc='lower left', fontsize=9, title_fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'risk_by_location.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ risk_by_location.png')

    # 打印统计信息
    print("\n各场景风险等级分布:")
    print("-" * 70)
    for loc in locations:
        lc_t = sum(lc_counts[loc].values())
        fl_t = sum(fl_counts[loc].values())
        print(f"{loc}:")
        print(f"  变道({lc_t}辆): 高={lc_counts[loc]['高风险']}({lc_counts[loc]['高风险']/max(lc_t,1)*100:.1f}%) "
              f"中={lc_counts[loc]['中风险']}({lc_counts[loc]['中风险']/max(lc_t,1)*100:.1f}%) "
              f"低={lc_counts[loc]['低风险']}({lc_counts[loc]['低风险']/max(lc_t,1)*100:.1f}%)")
        print(f"  跟驰({fl_t}辆): 高={fl_counts[loc]['高风险']}({fl_counts[loc]['高风险']/max(fl_t,1)*100:.1f}%) "
              f"中={fl_counts[loc]['中风险']}({fl_counts[loc]['中风险']/max(fl_t,1)*100:.1f}%) "
              f"低={fl_counts[loc]['低风险']}({fl_counts[loc]['低风险']/max(fl_t,1)*100:.1f}%)")


def plot_overall_risk_distribution(df):
    """图8: 左/右变道的风险等级汇总分布图（各一个饼图+柱状图）"""
    print("\n计算全量数据风险等级（按左/右变道分组）...")
    
    # 按车辆分组计算风险评分
    vehicle_risks = []
    grouped = df.groupby(['ID', 'Source'])
    
    for (vid, source), grp in grouped:
        try:
            risk_score = overall_risk(grp, v0_kmh=100)
            label, color = risk_label(risk_score)
            # 获取该车辆的变道方向（取第一个值）
            side = grp['side'].iloc[0] if 'side' in grp.columns else 'unknown'
            vehicle_risks.append({
                'ID': vid,
                'Source': source,
                'side': side,
                'risk_code': risk_score,
                'risk_label': label,
                'color': color
            })
        except Exception as e:
            continue
    
    if not vehicle_risks:
        print("  ⚠️ 无法计算风险等级，跳过此图")
        return
    
    risk_df = pd.DataFrame(vehicle_risks)
    colors_map = {'高风险': '#e74c3c', '中风险': '#f39c12', '低风险': '#27ae60'}
    labels_order = ['高风险', '中风险', '低风险']
    
    # 分别处理左变道和右变道
    for side_name, side_value in [('左变道', 'left'), ('右变道', 'right')]:
        side_data = risk_df[risk_df['side'] == side_value]
        
        if len(side_data) == 0:
            print(f"  ⚠️ {side_name}无数据，跳过")
            continue
        
        # 统计风险等级分布
        risk_counts = side_data['risk_label'].value_counts()
        counts_ordered = [risk_counts.get(l, 0) for l in labels_order]
        colors_ordered = [colors_map[l] for l in labels_order]
        total_vehicles = sum(counts_ordered)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：饼图
        ax = axes[0]
        wedges, texts, autotexts = ax.pie(
            counts_ordered, labels=labels_order, colors=colors_ordered,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
        
        ax.set_title(f'{side_name}风险等级分布 (n={total_vehicles} 辆)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 右图：柱状图
        ax = axes[1]
        bars = ax.bar(labels_order, counts_ordered, color=colors_ordered, 
                     width=0.5, alpha=0.85, edgecolor='white')
        
        for bar, count in zip(bars, counts_ordered):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('风险等级', fontsize=12, fontweight='bold')
        ax.set_ylabel('车辆数量', fontsize=12, fontweight='bold')
        ax.set_title(f'{side_name}风险等级车辆数量统计', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        fig.suptitle(f'{side_name}风险等级汇总分布', fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        filename = f'overall_risk_{side_value}.png'
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  ✅ {filename}')
        
        # 打印详细统计
        print(f"\n{side_name}风险等级汇总:")
        print("-" * 70)
        for level in labels_order:
            count = risk_counts.get(level, 0)
            pct = count / total_vehicles * 100 if total_vehicles > 0 else 0
            print(f"{level}: {count} 辆 ({pct:.1f}%)")
        print(f"总计: {total_vehicles} 辆")


def plot_risk_heatmap(df):
    """图8: 各场景风险位置热力图（仅高风险车辆点位）"""
    import os

    print("\n计算各车辆风险标签（仅高风险）...")

    # ── 变道：逐车辆算 risk_score → risk_label，仅标记高风险 ──
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

    # ── 跟驰：逐车辆算 following_risk_score → risk_label，仅标记高风险 ──
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

    n_vehicles = df_all.groupby(['ID', 'Source']).ngroups
    print(f"  总车辆: {n_vehicles} (变道 {df_lc.groupby(['ID','Source']).ngroups}, "
          f"跟驰 {df_fl.groupby(['ID','Source']).ngroups if len(df_fl) else 0})")
    print(f"  高风险帧数: {int(n_high_risk)}/{len(df_all)} ({n_high_risk/len(df_all)*100:.1f}%)"
          )

    loc_keys = list(LOCS.keys())
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), constrained_layout=True)
    last_im = None

    for idx, loc_name in enumerate(loc_keys):
        ax = axes[idx]
        loc_df = df_all[df_all['location'] == loc_name]
        if len(loc_df) == 0:
            ax.set_title(f'{loc_name} (无数据)', fontsize=13, fontweight='bold')
            continue

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

        last_im = ax.imshow(h, cmap='hot_r', interpolation='bilinear', alpha=0.85,
                            aspect='auto', extent=extent, origin='lower')

        # 车道线
        coeffs_path = os.path.join(LOCS[loc_name], 'lane_coeffs.csv')
        if os.path.exists(coeffs_path):
            coeffs_df = pd.read_csv(coeffs_path)
            first_src = coeffs_df['where'].iloc[0]
            src_coeffs = coeffs_df[coeffs_df['where'] == first_src]
            x_line = np.linspace(x_min - x_pad, x_max + x_pad, 400)
            for _, cr in src_coeffs.iterrows():
                coeffs = [cr['a5'], cr['a4'], cr['a3'], cr['a2'], cr['a1'], cr['a0']]
                y_line = np.polyval(coeffs, x_line)
                ax.plot(x_line, y_line, color='white', linewidth=2.5, alpha=0.9)
                ax.plot(x_line, y_line, color='#2c3e50', linewidth=0.8, linestyle='--', alpha=0.6)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_max + y_pad, y_min - y_pad)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'{loc_name}', fontsize=13, fontweight='bold')

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.6, label='高风险帧密度', pad=0.02)
    fig.suptitle('各场景风险位置分布热力图（变道 + 跟驰）', fontsize=18, fontweight='bold')
    fig.savefig(os.path.join(OUT_DIR, 'risk_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ risk_heatmap.png')


def main():
    print("加载全量数据...")
    df = load_all()
    n_veh = df.groupby(['ID', 'Source']).ngroups
    print(f"  共计 {len(df)} 行, {n_veh} 辆车")

    print("\n打印统计摘要...")
    print_stats(df)

    print("\n生成图表...")
    plot_distributions(df)
    plot_boxplot(df)
    plot_cat_distribution(df)
    plot_ettc_distributions(df)
    plot_rsd_distributions(df)
    plot_risk_distribution_by_location(df)
    plot_overall_risk_distribution(df)
    plot_risk_heatmap(df)

    print(f"\n✅ 全部完成! 图表已保存至 {OUT_DIR}")


if __name__ == '__main__':
    main()
