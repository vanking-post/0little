"""
绘制 10 个变道数据文件的 Velocity 分布直方图 + 曲线拟合
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.mixture import GaussianMixture

DATA_DIR = 'E:/0little/traffic_full'
OUT_DIR = os.path.join('E:/0little/data_statistics', 'plot_velocity_distribution_output')

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LOC_DIRS = ['location1', 'location2', 'location3', 'location4', 'location5']

# 构建 location 变道样本文件列表
loc_files = []  # (path, location_name)
for loc in LOC_DIRS:
    for side in ['left', 'right']:
        fp = os.path.join(DATA_DIR, '..', loc, f'traffic_{side}_change.csv')
        fp = os.path.normpath(fp)
        if os.path.exists(fp):
            loc_files.append((fp, loc))

# ===== Velocity 分布图 =====
print('===== Velocity 分布图 =====')
fig, axes = plt.subplots(5, 2, figsize=(20, 22))
axes = axes.flatten()

for idx, (fp, loc_name) in enumerate(loc_files):
    ax = axes[idx]
    side_label = 'left' if 'left' in os.path.basename(fp) else 'right'
    title_prefix = f'{loc_name}_{side_label}'

    df = pd.read_csv(fp)
    v = df['Velocity'].replace([np.inf, -np.inf], np.nan).dropna().values * 3.6  # km/h

    # 直方图
    n, bins, patches = ax.hist(v, bins=60, density=True, alpha=0.6,
                               color='#3498db', edgecolor='white', linewidth=0.3)

    # 拟合正态分布
    mu, sigma = stats.norm.fit(v)
    x = np.linspace(v.min(), v.max(), 200)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, 'r-', linewidth=2, alpha=0.8,
            label=f'Normal fit μ={mu:.1f} σ={sigma:.1f}')

    # 标注 V85
    v85 = np.percentile(v, 85)
    ax.axvline(v85, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(v85, ax.get_ylim()[1] * 0.9, f'V85={v85:.0f}km/h',
            fontsize=9, color='#e74c3c', ha='right')

    # 标注 V0=100km/h
    ax.axvline(100, color='#f39c12', linewidth=1, linestyle=':', alpha=0.5)
    ax.text(100, ax.get_ylim()[1] * 0.8, 'V0=100', fontsize=8, color='#f39c12', ha='left')

    ax.set_xlabel('Velocity (km/h)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{title_prefix}  (n={len(v)})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, 'velocity_distribution.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'[OK] {out}')

# 汇总表
print(f'\n{"File":<30s} {"n":>6s} {"Mean":>7s} {"SD":>7s} {"V85":>7s} {"V85>100":>8s}')
print('-' * 65)
for fp, loc_name in loc_files:
    side_label = 'left' if 'left' in os.path.basename(fp) else 'right'
    title_prefix = f'{loc_name}_{side_label}'
    df = pd.read_csv(fp)
    v = df['Velocity'].replace([np.inf, -np.inf], np.nan).dropna().values * 3.6
    mu, sigma = stats.norm.fit(v)
    v85 = np.percentile(v, 85)
    pct_over = (v > 100).mean() * 100
    print(f'{title_prefix:<30s} {len(v):>6d} {mu:>7.1f} {sigma:>7.1f} {v85:>7.1f} {pct_over:>7.1f}%')

# ===== 安全指标分布图：mTTC / F_ETTC / PET / OL_PET =====
print('\n===== 安全指标分布图 =====')

METRIC_CONFIG = {
    'mTTC':    {'column': 'mTTC',    'label': 'mTTC (s)',     'unit': 's'},
    'F_ETTC':  {'column': 'F_ETTC',  'label': 'F_ETTC (s)',   'unit': 's',   'xlim': (0, 200)},
    'PET':     {'column': 'PET',     'label': 'PET (s)',      'unit': 's'},
    'OL_PET':  {'column': 'OL_PET',  'label': 'OL_PET (s)',   'unit': 's'},
}

if not loc_files:
    print('[WARN] 未找到 traffic_*_change.csv，跳过安全指标分布图')
else:
    for metric_name, cfg in METRIC_CONFIG.items():
        col = cfg['column']
        fig, axes = plt.subplots(5, 2, figsize=(20, 22))
        axes = axes.flatten()

        for idx, (fp, loc_name) in enumerate(loc_files):
            ax = axes[idx]
            side_label = 'left' if 'left' in os.path.basename(fp) else 'right'
            title_prefix = f'{loc_name}_{side_label}'

            df = pd.read_csv(fp)

            # 检查列是否存在
            if col not in df.columns:
                ax.text(0.5, 0.5, f'Column "{col}" not found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{title_prefix}  (n=0)', fontsize=12, fontweight='bold')
                print(f'    [WARN] {title_prefix}: 缺少列 "{col}"，可用列: {list(df.columns)[:10]}...')
                continue

            # PET / OL_PET 是每车单值（在 50 帧中重复），去重
            if col in ('PET', 'OL_PET'):
                per_vehicle = df.groupby(['ID', 'Source'])[col].first()
                vals = per_vehicle.replace([np.inf, -np.inf], np.nan).dropna().values
            else:
                vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
            vals = vals[vals > 0]  # >0 表示存在有效的前车/目标车道前车

            if len(vals) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{title_prefix}  (n=0)', fontsize=12, fontweight='bold')
                continue

            # 直方图
            ax.hist(vals, bins=60, density=True, alpha=0.6,
                    color='#3498db', edgecolor='white', linewidth=0.3)

            # 15% / 50% / 85% 分位线
            p15 = np.percentile(vals, 15)
            p50 = np.percentile(vals, 50)
            p85 = np.percentile(vals, 85)
            y_top = ax.get_ylim()[1]

            for p_val, pct, color in [(p15, 15, '#e74c3c'), (p50, 50, '#2ecc71'), (p85, 85, '#e67e22')]:
                ax.axvline(p_val, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
                ax.text(p_val, y_top * (0.88 - 0.12 * (pct // 50)),
                        f'P{pct}={p_val:.2f}{cfg["unit"]}',
                        fontsize=8, color=color, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

            ax.set_xlabel(cfg['label'], fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{title_prefix}  (n={len(vals)})', fontsize=12, fontweight='bold')
            if 'xlim' in cfg:
                ax.set_xlim(*cfg['xlim'])
            ax.grid(axis='y', alpha=0.3)

        # 隐藏多余子图（如果 loc_files < 10 则不处理，当前正好 10 个）
        for j in range(len(loc_files), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f'{metric_name}_distribution.png')
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'[OK] {out}')

# ===== 跟驰车辆 Velocity 分布图（每场景分开保存） =====
print('\n===== 跟驰车辆 Velocity 分布图（逐场景保存） =====')

for idx, loc_name in enumerate(LOC_DIRS):
    fp = os.path.normpath(os.path.join(DATA_DIR, '..', loc_name, 'traffic_following_change.csv'))
    if not os.path.exists(fp):
        print(f'    [SKIP] {loc_name}: 文件不存在')
        continue

    df = pd.read_csv(fp, low_memory=False)
    v = df['Velocity'].replace([np.inf, -np.inf], np.nan).dropna().values * 3.6  # km/h

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(v, bins=60, density=True, alpha=0.6, color='#2ecc71', edgecolor='white', linewidth=0.3)

    # 双正态混合拟合（GMM）
    x = np.linspace(v.min(), v.max(), 200)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(v.reshape(-1, 1))
    comp_idx = np.argsort(gmm.means_.flatten())  # 按均值排序
    colors = ['#e74c3c', '#3498db']
    labels = []
    pdf_total = np.zeros_like(x)
    for i, ci in enumerate(comp_idx):
        w = gmm.weights_[ci]
        mu = gmm.means_[ci, 0]
        sigma = np.sqrt(gmm.covariances_[ci, 0, 0])
        comp_pdf = w * stats.norm.pdf(x, mu, sigma)
        ax.plot(x, comp_pdf, '--', color=colors[i], linewidth=1.5, alpha=0.7)
        labels.append(f'Comp{i+1} w={w:.2f} μ={mu:.1f} σ={sigma:.1f}')
        pdf_total += comp_pdf
    ax.plot(x, pdf_total, 'r-', linewidth=2, alpha=0.8, label='Mixture')
    ax.legend(labels + ['Mixture'], fontsize=7)

    v85 = np.percentile(v, 85)
    ax.axvline(v85, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(v85, ax.get_ylim()[1] * 0.9, f'V85={v85:.0f}km/h', fontsize=8, color='#e74c3c', ha='right')

    ax.axvline(100, color='#f39c12', linewidth=1, linestyle=':', alpha=0.5)
    ax.text(100, ax.get_ylim()[1] * 0.8, 'V0=100', fontsize=7, color='#f39c12', ha='left')

    ax.set_xlabel('Velocity (km/h)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{loc_name} Following vehicle velocity distribution (n={len(v)})', fontsize=13, fontweight='bold')

    # 各场景 X 轴范围
    xlim_map = {
        'location2': (40, 150),
        'location3': (40, 150),
        'location4': (40, 150),
        'location5': (15, 120),
    }
    if loc_name in xlim_map:
        ax.set_xlim(*xlim_map[loc_name])

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'velocity_following_distribution_{loc_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {out}')

# ===== 跟驰车辆 mTTC / B_mTTC 分布图 =====
print('\n===== 跟驰车辆 mTTC / B_mTTC 分布图 =====')
METRICS_FOLLOW = {
    'mTTC':   {'column': 'mTTC',   'label': 'mTTC (s)',  'unit': 's',  'xlim': (0, 50)},
    'B_mTTC': {'column': 'B_mTTC', 'label': 'B_mTTC (s)', 'unit': 's', 'xlim': (0, 50)},
}

for metric_name, cfg in METRICS_FOLLOW.items():
    col = cfg['column']
    fig, axes = plt.subplots(5, 1, figsize=(10, 22))
    fig.suptitle(f'Following vehicle {cfg["label"]} distribution', fontsize=16, fontweight='bold', y=1.01)

    for idx, loc_name in enumerate(LOC_DIRS):
        ax = axes[idx]
        fp = os.path.normpath(os.path.join(DATA_DIR, '..', loc_name, 'traffic_following_change.csv'))
        if not os.path.exists(fp):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(loc_name, fontsize=12, fontweight='bold')
            continue

        df = pd.read_csv(fp, low_memory=False)
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        vals = vals[vals > 0]

        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{loc_name}  (n=0)', fontsize=12, fontweight='bold')
            continue

        ax.hist(vals, bins=60, density=True, alpha=0.6,
                color='#3498db', edgecolor='white', linewidth=0.3)

        # 双正态混合拟合（GMM）
        x_fit = np.linspace(0, 50, 200)
        gmm = GaussianMixture(n_components=2, random_state=42).fit(vals.reshape(-1, 1))
        comp_idx = np.argsort(gmm.means_.flatten())
        colors = ['#e74c3c', '#3498db']
        labels = []
        pdf_total = np.zeros_like(x_fit)
        for i, ci in enumerate(comp_idx):
            w = gmm.weights_[ci]
            mu = gmm.means_[ci, 0]
            sigma = np.sqrt(gmm.covariances_[ci, 0, 0])
            comp_pdf = w * stats.norm.pdf(x_fit, mu, sigma)
            ax.plot(x_fit, comp_pdf, '--', color=colors[i], linewidth=1.5, alpha=0.7)
            labels.append(f'Comp{i+1} w={w:.2f} μ={mu:.2f} σ={sigma:.2f}')
            pdf_total += comp_pdf
        ax.plot(x_fit, pdf_total, 'r-', linewidth=2, alpha=0.8, label='Mixture')
        ax.legend(labels + ['Mixture'], fontsize=7)

        # 15% / 50% / 85% 分位线
        p15 = np.percentile(vals, 15)
        p50 = np.percentile(vals, 50)
        p85 = np.percentile(vals, 85)
        y_top = ax.get_ylim()[1]

        for p_val, pct, color in [(p15, 15, '#e74c3c'), (p50, 50, '#2ecc71'), (p85, 85, '#e67e22')]:
            ax.axvline(p_val, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
            ax.text(p_val, y_top * (0.88 - 0.12 * (pct // 50)),
                    f'P{pct}={p_val:.2f}{cfg["unit"]}',
                    fontsize=8, color=color, ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_xlabel(cfg['label'], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{loc_name}  (n={len(vals)})', fontsize=12, fontweight='bold')
        if 'xlim' in cfg:
            ax.set_xlim(*cfg['xlim'])
            ax.set_xticks(np.arange(0, cfg['xlim'][1] + 1, 5))
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'{metric_name}_following_distribution.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[OK] {out}')
