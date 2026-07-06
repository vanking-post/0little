"""
跟驰 vs 变道车辆 — 连续风险分 (exp) 分布对比
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_scoring_exp import risk_score, following_risk_score

# ==================== 全局参数 ====================
DATA_DIR = 'E:/0little/traffic_full'
OUT_DIR = os.path.join(DATA_DIR, 'analysis')
LOC_DIRS = ['location1', 'location2', 'location3', 'location4', 'location5']
LOC_V0 = {'location5': 80}  # location5 基准速度 80km/h，其余默认 100

# 场景标签与颜色
COLORS = {'lane_change': '#e74c3c', 'following': '#3498db'}
LABELS = {'lane_change': 'Lane Change', 'following': 'Following'}

# 风险等级阈值
THRESHOLDS = {
    'lane_change': {'mid': 0.40, 'high': 0.60},
    'following':   {'mid': 0.20, 'high': 0.35},
}

# 图表尺寸与标题
FIG_SIZE = (16, 12)
FIG_TITLE = '跟驰 vs 变道车辆 — 连续风险分分布对比'

# score=0 小窗位置
INSET_POS = [0.65, 0.5, 0.35, 0.30]
# ==================================================

os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_scores():
    """加载所有 location 的变道和跟驰车辆风险分"""
    lc_scores, fl_scores = [], []

    for loc in LOC_DIRS:
        # 变道
        for side in ['left', 'right']:
            fp = os.path.join(DATA_DIR, '..', loc, f'traffic_{side}_change.csv')
            fp = os.path.normpath(fp)
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp, low_memory=False)
            for vid in df['ID'].unique():
                grp = df[df['ID'] == vid]
                s = risk_score(grp, v0_kmh=LOC_V0.get(loc, 100))
                lc_scores.append({'loc': loc, 'vid': vid, 'score': s})

        # 跟驰
        fp = os.path.normpath(os.path.join(DATA_DIR, '..', loc, 'traffic_following_change.csv'))
        if os.path.exists(fp):
            df = pd.read_csv(fp, low_memory=False)
            for vid in df['ID'].unique():
                grp = df[df['ID'] == vid]
                s = following_risk_score(grp, v0_kmh=LOC_V0.get(loc, 100))
                fl_scores.append({'loc': loc, 'vid': vid, 'score': s})

    return pd.DataFrame(lc_scores), pd.DataFrame(fl_scores)


print('正在读取数据并计算风险分...')
df_lc, df_fl = load_all_scores()
print(f'变道车辆: {len(df_lc)} 辆')
print(f'跟驰车辆: {len(df_fl)} 辆')

# ── 图1：风险分分布（排除 score=0）+ 箱线图 | 左右排布 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Following vs Lane Change — Risk Score Distribution & Box Plot', fontsize=16, fontweight='bold')

# 左：分布直方图（含 score=0 占比小窗）
ax = axes[0]

ax2 = axes[0].inset_axes(INSET_POS)
for i, (key, scores) in enumerate([('lane_change', df_lc['score']), ('following', df_fl['score'])]):
    zero_pct = (scores == 0).mean() * 100
    ax2.bar(i, zero_pct, width=0.4, color=COLORS[key], alpha=0.7)
    ax2.text(i, zero_pct + 1, f'{zero_pct:.0f}%', ha='center', fontsize=9, fontweight='bold', color=COLORS[key])
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Lane Change', 'Following'], fontsize=8)
ax2.set_ylabel('score=0 ratio (%)', fontsize=8)
ax2.set_ylim(0, 35)
ax2.grid(axis='y', alpha=0.3)

bins = np.linspace(0, 1, 50)
for key, scores in [('lane_change', df_lc['score']), ('following', df_fl['score'])]:
    pos = scores[scores > 0]
    ax.hist(pos, bins=bins, density=True, alpha=0.5,
            color=COLORS[key], label=f'{LABELS[key]} (n={len(pos)})')
ax.set_xlabel('Risk Score', fontsize=11)
ax.set_ylabel('Density (>0)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Risk Score Distribution (excl. score=0)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 1)

# 右：箱线图（横向）
ax = axes[1]
bp_data = [df_lc['score'].values, df_fl['score'].values]
bp = ax.boxplot(bp_data, tick_labels=['Lane Change', 'Following'], patch_artist=True, vert=False,
                widths=0.4, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black'))
for patch, color in zip(bp['boxes'], [COLORS['lane_change'], COLORS['following']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_xlabel('Risk Score', fontsize=11)
ax.grid(axis='x', alpha=0.3)
ax.set_title('Risk Score Box Plot', fontsize=13, fontweight='bold')

for i, (name, scores) in enumerate([('Lane Change', df_lc['score']), ('Following', df_fl['score'])]):
    stats_text = (f'{name}\n'
                  f'Mean={scores.mean():.3f}  Median={np.median(scores):.3f}\n'
                  f'P10={np.percentile(scores,10):.3f}  P90={np.percentile(scores,90):.3f}\n'
                  f'Max={scores.max():.3f}  n={len(scores)}')
    ax.text(0.95, 0.95 - i * 0.25, stats_text, transform=ax.transAxes,
            fontsize=8, color=COLORS[['lane_change', 'following'][i]],
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'risk_score_distribution_box.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\n[OK] {out1}')

# ── 图2：CDF + 风险等级比例 | 左右排布 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Following vs Lane Change — CDF & Risk Level Ratio', fontsize=16, fontweight='bold')

# 左：CDF
ax = axes[0]
for key, scores in [('lane_change', df_lc['score']), ('following', df_fl['score'])]:
    sorted_s = np.sort(scores)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(sorted_s, cdf, color=COLORS[key], linewidth=2, label=LABELS[key])
    t = THRESHOLDS[key]
    for p, pct in [(t['mid'], 50), (t['high'], 85)]:
        ax.axvline(p, color=COLORS[key], linewidth=1, linestyle='--', alpha=0.4)
        ax.axhline(pct / 100, color=COLORS[key], linewidth=1, linestyle=':', alpha=0.4)
ax.set_xlabel('Risk Score', fontsize=11)
ax.set_ylabel('Cumulative Ratio', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')

# 右：风险等级比例堆叠图
ax = axes[1]
x_pos = np.arange(2)
width = 0.35

for i, (key, scores) in enumerate([('lane_change', df_lc['score']), ('following', df_fl['score'])]):
    t = THRESHOLDS[key]
    low = ((scores < t['mid']) | (scores == 0)).sum()
    mid = ((scores >= t['mid']) & (scores < t['high'])).sum()
    high = (scores >= t['high']).sum()
    total = low + mid + high
    ax.bar(x_pos[i], low / total * 100, width, label='Low Risk' if i == 0 else None,
           color='#27ae60', alpha=0.8)
    ax.bar(x_pos[i], mid / total * 100, width, bottom=low / total * 100,
           label='Medium Risk' if i == 0 else None, color='#f39c12', alpha=0.8)
    ax.bar(x_pos[i], high / total * 100, width,
           bottom=(low + mid) / total * 100,
           label='High Risk' if i == 0 else None, color='#e74c3c', alpha=0.8)
    y_offset = 0
    for pct, c in [(low / total * 100, '#27ae60'), (mid / total * 100, '#f39c12'), (high / total * 100, '#e74c3c')]:
        if pct > 5:
            ax.text(x_pos[i], y_offset + pct / 2, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        y_offset += pct

ax.set_xticks(x_pos)
ax.set_xticklabels(['Lane Change', 'Following'], fontsize=11)
ax.set_ylabel('Ratio (%)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
ax.set_title('Risk Level Ratio', fontsize=13, fontweight='bold')

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'risk_score_cdf_levels.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] {out2}')
