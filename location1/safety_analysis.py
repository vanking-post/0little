# "数据加载与处理
# 从多个CSV文件读取location1和location5的变道数据（左/右变道）
# 自动计算安全指标分类（TTC、PET、mTTC、THW等）
# 将车辆按风险等级分类（高风险/中风险/低风险）
# 安全指标分析
# TTC (Time To Collision): 碰撞时间
# PET (Post Encroachment Time): 后车侵入时间
# mTTC: 最小TTC
# THW (Time Headway): 车头时距
# 周边车辆距离分析
# 可视化输出（共6张图）
# 图1: 安全指标分布对比（箱线图）
# 图2: 风险等级分布饼图
# 图3: 风险演变趋势（TTC/PET随时间变化）
# 图4: 周边安全缺口分析（高风险vs低风险车辆距离对比）
# 图5: 速度与风险关系散点图
# 图6: 综合仪表盘（包含多个子图和数据表格）"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = r"E:\0little\location1"
LOC5_DIR = r"E:\0little\read\CQSkyEyedata5\location5t"
OUT_DIR = r"E:\0little\location1\safety_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    ('1-1',  '左变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-1_left_change.csv'),
    ('1-1',  '右变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-1_right_change.csv'),
    ('1-2',  '左变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-2_left_change.csv'),
    ('1-2',  '右变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-2_right_change.csv'),
    ('loc5', '左变道', 'loc5'): os.path.join(LOC5_DIR, 'traffic_left_change.csv'),
    ('loc5', '右变道', 'loc5'): os.path.join(LOC5_DIR, 'traffic_right_change.csv'),
}

SCENE_LABELS = [
    ('1-1', '左变道'), ('1-1', '右变道'),
    ('1-2', '左变道'), ('1-2', '右变道'),
    ('loc5', '左变道'), ('loc5', '右变道'),
]
SCENE_NAMES = ['1-1 左', '1-1 右', '1-2 左', '1-2 右', 'loc5 左', 'loc5 右']

risk_colors = {'高风险': '#F44336', '中风险': '#FF9800', '低风险': '#4CAF50'}
colors_lr = {'左变道': '#2196F3', '右变道': '#F44336'}

# ==================== 数据加载 ====================
def load_all_data():
    frames = []
    for (section, behavior, src_type), fpath in FILES.items():
        if not os.path.exists(fpath):
            print(f"[!] 跳过: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df['Source'] = section
        df['Behavior'] = behavior
        df['SourceType'] = src_type
        df['FrameIdx'] = df.groupby('ID').cumcount()
        df = _ensure_safety_categories(df)  # 逐文件编码，避免合并后跳行
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all['Velocity_ms'] = df_all['Velocity']
    df_all['Velocity_kmh'] = (df_all['Velocity'] * 3.6).round(1)
    return df_all


def _ensure_safety_categories(df):
    """确保 DataFrame 包含安全分类列，若没有则自动计算"""
    if 'TTC_cat' not in df.columns:
        df['TTC_cat'] = 'safe'
        df.loc[df['TTC'] == 0, 'TTC_cat'] = 'no_leader'
        df.loc[(df['TTC'] > 0) & (df['TTC'] < 2), 'TTC_cat'] = 'dangerous'
        df.loc[(df['TTC'] >= 2) & (df['TTC'] < 5), 'TTC_cat'] = 'cautious'
    if 'PET_cat' not in df.columns:
        df['PET_cat'] = 'safe'
        pet_inv = np.isinf(df['PET'].values) | pd.isna(df['PET'].values)
        df.loc[pet_inv, 'PET_cat'] = 'no_follower'
        df.loc[(df['PET'] > 0) & (df['PET'] < 2), 'PET_cat'] = 'dangerous'
        df.loc[(df['PET'] >= 2) & (df['PET'] < 5), 'PET_cat'] = 'cautious'
    if 'mTTC_cat' not in df.columns:
        df['mTTC_cat'] = 'safe'
        m_inv = np.isinf(df['mTTC'].values) | (df['mTTC'] == 0) | pd.isna(df['mTTC'].values)
        df.loc[m_inv, 'mTTC_cat'] = 'no_leader'
        df.loc[(df['mTTC'] > 0) & (df['mTTC'] < 2), 'mTTC_cat'] = 'dangerous'
        df.loc[(df['mTTC'] >= 2) & (df['mTTC'] < 5), 'mTTC_cat'] = 'cautious'
    if 'THW_cat' not in df.columns:
        df['THW_cat'] = 'safe'
        df.loc[df['Time_Headway'] == 0, 'THW_cat'] = 'no_leader'
        df.loc[(df['Time_Headway'] > 0) & (df['Time_Headway'] < 1), 'THW_cat'] = 'dangerous'
        df.loc[(df['Time_Headway'] >= 1) & (df['Time_Headway'] < 2), 'THW_cat'] = 'cautious'
    if 'has_front_vehicle' not in df.columns:
        df['has_front_vehicle'] = (df['TTC'] > 0) | (df['Following_dist'] > 0)
    if 'has_rear_vehicle' not in df.columns:
        df['has_rear_vehicle'] = ~(np.isinf(df['PET'].values) | pd.isna(df['PET'].values))
    # OL_PET: 原始车道后车 PET
    if 'OL_PET_cat' not in df.columns and 'OL_PET' in df.columns:
        df['OL_PET_cat'] = 'safe'
        olp_inv = np.isinf(df['OL_PET'].values) | pd.isna(df['OL_PET'].values)
        df.loc[olp_inv, 'OL_PET_cat'] = 'no_follower'
        df.loc[(df['OL_PET'] > 0) & (df['OL_PET'] < 2), 'OL_PET_cat'] = 'dangerous'
        df.loc[(df['OL_PET'] >= 2) & (df['OL_PET'] < 5), 'OL_PET_cat'] = 'cautious'
    return df


def extract_per_vehicle(df):
    last = df.groupby('ID').last().reset_index()
    mean = df.groupby('ID')[['Velocity_ms', 'Acceleration', 'TTC',
                               'Time_Headway', 'Following_dist', 'lat_Acc']].mean().reset_index()
    mean.columns = ['ID'] + [f'{c}_mean' for c in mean.columns if c != 'ID']
    veh = last.merge(mean, on='ID')

    for c in ['TTC', 'mTTC', 'PET', 'Time_Headway']:
        if c in veh.columns:
            veh[c] = veh[c].replace([float('inf'), -float('inf')], float('nan'))
    for dist_col in ['LB_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RF_Dist']:
        if dist_col in veh.columns:
            veh[dist_col] = veh[dist_col].replace(0, float('nan'))

    def classify_risk(row):
        ttc_cat = row.get('TTC_cat', 'safe')
        pet_cat = row.get('PET_cat', 'safe')
        if ttc_cat == 'dangerous' or pet_cat == 'dangerous':
            return '高风险'
        if ttc_cat == 'cautious' or pet_cat == 'cautious':
            return '中风险'
        return '低风险'

    veh['RiskLevel'] = veh.apply(classify_risk, axis=1)
    return veh


# ==================== 主分析 ====================
print("加载数据...")
df_all = load_all_data()
veh_all = extract_per_vehicle(df_all)
loc1_n = len(veh_all[veh_all['SourceType'] == 'loc1'])
loc5_n = len(veh_all[veh_all['SourceType'] == 'loc5'])
print(f"总车辆: {len(veh_all)} (location1: {loc1_n}, location5: {loc5_n}), 总行: {len(df_all)}")

# 分组统计
for st in ['loc1', 'loc5']:
    sub = veh_all[veh_all['SourceType'] == st]
    risk_counts = sub.groupby(['Source', 'Behavior', 'RiskLevel']).size().unstack(fill_value=0)
    label = 'location1' if st == 'loc1' else 'location5'
    print(f"\n=== 风险等级分布 ({label}) ===")
    print(risk_counts)

for st in ['loc1', 'loc5']:
    label = 'location1' if st == 'loc1' else 'location5'
    print(f"\n=== PET/TTC 统计 ({label}) ===")
    for beh in ['左变道', '右变道']:
        sub = veh_all[(veh_all['SourceType'] == st) & (veh_all['Behavior'] == beh)]
        pet = sub['PET'].dropna()
        ttc = sub['TTC'].dropna()
        ttc_risky = ttc[(ttc > 0) & (ttc < 2)]
        print(f"  {beh}: PET 有效={len(pet)} 均值={pet.mean():.1f}s 中位={pet.median():.1f}s <2s={ (pet<2).mean():.1%} | "
              f"TTC 有效={len(ttc)} TTC<2s={len(ttc_risky)} ({len(ttc_risky)/len(ttc):.1%})")

print("\n=== 安全指标分类分布 ===")
for st in ['loc1', 'loc5']:
    sub = veh_all[veh_all['SourceType'] == st]
    label = 'location1' if st == 'loc1' else 'location5'
    print(f"\n--- {label} ---")
    for cat_col in ['PET_cat', 'TTC_cat', 'mTTC_cat', 'THW_cat']:
        if cat_col in sub.columns:
            dist = sub[cat_col].value_counts()
            print(f"  {cat_col}: {dict(dist)}")
    for flag in ['has_front_vehicle', 'has_rear_vehicle']:
        if flag in sub.columns:
            vc = sub[flag].value_counts()
            print(f"  {flag}: True={vc.get(True,0)}, False={vc.get(False,0)}")

# ==================== 图1. 安全指标分布对比 (分loc1/loc5) ====================
print("\n生成图1: 安全指标分布对比...")
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
metrics = ['PET', 'TTC', 'mTTC', 'Time_Headway']
metric_labels = ['PET (s)', 'TTC (s)', 'mTTC (s)', 'THW (s)']

# 布局: 行0=loc1, 行1=loc5; 列0-3=PET/TTC/mTTC/THW
for row, (st, lbl) in enumerate([('loc1', 'Location1'), ('loc5', 'Location5')]):
    sub = veh_all[veh_all['SourceType'] == st]
    for col, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[row, col]
        data_pairs, labels_data = [], []
        for beh in ['左变道', '右变道']:
            vals = sub[sub['Behavior'] == beh][metric].dropna()
            cutoff = vals.quantile(0.99)
            vals = vals[vals <= cutoff]
            data_pairs.append(vals.values)
            labels_data.append(beh)
        bp = ax.boxplot(data_pairs, tick_labels=labels_data, patch_artist=True,
                        widths=0.4, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white', markersize=7))
        for patch, beh in zip(bp['boxes'], ['左变道', '右变道']):
            patch.set_facecolor(colors_lr[beh])
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{lbl} {metric}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (vals, beh) in enumerate(zip(data_pairs, labels_data)):
            med = np.median(vals)
            ax.annotate(f'{med:.1f}', xy=(i + 1, med), fontsize=8,
                        ha='center', va='bottom', fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors_lr[beh], alpha=0.9))

plt.suptitle('变道安全指标分布对比 (Location1 vs Location5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_metrics_comparison.png'), dpi=150)
plt.close()
print("  ✓ 图1 完成")

# ==================== 图2. 总体风险分布饼图 ====================
print("生成图2: 总体风险分布...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for idx, (st, lbl) in enumerate([('loc1', 'Location1'), ('loc5', 'Location5'), ('__all__', '全部')]):
    ax = axes[idx]
    if st == '__all__':
        sub = veh_all
    else:
        sub = veh_all[veh_all['SourceType'] == st]
    risk_summary = sub['RiskLevel'].value_counts()
    wedges, texts, autotexts = ax.pie(
        risk_summary.values, labels=risk_summary.index,
        colors=[risk_colors[l] for l in risk_summary.index],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}, pctdistance=0.55)
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(12)
    ax.set_title(f'{lbl} (N={len(sub)})', fontsize=13, fontweight='bold')
plt.suptitle('变道安全风险等级分布总览', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_risk_distribution.png'), dpi=150)
plt.close()
print("  ✓ 图2 完成")

# ==================== 图3. 风险演变 ====================
print("生成图3: 风险演变趋势...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for idx, beh in enumerate(['左变道', '右变道']):
    beh_df = df_all[df_all['Behavior'] == beh]
    ax = axes[0, idx]
    for level, color, ls in [('高风险', '#F44336', '-'), ('中风险', '#FF9800', '--'), ('低风险', '#4CAF50', ':')]:
        level_ids = veh_all[(veh_all['Behavior'] == beh) & (veh_all['RiskLevel'] == level)]['ID']
        level_data = beh_df[beh_df['ID'].isin(level_ids)]
        pivot = level_data.pivot_table(index='FrameIdx', columns='ID', values='TTC', aggfunc='first')
        mean_ttc = pivot.mean(axis=1)
        ax.plot(pivot.index, mean_ttc, color=color, linestyle=ls, linewidth=2.5,
                label=f'{level} (n={pivot.shape[1]})')
    ax.axhline(y=2, color='red', linestyle='-.', alpha=0.4, linewidth=1.5)
    ax.axhline(y=5, color='orange', linestyle='-.', alpha=0.4, linewidth=1.5)
    ax.set_xlabel('变道前帧序号 (0->49)', fontsize=11)
    ax.set_ylabel('TTC (s)', fontsize=11)
    ax.set_title(f'{beh} TTC 风险演变 (全部数据)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1, idx]
    pet_data, pet_labels = [], []
    for level in ['高风险', '中风险', '低风险']:
        level_ids = veh_all[(veh_all['Behavior'] == beh) & (veh_all['RiskLevel'] == level)]['ID']
        s = veh_all[(veh_all['ID'].isin(level_ids)) & (veh_all['Behavior'] == beh)]
        vals = s['PET'].dropna()
        if len(vals) > 0:
            pet_data.append(vals.values)
            pet_labels.append(f'{level}\n(n={len(vals)})')
    bp = ax.boxplot(pet_data, tick_labels=pet_labels, patch_artist=True, widths=0.4,
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=7))
    for patch, level in zip(bp['boxes'], pet_labels):
        color = risk_colors.get(level.split('\n')[0], '#999')
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=2, color='red', linestyle='-.', alpha=0.4, linewidth=1.5, label='PET=2s')
    ax.set_ylabel('PET (s)', fontsize=11)
    ax.set_title(f'{beh} PET 按风险等级分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('风险演变趋势分析', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_risk_evolution.png'), dpi=150)
plt.close()
print("  ✓ 图3 完成")

# ==================== 图4. 周边安全缺口分析 ====================
print("生成图4: 周边安全缺口分析...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
dist_cols = ['LB_Dist', 'LF_Dist', 'B_Dist', 'RF_Dist', 'RB_Dist']
dist_labels = ['左后方', '左前方', '正后方', '右前方', '右后方']
for idx, beh in enumerate(['左变道', '右变道']):
    ax = axes[idx]
    veh_beh = veh_all[veh_all['Behavior'] == beh]
    x = np.arange(len(dist_cols))
    width = 0.35
    means_high = [veh_beh[veh_beh['RiskLevel'] == '高风险'][c].dropna().mean()
                  if len(veh_beh[veh_beh['RiskLevel'] == '高风险'][c].dropna()) > 0 else 0 for c in dist_cols]
    means_low = [veh_beh[veh_beh['RiskLevel'] == '低风险'][c].dropna().mean()
                 if len(veh_beh[veh_beh['RiskLevel'] == '低风险'][c].dropna()) > 0 else 0 for c in dist_cols]
    bars1 = ax.bar(x - width / 2, means_high, width, label='高风险车辆', color='#F44336', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width / 2, means_low, width, label='低风险车辆', color='#4CAF50', alpha=0.8, edgecolor='white')
    for bar, val in zip(bars1, means_high):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='#F44336')
    for bar, val in zip(bars2, means_low):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='#4CAF50')
    ax.set_xticks(x)
    ax.set_xticklabels(dist_labels, fontsize=11)
    ax.set_ylabel('平均距离 (m)', fontsize=12)
    ax.set_title(f'{beh} 周边车辆距离 (决策时刻)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('变道安全缺口分析: 高风险 vs 低风险车辆', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_gap_analysis.png'), dpi=150)
plt.close()
print("  ✓ 图4 完成")

# ==================== 图5. 速度-风险关系 ====================
print("生成图5: 速度-风险关系...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for idx, beh in enumerate(['左变道', '右变道']):
    ax = axes[idx]
    veh_beh = veh_all[veh_all['Behavior'] == beh]
    for level, color, marker in [('高风险', '#F44336', 'o'), ('中风险', '#FF9800', 's'), ('低风险', '#4CAF50', '^')]:
        s = veh_beh[veh_beh['RiskLevel'] == level]
        ax.scatter(s['Velocity_ms'], s['TTC'], c=color, marker=marker, s=80, alpha=0.75,
                   edgecolors='black', linewidths=0.5, label=f'{level} (n={len(s)})', zorder=3)
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('速度 (m/s)', fontsize=12)
    ax.set_ylabel('TTC (s)', fontsize=12)
    ax.set_title(f'{beh} - 速度 vs TTC (决策时刻)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
plt.suptitle('变道速度与安全风险关系', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_speed_risk.png'), dpi=150)
plt.close()
print("  ✓ 图5 完成")

# ==================== 图6. 综合仪表盘 ====================
print("生成图6: 综合安全仪表盘...")
fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# 6a) 各场景 PET<2s 比例
ax = fig.add_subplot(gs[0, 0])
pet_ratios, ttc_ratios = [], []
for src, beh in SCENE_LABELS:
    s = veh_all[(veh_all['Source'] == src) & (veh_all['Behavior'] == beh)]
    pet = s['PET'].dropna()
    ttc = s['TTC'].dropna()
    ttc_risky = ttc[(ttc > 0) & (ttc < 2)]
    pet_ratios.append((pet < 2).mean() * 100 if len(pet) > 0 else 0)
    ttc_ratios.append(len(ttc_risky) / len(ttc) * 100 if len(ttc) > 0 else 0)
x = np.arange(len(SCENE_NAMES))
width = 0.35
bars1 = ax.bar(x - width / 2, pet_ratios, width, label='PET<2s', color='#F44336', alpha=0.85)
bars2 = ax.bar(x + width / 2, ttc_ratios, width, label='TTC<2s', color='#2196F3', alpha=0.85)
for bar, val in zip(bars1, pet_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=7, fontweight='bold')
for bar, val in zip(bars2, ttc_ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=7, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SCENE_NAMES, fontsize=8, rotation=15)
ax.set_ylabel('比例 (%)', fontsize=10)
ax.set_title('各场景危险指标比例', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.set_ylim(0, max(max(pet_ratios), max(ttc_ratios)) * 1.35 + 2)

# 6b) 加速度变化趋势
ax = fig.add_subplot(gs[0, 1])
for level, color, ls in [('高风险', '#F44336', '-'), ('低风险', '#4CAF50', '--')]:
    level_ids = veh_all[veh_all['RiskLevel'] == level]['ID']
    level_data = df_all[df_all['ID'].isin(level_ids)]
    pivot = level_data.pivot_table(index='FrameIdx', columns='ID', values='Acceleration', aggfunc='first')
    mean_acc, std_acc = pivot.mean(axis=1), pivot.std(axis=1)
    ax.plot(pivot.index, mean_acc, color=color, linestyle=ls, linewidth=2, label=f'{level}')
    ax.fill_between(pivot.index, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.1)
ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
ax.set_xlabel('变道前帧序号', fontsize=10)
ax.set_ylabel('加速度 (m/s^2)', fontsize=10)
ax.set_title('变道前加速度变化趋势', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 6c) TTC CDF
ax = fig.add_subplot(gs[0, 2])
for beh, color, ls in [('左变道', '#2196F3', '-'), ('右变道', '#F44336', '--')]:
    vals = veh_all[veh_all['Behavior'] == beh]['TTC'].dropna()
    vals = vals[vals > 0]
    sorted_vals = np.sort(vals)
    cum = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cum, color=color, linestyle=ls, linewidth=2.5, label=f'{beh} (n={len(vals)})')
ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(x=5, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('TTC (s)', fontsize=10)
ax.set_ylabel('累积概率', fontsize=10)
ax.set_title('TTC 累积分布 (CDF)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, min(sorted_vals.max(), 80))

# 6d) Location1 对比表
ax = fig.add_subplot(gs[1, :])
ax.axis('off')
table_data_loc1, table_data_loc5 = [], []
headers = ['指标', '左变道', '右变道', '差异', '解读']

for st, table_data in [('loc1', table_data_loc1), ('loc5', table_data_loc5)]:
    v = veh_all[veh_all['SourceType'] == st]
    for metric_name, col, unit, note in [
        ('PET (s)', 'PET', 's', 'lo'),
        ('TTC (s)', 'TTC', 's', 'lo'),
        ('mTTC (s)', 'mTTC', 's', 'lo'),
        ('THW (s)', 'Time_Headway', 's', 'lo'),
        ('跟驰距离 (m)', 'Following_dist', 'm', 'lo'),
        ('速度 (m/s)', 'Velocity_ms', 'm/s', '-'),
        ('高风险占比', None, '%', '-'),
        ('PET<2s 占比', None, '%', '-'),
    ]:
        if col == 'Velocity_ms':
            lv = v[v['Behavior'] == '左变道'][col].mean()
            rv = v[v['Behavior'] == '右变道'][col].mean()
        elif metric_name == '高风险占比':
            lv = (v[v['Behavior'] == '左变道']['RiskLevel'] == '高风险').mean() * 100
            rv = (v[v['Behavior'] == '右变道']['RiskLevel'] == '高风险').mean() * 100
        elif metric_name == 'PET<2s 占比':
            lp = v[v['Behavior'] == '左变道']['PET'].dropna()
            rp = v[v['Behavior'] == '右变道']['PET'].dropna()
            lv = (lp < 2).mean() * 100 if len(lp) > 0 else 0
            rv = (rp < 2).mean() * 100 if len(rp) > 0 else 0
        elif col:
            lv = v[v['Behavior'] == '左变道'][col].dropna().mean()
            rv = v[v['Behavior'] == '右变道'][col].dropna().mean()
        diff = rv - lv
        if note == 'lo':
            interp = '!! 右更危险' if diff < -1 else ('!! 左更危险' if diff > 1 else '~ 相近')
        elif note == '-':
            interp = '—'
        else:
            interp = '—'
        table_data.append([metric_name, f'{lv:.2f} {unit}', f'{rv:.2f} {unit}', f'{diff:+.2f} {unit}', interp])

# 画 loc1 表
ax.text(0.02, 0.55, 'Location1', fontsize=12, fontweight='bold', transform=ax.transAxes)
t1 = ax.table(cellText=table_data_loc1, colLabels=headers, cellLoc='center',
              loc='upper left', bbox=[0.0, 0.02, 0.48, 0.52], colWidths=[0.22, 0.22, 0.22, 0.18, 0.16])
t1.auto_set_font_size(False)
t1.set_fontsize(9)
for i, row in enumerate(table_data_loc1):
    if '!!' in row[-1]:
        for j in range(5):
            t1[i + 1, j].set_facecolor('#FFF3E0')

# 画 loc5 表
ax.text(0.52, 0.55, 'Location5', fontsize=12, fontweight='bold', transform=ax.transAxes)
t2 = ax.table(cellText=table_data_loc5, colLabels=headers, cellLoc='center',
              loc='upper right', bbox=[0.50, 0.02, 0.48, 0.52], colWidths=[0.22, 0.22, 0.22, 0.18, 0.16])
t2.auto_set_font_size(False)
t2.set_fontsize(9)
for i, row in enumerate(table_data_loc5):
    if '!!' in row[-1]:
        for j in range(5):
            t2[i + 1, j].set_facecolor('#FFF3E0')

# 6e) Location1 vs Location5 PET 对比
ax = fig.add_subplot(gs[2, 0])
pet_loc = []
for st in ['loc1', 'loc5']:
    for beh in ['左变道', '右变道']:
        vals = veh_all[(veh_all['SourceType'] == st) & (veh_all['Behavior'] == beh)]['PET'].dropna()
        pet_loc.append(vals.values)
labels_loc = ['loc1\n左', 'loc1\n右', 'loc5\n左', 'loc5\n右']
box_colors = ['#90CAF9', '#EF9A9A', '#42A5F5', '#E53935']
bp = ax.boxplot(pet_loc, tick_labels=labels_loc, patch_artist=True, widths=0.5,
                showmeans=True, meanprops=dict(marker='D', markerfacecolor='white', markersize=6))
for patch, c in zip(bp['boxes'], box_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='PET=2s')
ax.set_ylabel('PET (s)', fontsize=11)
ax.set_title('PET 分布: Location1 vs Location5', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 6f) 风险等级占比对比
ax = fig.add_subplot(gs[2, 1])
st_labels = ['loc1 左', 'loc1 右', 'loc5 左', 'loc5 右']
pairs = [('loc1', '左变道'), ('loc1', '右变道'), ('loc5', '左变道'), ('loc5', '右变道')]
xs = np.arange(len(st_labels))
bottom = np.zeros(len(st_labels))
for level in ['高风险', '中风险', '低风险']:
    counts = []
    for st, beh in pairs:
        cnt = len(veh_all[(veh_all['SourceType'] == st) & (veh_all['Behavior'] == beh) & (veh_all['RiskLevel'] == level)])
        counts.append(cnt)
    ax.bar(xs, counts, 0.5, bottom=bottom, label=level, color=risk_colors[level], edgecolor='white')
    for i, (bar_x, val) in enumerate(zip(xs, counts)):
        if val > 0:
            ax.text(bar_x, bottom[i] + val / 2, str(val), ha='center', va='center', fontsize=9, fontweight='bold')
    bottom += np.array(counts)
ax.set_xticks(xs)
ax.set_xticklabels(st_labels, fontsize=10)
ax.set_ylabel('车辆数', fontsize=11)
ax.set_title('Location1 vs Location5 风险对比', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, max(bottom) * 1.15)

# 6g) 高风险占比汇总
ax = fig.add_subplot(gs[2, 2])
high_risk_pcts = []
for st, beh in pairs:
    s = veh_all[(veh_all['SourceType'] == st) & (veh_all['Behavior'] == beh)]
    pct = (s['RiskLevel'] == '高风险').mean() * 100
    high_risk_pcts.append(pct)
bar_colors = ['#2196F3', '#F44336', '#1565C0', '#B71C1C']
bars = ax.bar(st_labels, high_risk_pcts, color=bar_colors, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, high_risk_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('高风险占比 (%)', fontsize=11)
ax.set_title('各场景高风险占比', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(high_risk_pcts) * 1.25)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('变道安全综合分析仪表盘', fontsize=17, fontweight='bold')
plt.savefig(os.path.join(OUT_DIR, '06_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图6 完成")

# ==================== 统计检验 ====================
print("\n=== Mann-Whitney U 检验 (左变道 vs 右变道) ===")
from scipy.stats import mannwhitneyu
for st in ['loc1', 'loc5', 'all']:
    if st == 'all':
        sub = veh_all
        label = '全部数据'
    else:
        sub = veh_all[veh_all['SourceType'] == st]
        label = 'location1' if st == 'loc1' else 'location5'
    print(f"\n--- {label} ---")
    for metric, name in [('PET', 'PET'), ('TTC', 'TTC'), ('mTTC', 'mTTC'),
                          ('Time_Headway', 'THW'), ('Velocity_ms', '速度')]:
        left_vals = sub[sub['Behavior'] == '左变道'][metric].dropna()
        right_vals = sub[sub['Behavior'] == '右变道'][metric].dropna()
        if len(left_vals) > 0 and len(right_vals) > 0:
            stat, p = mannwhitneyu(left_vals, right_vals, alternative='two-sided')
            sig = '显著' if p < 0.05 else '不显著'
            print(f"  {name}: p={p:.4f} ({sig})")

# 跨路段对比
print("\n=== Location1 vs Location5 对比 ===")
for beh in ['左变道', '右变道']:
    print(f"\n--- {beh} ---")
    for metric, name in [('PET', 'PET'), ('TTC', 'TTC')]:
        loc1_vals = veh_all[(veh_all['SourceType'] == 'loc1') & (veh_all['Behavior'] == beh)][metric].dropna()
        loc5_vals = veh_all[(veh_all['SourceType'] == 'loc5') & (veh_all['Behavior'] == beh)][metric].dropna()
        if len(loc1_vals) > 0 and len(loc5_vals) > 0:
            stat, p = mannwhitneyu(loc1_vals, loc5_vals, alternative='two-sided')
            sig = '显著' if p < 0.05 else '不显著'
            print(f"  {name}: loc1均值={loc1_vals.mean():.1f}, loc5均值={loc5_vals.mean():.1f}, p={p:.4f} ({sig})")

print(f"\n✅ 全部分析图表已保存至: {OUT_DIR}")
print(f"   共 6 张图 + 统计检验结果")
