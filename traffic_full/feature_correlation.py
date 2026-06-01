"""
特征关联与重要性分析
读取全部 1,494 辆变道样本，计算安全指标间的相关系数并可视化
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_scoring import overall_risk, risk_label

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'E:/0little/traffic_full/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}

# 用于相关性分析的特征列（数值型安全指标 + 运动学）
FEATURES = [
    'TTC', 'mTTC', 'ETTC', 'F_ETTC', 'B_ETTC',
    'PET', 'OL_PET', 'Time_Headway',
    'RSD', 'F_ERSD', 'B_ERSD',
    'Following_dist', 'B_Dist', 'LF_Dist', 'RF_Dist', 'LB_Dist', 'RB_Dist',
    'Velocity', 'long_Vel', 'lat_Vel', 'long_Acc', 'lat_Acc', 'Lateral_Jerk',
    'Length', 'Width',
]


def load_all():
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


def compute_risk_label(df):
    """每辆车聚合后计算综合风险标签"""
    grp = df.groupby(['ID', 'Source'])
    risk_map = {}
    for (vid, src), g in grp:
        v0 = 80 if g['location'].iloc[0] == 'location5' else 100
        risk_map[(vid, src)] = overall_risk(g, v0_kmh=v0)
    return risk_map


# ==================== 主流程 ====================
def main():
    print("加载全量数据...")
    df = load_all()
    n_veh = df.groupby(['ID', 'Source']).ngroups
    print(f"  共计 {len(df)} 行, {n_veh} 辆车")

    # 每辆车取第一帧代表该车的特征（聚合）
    print("\n聚合为逐车特征...")
    first = df.groupby(['ID', 'Source']).first().reset_index()

    # 计算每辆车的风险标签
    risk_map = compute_risk_label(df)
    first['risk'] = first.apply(lambda r: risk_map.get((r['ID'], r['Source']), 2), axis=1)
    first['risk_label'] = first['risk'].map({0: '高风险', 1: '中风险', 2: '低风险'})

    # 选取可用特征
    avail = [c for c in FEATURES if c in first.columns]
    print(f"  参与分析的特征: {avail}")

    # ==================== 图1: 相关系数热力图 ====================
    print("\n生成相关系数热力图...")
    corr_df = first[avail].replace([np.inf, -np.inf], np.nan).dropna()
    corr = corr_df.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(avail)))
    ax.set_yticks(range(len(avail)))
    ax.set_xticklabels(avail, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(avail, fontsize=8)

    # 在格子中标注数值
    for i in range(len(avail)):
        for j in range(len(avail)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else '#2c3e50'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=5.5, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman ρ')
    ax.set_title('安全指标 Spearman 相关系数矩阵', fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ feature_correlation.png')

    # ==================== 图2: 特征与风险标签的相关性 ====================
    print("\n生成特征-风险相关性柱状图...")
    corr_with_risk = {}
    for c in avail:
        sub = first[[c, 'risk']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) > 10:
            corr_with_risk[c] = sub[c].corr(sub['risk'], method='spearman')

    sorted_corr = sorted(corr_with_risk.items(), key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [x[0] for x in sorted_corr]
    values = [x[1] for x in sorted_corr]
    colors = ['#e74c3c' if v > 0 else '#2980b9' for v in values]
    bars = ax.barh(range(len(labels)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(x=0, color='#2c3e50', linewidth=0.8)
    ax.set_xlabel('Spearman ρ 与风险等级', fontsize=12)
    ax.set_title('各特征与综合风险等级的相关性', fontsize=14, fontweight='bold')
    ax.text(0.98, 0.02, '正值 = 高风险方向', transform=ax.transAxes, fontsize=9,
            ha='right', color='#e74c3c', fontstyle='italic')
    ax.text(0.98, 0.06, '负值 = 低风险方向', transform=ax.transAxes, fontsize=9,
            ha='right', color='#2980b9', fontstyle='italic')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'feature_risk_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ feature_risk_correlation.png')

    # ==================== 打印关键发现 ====================
    print(f"\n{'='*60}")
    print("关键发现")
    print(f"{'='*60}")
    print(f"与风险等级正相关最强的特征（值越大风险越高）:")
    for name, val in sorted_corr[:5]:
        print(f"  {name:20s}: ρ = {val:+.3f}")
    print(f"\n与风险等级负相关最强的特征（值越小风险越高）:")
    for name, val in sorted_corr[-5:]:
        print(f"  {name:20s}: ρ = {val:+.3f}")

    # 高相关特征对
    print(f"\n高相关特征对 (|ρ| > 0.7):")
    high_pairs = []
    for i in range(len(avail)):
        for j in range(i+1, len(avail)):
            val = corr.values[i, j]
            if abs(val) > 0.7:
                high_pairs.append((avail[i], avail[j], val))
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, v in high_pairs[:10]:
        print(f"  {a:20s} ↔ {b:20s}: ρ = {v:+.3f}")


if __name__ == '__main__':
    main()
