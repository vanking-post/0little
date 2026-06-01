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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'E:/0little/traffic_full/analysis'
os.makedirs(OUT_DIR, exist_ok=True)

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3_part1': 'E:/0little/location3_part1', 'location4_part1': 'E:/0little/location4_part1',
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
    """图7: 5个场景的高中低风险车辆数量对比柱状图"""
    print("\n计算各车辆风险等级...")
    
    # 按车辆分组计算风险评分
    vehicle_risks = []
    grouped = df.groupby(['ID', 'Source', 'location'])
    
    for (vid, source, loc), grp in grouped:
        try:
            risk_score = overall_risk(grp, v0_kmh=100)
            label, color = risk_label(risk_score)
            vehicle_risks.append({
                'ID': vid,
                'Source': source,
                'location': loc,
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
    
    # 统计每个场景的风险等级分布
    locations = ['location1', 'location2', 'location3_part1', 'location4_part1', 'location5']
    risk_levels = ['高风险', '中风险', '低风险']
    risk_codes = [0, 1, 2]
    colors_map = {'高风险': '#e74c3c', '中风险': '#f39c12', '低风险': '#27ae60'}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(locations))
    width = 0.25
    
    for idx, (level, code) in enumerate(zip(risk_levels, risk_codes)):
        counts = []
        for loc in locations:
            count = len(risk_df[(risk_df['location'] == loc) & (risk_df['risk_code'] == code)])
            counts.append(count)
        
        bars = ax.bar(x_pos + idx * width, counts, width, 
                     label=level, color=colors_map[level], alpha=0.85, edgecolor='white')
        
        # 在柱状图上标注数值
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                       str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置标签
    loc_labels = ['场景1', '场景2', '场景3', '场景4', '场景5']
    ax.set_xlabel('场景位置', fontsize=12, fontweight='bold')
    ax.set_ylabel('车辆数量', fontsize=12, fontweight='bold')
    ax.set_title('5个场景的高中低风险车辆数量对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(loc_labels, fontsize=11)
    ax.legend(title='风险等级', fontsize=10, title_fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'risk_by_location.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ risk_by_location.png')
    
    # 打印统计信息
    print("\n各场景风险等级分布:")
    print("-" * 70)
    for loc in locations:
        loc_data = risk_df[risk_df['location'] == loc]
        total = len(loc_data)
        high = len(loc_data[loc_data['risk_code'] == 0])
        mid = len(loc_data[loc_data['risk_code'] == 1])
        low = len(loc_data[loc_data['risk_code'] == 2])
        print(f"{loc}: 总计={total}, 高风险={high}({high/total*100:.1f}%), "
              f"中风险={mid}({mid/total*100:.1f}%), 低风险={low}({low/total*100:.1f}%)")


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
    """图8: 各场景高风险位置热力图"""
    from safety_scoring import overall_risk

    print("\n计算各车辆风险...")
    risk_map = {}
    for (vid, source, loc), grp in df.groupby(['ID', 'Source', 'location']):
        v0 = 80 if loc == 'location5' else 100
        risk_map[(vid, source, loc)] = overall_risk(grp, v0_kmh=v0)

    df['risk_code'] = df.apply(lambda r: risk_map.get((r['ID'], r['Source'], r['location']), 2), axis=1)
    high = df[df['risk_code'] == 0]
    print(f"  高风险车辆: {high.groupby(['ID','Source']).ngroups} 辆")

    loc_keys = list(LOCS.keys())
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), constrained_layout=True)
    last_im = None

    for idx, loc_name in enumerate(loc_keys):
        ax = axes[idx]
        loc_high = high[high['location'] == loc_name]
        loc_all = df[df['location'] == loc_name]
        if len(loc_high) == 0:
            ax.set_title(f'{loc_name} (无高风险)', fontsize=13, fontweight='bold')
            continue

        x_all = loc_all['X'].values
        y_all = loc_all['Y'].values
        x_min, x_max = np.percentile(x_all, 1), np.percentile(x_all, 99)
        x_pad = (x_max - x_min) * 0.05
        y_min, y_max = np.percentile(y_all, 1), np.percentile(y_all, 99)
        y_pad = (y_max - y_min) * 0.05

        bins = [80, 40]
        h, x_edges, y_edges = np.histogram2d(
            loc_high['X'].values, loc_high['Y'].values,
            bins=bins, range=[[x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]]
        )
        h = h.T
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        last_im = ax.imshow(h, cmap='hot_r', interpolation='bilinear', alpha=0.85,
                            aspect='auto', extent=extent, origin='lower')

        # 车道线
        loc_dir = LOCS[loc_name]
        coeffs_path = os.path.join(loc_dir, 'lane_coeffs.csv')
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

        n_high = loc_high.groupby(['ID', 'Source']).ngroups
        n_total = loc_all.groupby(['ID', 'Source']).ngroups
        ax.text(0.98, 0.02, f'高风险 {n_high}/{n_total} ({n_high/n_total*100:.1f}%)',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.8))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_max + y_pad, y_min - y_pad)  # y轴倒序
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(f'{loc_name} 高风险热力图', fontsize=13, fontweight='bold')

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.6, label='高风险点密度', pad=0.02)
    fig.suptitle('各场景高风险位置分布热力图', fontsize=18, fontweight='bold')
    # constrained_layout 已启用，无需调用 tight_layout
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
