"""
敏感性验证：不同赋权方法对风险评分及标签的影响

比较 6 种赋权方法（Expert / EWM / CRITIC / AHP / Multiplicative / MeanDev）
对变道和跟驰车辆风险评分和三级标签的差异。

输出到 E:\0little\traffic_full\weight_sensitivity/
"""
import numpy as np
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import spearmanr
from itertools import combinations

# ── 路径 ──
BASE_DIR = 'E:/0little'
OUT_DIR = os.path.join(BASE_DIR, 'traffic_full', 'weight_sensitivity')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, 'traffic_full'))

from safety_scoring_exp import METRICS, FOLLOWING_METRICS, _frame_contrib, \
    THRESH_LANE_CHANGE, THRESH_FOLLOWING
from weight_safety_exp import compute_ewm_weights, compute_critic_weights, \
    compute_ahp_weights, compute_combined_weights, \
    LC_AHP_MATRIX, FL_AHP_MATRIX, build_activation_matrix

# ── 全局设置 ──
LOC_V0 = {f'location{i}': 100 if i < 5 else 80 for i in range(1, 6)}
LOC_DIRS = {f'location{i}': os.path.join(BASE_DIR, f'location{i}') for i in range(1, 6)}
LABEL_ORDER = ['高风险', '中风险', '低风险']
LABEL_COLORS_JAP = ['#e74c3c', '#f39c12', '#27ae60']  # 红黄绿
METHOD_COLORS = ['#2c3e50', '#3498db', '#1abc9c', '#e67e22', '#9b59b6', '#e84393']
B = 1.3  # 速度修正参数


# ── 辅助函数 ──


def _k_speed_factor(grp, v0_kmh):
    """速度修正系数 K"""
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        return 1.0
    v0 = v0_kmh / 3.6
    return 1.0 if v85 <= v0 else (v85 / v0) ** B


def _get_csv_files(data_type='lc'):
    """获取所有 location 的 CSV 文件路径列表，附带 loc_key"""
    files = []
    if data_type == 'lc':
        for loc_key, base_dir in LOC_DIRS.items():
            for side in ['left', 'right']:
                fp = os.path.normpath(f'{base_dir}/traffic_{side}_change.csv')
                if os.path.exists(fp):
                    files.append((fp, loc_key))
    else:
        for loc_key, base_dir in LOC_DIRS.items():
            fp = os.path.normpath(f'{base_dir}/traffic_following_change.csv')
            if os.path.exists(fp):
                files.append((fp, loc_key))
    return files


def _get_weight_dict(files, metrics_config, scenario_key):
    """计算 6 种方法的权重字典"""
    paths = [fp for fp, _ in files]
    X, _ = build_activation_matrix(paths, metrics_config)

    w_expert = np.array([m['w'] for m in metrics_config])
    w_ewm, _, _ = compute_ewm_weights(X)
    w_critic, _, _, _ = compute_critic_weights(X)

    matrix = LC_AHP_MATRIX if scenario_key == 'lc' else FL_AHP_MATRIX
    w_ahp, _, _, _ = compute_ahp_weights(matrix)

    combo = compute_combined_weights({
        'EWM': w_ewm, 'CRITIC': w_critic, 'AHP': w_ahp,
    })
    w_multi = combo['multiplicative']
    w_mean = combo['min_deviation']

    return {
        'Expert': w_expert,
        'EWM': w_ewm,
        'CRITIC': w_critic,
        'AHP': w_ahp,
        'Multiplicative': w_multi,
        'MeanDev': w_mean,
    }


def compute_scores(files, metrics_config, weight_dict, thresholds):
    """用多组权重计算每辆车的风险分和标签"""
    score_cols = []
    for method_name in weight_dict:
        score_cols.append(f'score_{method_name}')

    rows = []
    for fp, loc_key in files:
        if not os.path.exists(fp):
            continue
        v0_kmh = LOC_V0[loc_key]
        df = pd.read_csv(fp)
        for (vid, src), grp in df.groupby(['ID', 'Source']):
            grp = grp.sort_values('Frame')
            k_speed = _k_speed_factor(grp, v0_kmh)

            contribs = []
            for m in metrics_config:
                if m['col'] not in grp.columns:
                    contribs.append(0.0)
                    continue
                vals = grp[m['col']].values.astype(float)
                valid = m['valid'](vals)
                if valid.any():
                    c = float(np.nanmax(_frame_contrib(vals[valid], m['k'])))
                else:
                    c = 0.0
                contribs.append(c)
            contribs = np.array(contribs)

            row = {'ID': vid, 'Source': src, 'loc_key': loc_key}
            for method_name, weights in weight_dict.items():
                score = float(np.dot(contribs, weights)) * k_speed
                row[f'score_{method_name}'] = score
                if score >= thresholds['high']:
                    row[f'label_{method_name}'] = '高风险'
                elif score >= thresholds['mid']:
                    row[f'label_{method_name}'] = '中风险'
                else:
                    row[f'label_{method_name}'] = '低风险'
            rows.append(row)

    result_df = pd.DataFrame(rows)
    return result_df, score_cols


# ── 分析函数 ──


def label_consistency(df, method_names, baseline='Expert'):
    """计算标签一致性百分比矩阵"""
    n = len(method_names)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            agree = (df[f'label_{m1}'] == df[f'label_{m2}']).mean() * 100
            matrix[i, j] = agree
    return pd.DataFrame(matrix, index=method_names, columns=method_names)


def label_consistency_vs_baseline(df, method_names, baseline='Expert'):
    """各方法 vs 基准方法的标签一致性"""
    results = {'method': [], 'overall_agree': []}
    for level in LABEL_ORDER:
        results[f'{level}_agree'] = []
    base_labels = df[f'label_{baseline}']
    for method in method_names:
        method_labels = df[f'label_{method}']
        results['method'].append(method)
        results['overall_agree'].append((base_labels == method_labels).mean() * 100)
        for level in LABEL_ORDER:
            mask = base_labels == level
            if mask.sum() > 0:
                agree = (method_labels[mask] == level).mean() * 100
            else:
                agree = np.nan
            key = f'{level}_agree'
            results[key].append(agree)
    return pd.DataFrame(results)


def label_transition_matrix(df, method_a, method_b):
    """两个方法之间的标签变迁矩阵"""
    a_labels = df[f'label_{method_a}']
    b_labels = df[f'label_{method_b}']
    matrix = pd.crosstab(a_labels, b_labels, normalize='index')
    # 确保行列顺序一致
    matrix = matrix.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    return matrix


def top10_overlap(df, method_names, baseline='Expert'):
    """Top-10% 高风险样本的 Jaccard 相似度矩阵"""
    n = len(method_names)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(method_names):
        scores1 = df[f'score_{m1}']
        thresh1 = np.percentile(scores1, 90)
        top1 = set(df[scores1 >= thresh1].index)
        for j, m2 in enumerate(method_names):
            scores2 = df[f'score_{m2}']
            thresh2 = np.percentile(scores2, 90)
            top2 = set(df[scores2 >= thresh2].index)
            intersection = len(top1 & top2)
            union = len(top1 | top2)
            matrix[i, j] = intersection / union if union > 0 else 0.0
    return pd.DataFrame(matrix, index=method_names, columns=method_names)


def spearman_corr(df, method_names):
    """Spearman 秩相关系数矩阵"""
    n = len(method_names)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            rho, _ = spearmanr(df[f'score_{m1}'], df[f'score_{m2}'])
            matrix[i, j] = rho
    return pd.DataFrame(matrix, index=method_names, columns=method_names)


# ── 可视化函数 ──


def plot_label_distribution(df, method_names, title, save_path):
    """各方法标签分布堆叠柱状图"""
    data = {}
    for m in method_names:
        counts = df[f'label_{m}'].value_counts()
        data[m] = [counts.get(l, 0) for l in LABEL_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(method_names))
    width = 0.55
    bottom = np.zeros(len(method_names))

    for i, level in enumerate(LABEL_ORDER):
        vals = [data[m][i] for m in method_names]
        bars = ax.bar(x, vals, width, bottom=bottom,
                       color=LABEL_COLORS_JAP[i], label=level, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(v), ha='center', va='center', fontsize=8, color='white')
        bottom += vals

    ax.set_xlabel('Weighting Method', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, bottom.max() * 1.12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_label_consistency_heatmap(consistency_df, title, save_path):
    """标签一致性热力图"""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(consistency_df.values, dtype=bool)
    np.fill_diagonal(mask, True)
    cmap = sns.light_palette("#2ecc71", as_cmap=True, reverse=False)

    sns.heatmap(consistency_df, annot=True, fmt='.1f', cmap=cmap,
                vmin=50, vmax=100, square=True, linewidths=1,
                cbar_kws={'label': 'Label Agreement (%)'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_score_distribution(df, score_cols, method_names, title, save_path):
    """风险评分箱线图"""
    plot_data = []
    for i, m in enumerate(method_names):
        col = score_cols[i]
        vals = df[col].values
        for v in vals:
            plot_data.append({'Method': m, 'Risk Score': v})
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    parts = ax.boxplot([plot_df[plot_df['Method'] == m]['Risk Score'].values
                        for m in method_names],
                       patch_artist=True,
                       widths=0.6,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=4))

    for patch, color in zip(parts['boxes'], METHOD_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylabel('Risk Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_score_violin(df, score_cols, method_names, title, save_path):
    """风险评分小提琴图（比箱线图展示分布更精细）"""
    plot_data = []
    for i, m in enumerate(method_names):
        col = score_cols[i]
        for v in df[col].values:
            plot_data.append({'Method': m, 'Risk Score': v})
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = dict(zip(method_names, METHOD_COLORS))
    sns.violinplot(x='Method', y='Risk Score', data=plot_df,
                   palette=palette, inner='quartile', ax=ax, linewidth=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Risk Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_label_transition_heatmaps(df, method_names, baseline, title, save_path):
    """各方法 vs 基准的标签变迁热力图（4 张小图排列）"""
    others = [m for m in method_names if m != baseline]
    n_others = len(others)
    cols = 3
    rows = (n_others + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4 + 1, rows * 3.5 + 0.5))
    axes = axes.flatten()

    for idx, method in enumerate(others):
        ax = axes[idx]
        matrix = label_transition_matrix(df, baseline, method)
        sns.heatmap(matrix, annot=True, fmt='.1%', cmap='YlOrRd',
                    vmin=0, vmax=1, square=True, linewidths=1,
                    cbar_kws={'shrink': 0.75}, ax=ax)
        ax.set_title(f'{baseline} vs {method}', fontsize=11)
        ax.set_ylabel(f'{baseline} Label', fontsize=9)
        ax.set_xlabel(f'{method} Label', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    for idx in range(n_others, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_top10_overlap_heatmap(overlap_df, title, save_path):
    """Top-10% 高风险重合度热力图"""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(overlap_df, annot=True, fmt='.1%', cmap='YlGnBu',
                vmin=0.3, vmax=1.0, square=True, linewidths=1,
                cbar_kws={'label': 'Jaccard Similarity'}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_spearman_heatmap(spearman_df, title, save_path):
    """Spearman 秩相关系数热力图"""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(spearman_df, annot=True, fmt='.4f', cmap='RdYlBu',
                vmin=0.7, vmax=1.0, square=True, linewidths=1,
                cbar_kws={'label': "Spearman's ρ"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_score_scatter_matrix(df, score_cols, method_names, title, save_path):
    """评分散点矩阵（下三角散点 + 上三角 Spearman ρ）"""
    n = len(method_names)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.5 + 1, n * 2.5 + 0.5))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # 对角线：直方图
                ax.hist(df[score_cols[i]].values, bins=30,
                        color=METHOD_COLORS[i], alpha=0.6, edgecolor='white')
                ax.set_title(method_names[i], fontsize=8, fontweight='bold')
            elif i > j:
                # 下三角：散点图
                ax.scatter(df[score_cols[j]], df[score_cols[i]],
                          s=2, alpha=0.15, color=METHOD_COLORS[j], edgecolors='none')
                # 对角线参考线
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                        max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
            else:
                # 上三角：Spearman ρ
                rho, _ = spearmanr(df[score_cols[j]], df[score_cols[i]])
                ax.text(0.5, 0.5, f'ρ={rho:.4f}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='darkred' if rho < 0.9 else 'darkgreen')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if j == 0:
                ax.set_ylabel(method_names[i], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(method_names[j], fontsize=7)
            ax.tick_params(labelsize=5)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


# ── 主流程 ──


def run_sensitivity(data_type='lc'):
    """运行敏感性验证"""
    scenario = '变道车辆' if data_type == 'lc' else '跟驰车辆'
    scenario_en = 'Lane Change' if data_type == 'lc' else 'Following'
    metrics_config = METRICS if data_type == 'lc' else FOLLOWING_METRICS
    thresholds = THRESH_LANE_CHANGE if data_type == 'lc' else THRESH_FOLLOWING

    print(f'\n{"=" * 72}')
    print(f'  敏感性验证：{scenario}')
    print(f'{"=" * 72}')

    # 1. 获取文件列表
    files = _get_csv_files(data_type)
    print(f'  数据文件: {len(files)} 个')

    # 2. 计算 6 种权重
    weight_dict = _get_weight_dict(files, metrics_config, data_type)
    method_names = list(weight_dict.keys())
    print(f'  权重方法: {", ".join(method_names)}')
    for m in method_names:
        print(f'    {m}: {np.array2string(weight_dict[m], precision=4, separator=", ")}')

    # 3. 计算风险评分和标签
    result_df, score_cols = compute_scores(files, metrics_config, weight_dict, thresholds)
    print(f'  样本数: {len(result_df)}')

    # 4. 分析指标
    # 4a. 标签一致性
    consistency_df = label_consistency(result_df, method_names, 'Expert')

    # 4b. vs 基准标签一致性明细
    vs_baseline = label_consistency_vs_baseline(result_df, method_names, 'Expert')

    # 4c. Top-10% 重合度
    overlap_df = top10_overlap(result_df, method_names, 'Expert')

    # 4d. Spearman 相关
    spearman_df = spearman_corr(result_df, method_names)

    # 5. 控制台输出
    print(f'\n  ── 标签一致性 vs Expert（基准） ──')
    for _, row in vs_baseline.iterrows():
        m = row['method']
        print(f'  {m:<15s} 总体={row["overall_agree"]:5.1f}%  '
              f'高风险={row["高风险_agree"]:5.1f}%  '
              f'中风险={row["中风险_agree"]:5.1f}%  '
              f'低风险={row["低风险_agree"]:5.1f}%')

    print(f'\n  ── Top-10% 高风险 Jaccard 相似度 ──')
    for m in method_names:
        print(f'  Expert vs {m:<15s} = {overlap_df.loc["Expert", m]:.1%}')

    print(f'\n  ── Spearman 秩相关系数 vs Expert ──')
    for m in method_names:
        print(f'  Expert vs {m:<15s} = {spearman_df.loc["Expert", m]:.4f}')

    # 6. 可视化
    out_sub = os.path.join(OUT_DIR, data_type)
    os.makedirs(out_sub, exist_ok=True)

    # 6a. 标签分布堆叠柱状图
    plot_label_distribution(result_df, method_names,
                            f'{scenario_en} Label Distribution by Weighting Method',
                            os.path.join(out_sub, '01_label_distribution.png'))

    # 6b. 标签一致性热力图
    plot_label_consistency_heatmap(consistency_df,
                                   f'{scenario_en} Label Agreement (%)',
                                   os.path.join(out_sub, '02_label_consistency_heatmap.png'))

    # 6c. 风险评分箱线图
    plot_score_distribution(result_df, score_cols, method_names,
                            f'{scenario_en} Risk Score Box Plot by Method',
                            os.path.join(out_sub, '03_score_boxplot.png'))

    # 6d. 风险评分小提琴图
    plot_score_violin(result_df, score_cols, method_names,
                      f'{scenario_en} Risk Score Violin Plot by Method',
                      os.path.join(out_sub, '04_score_violin.png'))

    # 6e. Top-10% 重合度热力图
    plot_top10_overlap_heatmap(overlap_df,
                               f'{scenario_en} Top-10% High-Risk Jaccard Similarity',
                               os.path.join(out_sub, '05_top10_overlap_heatmap.png'))

    # 6f. Spearman 相关系数
    plot_spearman_heatmap(spearman_df,
                          f'{scenario_en} Spearman Rank Correlation',
                          os.path.join(out_sub, '06_spearman_heatmap.png'))

    # 6g. 标签变迁热力图
    plot_label_transition_heatmaps(result_df, method_names, 'Expert',
                                   f'{scenario_en} Label Transitions vs Expert',
                                   os.path.join(out_sub, '07_label_transition.png'))

    # 6h. 评分散点矩阵
    plot_score_scatter_matrix(result_df, score_cols, method_names,
                              f'{scenario_en} Risk Score Scatter Matrix',
                              os.path.join(out_sub, '08_score_scatter_matrix.png'))

    # 7. 保存结果 CSV
    csv_path = os.path.join(OUT_DIR, f'{data_type}_sensitivity_results.csv')
    result_df.to_csv(csv_path, index=False)
    print(f'\n  Saved: {csv_path}')

    return result_df, weight_dict, consistency_df, vs_baseline, overlap_df, spearman_df


if __name__ == '__main__':
    print('=' * 72)
    print('  赋权方法敏感性验证')
    print('  比较 Expert / EWM / CRITIC / AHP / Multiplicative / MeanDev')
    print('=' * 72)

    # 变道
    lc_result, lc_w, lc_c, lc_vs, lc_ov, lc_sp = run_sensitivity('lc')

    # 跟驰
    fl_result, fl_w, fl_c, fl_vs, fl_ov, fl_sp = run_sensitivity('fl')

    print(f'\n{"=" * 72}')
    print('  所有结果已保存到:')
    print(f'  {OUT_DIR}')
    print(f'{"=" * 72}')
