"""
雷达图可视化 — 多维度模型性能对比

提供三种雷达图：
  1. 跨区域六轴雷达图（所有模型）：Acc, WeightedF1, MacroF1, 1-CV, 高风险F1, 高风险Recall
  2. 随机划分六轴雷达图（所有模型）：同上（1-CV 从跨区域传入）
  3. XGBoost per-location 雷达图：Acc, WeightedF1, MacroF1, 1-CV?, 高风险F1, test/train ratio

用法:
    from radar_chart import (plot_cross_location_radar,
                             plot_random_split_radar,
                             plot_xgboost_per_location_radar)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_COLORS = {'XGBoost': '#e74c3c', 'RandomForest': '#3498db',
                'MLP': '#2ecc71', 'LSTM': '#f39c12'}
LOC_COLORS = {'Loc1': '#3498db', 'Loc2': '#2ecc71',
              'Loc3': '#f39c12', 'Loc4': '#9b59b6', 'Loc5': '#e74c3c'}
LOC_LABELS = {'location1': 'Loc1', 'location2': 'Loc2', 'location3': 'Loc3',
              'location4': 'Loc4', 'location5': 'Loc5'}

AXIS_LABELS = ['Accuracy', 'Weighted\nF1', 'Macro\nF1', '1-CV\n(Stability)', 'High-Risk\nF1', 'High-Risk\nRecall']
AXIS_LABELS_LOC = ['Accuracy', 'Weighted\nF1', 'Macro\nF1', '1-CV\n(Stability)', 'High-Risk\nF1', 'Test/Train\nRatio']
# 单轴上限：非 1-CV 轴压缩到 0.8，1-CV（索引3）保持 1.0
RLIM_PER_AXIS = [0.9, 0.9, 0.9, 1.0, 0.9, 0.9]


def _radar_factory(num_vars):
    """生成雷达图角度"""
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    return angles


def _plot_radar(values_dict, title, save_name, axis_labels, colors, out_dir,
                axis_rlim=None):
    """通用雷达图绘制函数

    values_dict: {name: [v1, v2, v3, v4, v5, v6]}  每个条目一条线
    axis_rlim: [max1, max2, ..., maxN]  每轴独立上限，None 表示与全局 rlim 一致
             例: [0.8, 0.8, 0.8, 1.0, 0.8, 0.8]  让 1-CV 占更大范围
    """
    N = len(axis_labels)
    angles = _radar_factory(N)

    # 如果没有指定单轴上限，统一默认值
    if axis_rlim is None:
        axis_rlim = [1.0] * N

    global_rlim = max(axis_rlim)  # 全局径向轴以最宽值为准

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 轴标签附带最大值标注
    labeled_axis = [f'{a}\n(max={r:.1f})' for a, r in zip(axis_labels, axis_rlim)]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labeled_axis, fontsize=9)

    # 刻度环：保留网格线，隐藏数字
    ax.set_rlabel_position(30)
    ax.set_rlim(0, global_rlim)
    ticks = np.linspace(0, global_rlim, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([''] * len(ticks))
    ax.grid(True, alpha=0.3)

    # 在软上限处画一条虚线环（如果 global_rlim > 软上限）
    soft_max = min(axis_rlim)
    if global_rlim > soft_max + 0.01:
        theta_line = np.linspace(0, 2 * np.pi, 200)
        ax.plot(theta_line, [soft_max] * 200, color='gray', linewidth=1,
                linestyle='--', alpha=0.4)

    # 对每个模型的数值做单轴归一化
    for i, (name, vals) in enumerate(values_dict.items()):
        values = vals + vals[:1]
        # 归一化到全局尺度
        norm_vals = np.array([v / r if r > 0 else v for v, r in zip(values, axis_rlim + axis_rlim[:1])])
        color = colors.get(name, '#333333')
        ax.plot(angles, norm_vals, 'o-', linewidth=2, color=color, label=name, alpha=0.85)
        ax.fill(angles, norm_vals, alpha=0.08, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.95)
    fig.tight_layout()
    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {save_path}')
    return True


def plot_cross_location_radar(fold_metrics, models, save_name, out_dir='.'):
    """跨区域六轴雷达图（所有模型对比）

    六轴: Acc(mean), WeightedF1(mean), MacroF1(mean), 1-CV(F1), 高风险F1(mean), 高风险Recall(mean)
    """
    names = [m for m in models if m in fold_metrics]
    values_dict = {}
    for name in names:
        d = fold_metrics[name]
        cv = np.std(d['f1']) / max(np.mean(d['f1']), 1e-6)
        stability = 1 - min(cv, 1.0)
        values_dict[name] = [
            np.mean(d['acc']),
            np.mean(d['f1']),
            np.mean(d['macro_f1']),
            stability,
            np.mean(d['high_risk_f1']),
            np.mean(d['high_risk_recall']),
        ]
    return _plot_radar(values_dict, '跨区域验证 — 模型性能雷达图',
                       save_name, AXIS_LABELS, MODEL_COLORS, out_dir,
                       axis_rlim=RLIM_PER_AXIS)


def plot_random_split_radar(random_results, models, save_name, out_dir='.', stability_vals=None):
    """随机划分六轴雷达图（所有模型对比）

    stability_vals: {name: stability} 从跨区域结果传入的 1-CV 值
    六轴: Acc, WeightedF1, MacroF1, 1-CV(external), 高风险F1, 高风险Recall
    """
    names = [m for m in models if m in random_results]
    values_dict = {}
    for name in names:
        d = random_results[name]
        stability = stability_vals.get(name, 0.5) if stability_vals else 0.5
        values_dict[name] = [
            d['acc'],
            d['f1'],
            d['macro_f1'],
            stability,
            d.get('high_risk_f1', 0),
            d.get('high_risk_recall', 0),
        ]
    return _plot_radar(values_dict, '随机划分 — 模型性能雷达图',
                       save_name, AXIS_LABELS, MODEL_COLORS, out_dir,
                       axis_rlim=RLIM_PER_AXIS)


def plot_xgboost_per_location_radar(fold_metrics, loc_keys, save_name, out_dir='.'):
    """XGBoost 各 location 雷达图（per-location 对比）

    六轴: Acc, WeightedF1, MacroF1, (同一 CV 或本 fold 值), 高风险F1, test/train ratio
    每条线代表一个 location
    """
    if 'XGBoost' not in fold_metrics:
        print('  [SKIP] radar: XGBoost not found in fold_metrics')
        return

    n_locs = len(loc_keys)
    xgb = fold_metrics['XGBoost']
    values_dict = {}

    for i, loc in enumerate(loc_keys):
        # 同一模型各 fold 间的 CV
        cv = np.std(xgb['f1']) / max(np.mean(xgb['f1']), 1e-6)
        values_dict[LOC_LABELS.get(loc, loc)] = [
            xgb['acc'][i],
            xgb['f1'][i],
            xgb['macro_f1'][i],
            1 - min(cv, 1.0),  # 使用全局 CV（跨所有 location 的稳定性）
            xgb['high_risk_f1'][i],
            xgb['test_ratio'][i],
        ]
    return _plot_radar(values_dict, 'XGBoost — 各场景雷达图',
                       save_name, AXIS_LABELS_LOC, LOC_COLORS, out_dir,
                       axis_rlim=RLIM_PER_AXIS)
