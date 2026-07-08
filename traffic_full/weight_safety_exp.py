"""
熵权法 (EWM) 客观权重计算 — 与 safety_scoring_exp.py 独立

从 safety_scoring_exp.py 中分离出来的 EWM 计算函数，
safety_scoring_exp.py 中仍保留 METRICS/FOLLOWING_METRICS 权重定义。

用法:
    from weight_safety_exp import run_ewm_analysis
    run_ewm_analysis({'location1': 'E:/0little/location1', ...})
"""
import numpy as np
import os
import pandas as pd


# ── 引入指标配置和 _frame_contrib ──
from safety_scoring_exp import METRICS, FOLLOWING_METRICS, _frame_contrib


def compute_ewm_weights(X):
    """熵权法计算客观权重

    参数:
        X: np.ndarray, shape (n_samples, n_metrics)

    返回:
        weights:  np.ndarray (n_metrics,)  客观权重，求和为 1
        entropy:  np.ndarray (n_metrics,)  各指标信息熵 E_j
        diversity: np.ndarray (n_metrics,) 各指标差异系数 D_j = 1 - E_j
    """
    n = X.shape[0]
    m = X.shape[1]
    if n == 0 or m == 0:
        raise ValueError(f'X 不能为空: shape={X.shape}')

    col_sums = X.sum(axis=0)
    P = np.zeros_like(X, dtype=float)
    for j in range(m):
        if col_sums[j] > 0:
            P[:, j] = X[:, j] / col_sums[j]

    entropy = np.zeros(m)
    ln_n = np.log(n) if n > 1 else 1.0
    for j in range(m):
        pj = P[:, j]
        pj_pos = pj[pj > 0]
        if len(pj_pos) > 0:
            entropy[j] = -np.sum(pj_pos * np.log(pj_pos)) / ln_n

    diversity = 1.0 - entropy
    if diversity.sum() > 0:
        weights = diversity / diversity.sum()
    else:
        weights = np.ones(m) / m

    return weights, entropy, diversity


def compute_conditional_ewm_weights(X):
    """条件熵权法：每个指标仅在激活值>0的车辆子集上计算"""
    n, m = X.shape
    if n == 0 or m == 0:
        raise ValueError(f'X 不能为空: shape={X.shape}')

    entropy = np.zeros(m)
    subset_sizes = np.zeros(m, dtype=int)

    for j in range(m):
        col = X[:, j]
        mask = col > 0
        n_j = mask.sum()
        subset_sizes[j] = n_j
        if n_j == 0:
            entropy[j] = 0.0
            continue
        col_sum = col[mask].sum()
        P = col[mask] / col_sum if col_sum > 0 else np.zeros(n_j)
        P_pos = P[P > 0]
        ln_nj = np.log(n_j) if n_j > 1 else 1.0
        if len(P_pos) > 0:
            entropy[j] = -np.sum(P_pos * np.log(P_pos)) / ln_nj

    diversity = 1.0 - entropy
    if diversity.sum() > 0:
        weights = diversity / diversity.sum()
    else:
        weights = np.ones(m) / m

    return weights, entropy, diversity, list(subset_sizes)


def _metric_activation(grp, col, k, valid_fn):
    """计算单辆车在单个指标上的最大风险激活值"""
    if col not in grp.columns:
        return 0.0
    vals = grp[col].values.astype(float)
    valid = valid_fn(vals)
    if not valid.any():
        return 0.0
    return float(np.nanmax(_frame_contrib(vals[valid], k)))


def build_activation_matrix(file_paths, metrics_config):
    """从 CSV 文件列表构建风险激活矩阵 X"""
    rows = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        for (vid, src), grp in df.groupby(['ID', 'Source']):
            grp = grp.sort_values('Frame')
            row = [_metric_activation(grp, m['col'], m['k'], m['valid'])
                   for m in metrics_config]
            rows.append(row)

    X = np.array(rows, dtype=float)
    names = [m['name'] for m in metrics_config]
    return X, names


def _print_ewm_table(names, orig_metrics, w_unc, e_unc, d_unc,
                     w_cond, e_cond, d_cond, sizes, title, n_total):
    """统一的 EWM 对比表输出"""
    print(f'\n  {title} (n={n_total} 辆)')
    print(f'  {"指标":<10s} {"经验权重":>7s} {"无条件EWM":>9s}  '
          f'{"信息熵E":>8s}  {"差异D":>8s}')
    print(f'  {"-" * 60}')
    for j, name in enumerate(names):
        orig_w = orig_metrics[j]['w']
        print(f'  {name:<10s} {orig_w:7.4f} {w_unc[j]:9.4f}  '
              f'{e_unc[j]:8.4f}  {d_unc[j]:8.4f}')
    if len(w_unc) == len(names):
        rho_unc = np.corrcoef([m['w'] for m in orig_metrics], w_unc)[0, 1]
        print(f'  ── Spearman ρ(经验 vs 无条件) = {rho_unc:.4f}')
    else:
        rho_unc = float('nan')

    print(f'\n  条件 EWM（仅含对应邻车的车辆）:')
    print(f'  {"指标":<10s} {"经验权重":>7s} {"条件EWM":>9s}  '
          f'{"信息熵E":>8s}  {"差异D":>8s}  {"|S_j|":>6s}')
    print(f'  {"-" * 60}')
    for j, name in enumerate(names):
        orig_w = orig_metrics[j]['w']
        print(f'  {name:<10s} {orig_w:7.4f} {w_cond[j]:9.4f}  '
              f'{e_cond[j]:8.4f}  {d_cond[j]:8.4f}  {sizes[j]:>6d}')
    if len(w_cond) == len(names):
        rho_cond = np.corrcoef([m['w'] for m in orig_metrics], w_cond)[0, 1]
        print(f'  ── Spearman ρ(经验 vs 条件) = {rho_cond:.4f}')
    else:
        rho_cond = float('nan')

    print(f'\n  结论: 无条件 ρ={rho_unc:.4f}, 条件 ρ={rho_cond:.4f}')
    return rho_unc, rho_cond


def run_ewm_analysis(locs, base_dir='E:/0little', loc_v0=None):
    """对变道和跟驰车辆分别运行无条件+条件熵权法并打印对比结果"""
    print('\n' + '=' * 72)
    print('  熵权法 (EWM) 客观权重计算 —— 无条件 vs 条件')
    print('=' * 72)

    # ── 变道车辆 ──
    lc_files = []
    for loc_key in locs:
        for side in ['left', 'right']:
            fp = os.path.normpath(
                os.path.join(base_dir, loc_key, f'traffic_{side}_change.csv'))
            if os.path.exists(fp):
                lc_files.append(fp)

    if lc_files:
        X_lc, lc_names = build_activation_matrix(lc_files, METRICS)
        w_unc, e_unc, d_unc = compute_ewm_weights(X_lc)
        w_cond, e_cond, d_cond, sizes = compute_conditional_ewm_weights(X_lc)
        _print_ewm_table(lc_names, METRICS,
                         w_unc, e_unc, d_unc,
                         w_cond, e_cond, d_cond, sizes,
                         '变道车辆', X_lc.shape[0])
    else:
        X_lc = None
        print('\n  [WARN] 未找到变道数据文件')

    # ── 跟驰车辆 ──
    fl_files = []
    for loc_key in locs:
        fp = os.path.normpath(
            os.path.join(base_dir, loc_key, 'traffic_following_change.csv'))
        if os.path.exists(fp):
            fl_files.append(fp)

    if fl_files:
        X_fl, fl_names = build_activation_matrix(fl_files, FOLLOWING_METRICS)
        w_unc, e_unc, d_unc = compute_ewm_weights(X_fl)
        w_cond, e_cond, d_cond, sizes = compute_conditional_ewm_weights(X_fl)
        _print_ewm_table(fl_names, FOLLOWING_METRICS,
                         w_unc, e_unc, d_unc,
                         w_cond, e_cond, d_cond, sizes,
                         '跟驰车辆', X_fl.shape[0])
    else:
        print('\n  [WARN] 未找到跟驰数据文件')


if __name__ == '__main__':
    # 快速测试用
    locs_test = {f'location{i}': f'E:/0little/location{i}' for i in range(1, 6)}
    run_ewm_analysis(locs_test)
