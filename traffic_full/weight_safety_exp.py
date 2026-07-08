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


# ── CRITIC 法 ──


def compute_critic_weights(X):
    """CRITIC 法计算客观权重

    CRITIC (Criteria Importance Through Intercriteria Correlation) 同时考虑：
      - 对比强度: 标准差 σ_j（指标 j 的数值离散程度）
      - 冲突性: Σ(1 - r_jk)（指标 j 与其他指标的信息重叠程度，
                r_jk 为 Pearson 相关系数）

    综合信息量 C_j = σ_j · Σ(1 - r_jk)，归一化得权重 w_j = C_j / ΣC_k

    参数:
        X: np.ndarray, shape (n_samples, n_metrics)  风险激活矩阵

    返回:
        weights:     np.ndarray (m,)  归一化 CRITIC 权重
        contrast:    np.ndarray (m,)  对比强度 σ_j（总体标准差 ddof=0）
        conflict:    np.ndarray (m,)  冲突度 Σ(1 - r_jk)
        corr_matrix: np.ndarray (m,m) 指标间 Pearson 相关系数矩阵
    """
    n, m = X.shape
    if n == 0 or m == 0:
        raise ValueError(f'X 不能为空: shape={X.shape}')
    if m == 1:
        std_val = np.std(X[:, 0], ddof=0)
        return np.array([1.0]), np.array([std_val]), np.array([0.0]), np.ones((1, 1))

    # 对比强度：总体标准差
    contrast = np.std(X, axis=0, ddof=0)

    # 相关系数矩阵 (m × m)
    corr_matrix = np.corrcoef(X.T)
    # 处理 NaN：某指标方差为 0 时相关系数为 NaN，置为 0（无相关性）
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # 冲突度：每个指标与所有指标的不相关程度之和（含自相关 1-1=0 不影响）
    conflict = np.sum(1.0 - corr_matrix, axis=1)

    # 综合信息量 C_j = σ_j · Σ(1 - r_jk)
    C = contrast * conflict

    # 归一化权重
    C_sum = C.sum()
    if C_sum > 0:
        weights = C / C_sum
    else:
        weights = np.ones(m) / m

    return weights, contrast, conflict, corr_matrix


# ── AHP 法 ──


# AHP 判断矩阵（指标顺序与 METRICS/FOLLOWING_METRICS 一致）
# 变道: [mTTC, THW, PET, F_ETTC, OL_PET]
LC_AHP_MATRIX = np.array([
    [1,    1/5,  1/3,  1/2,  1/3],
    [5,    1,    3,    4,    3],
    [3,    1/3,  1,    2,    1],
    [2,    1/4,  1/2,  1,    1/2],
    [3,    1/3,  1,    2,    1],
])

# 跟驰: [mTTC, THW, B_mTTC]
FL_AHP_MATRIX = np.array([
    [1,    1/3,  3],
    [3,    1,    5],
    [1/3,  1/5,  1],
])


def compute_ahp_weights(judgment_matrix, names=None):
    """AHP 层次分析法计算主观权重

    AHP (Analytic Hierarchy Process) 基于 Saaty 1-9 标度判断矩阵，
    通过最大特征值对应的特征向量确定权重，并进行一致性检验。

    参数:
        judgment_matrix: np.ndarray (m, m)  成对比较判断矩阵
        names: list, optional               指标名称（用于一致性提示）

    返回:
        weights:    np.ndarray (m,)    归一化权重
        lambda_max: float              最大特征值
        cr:         float              一致性比率 CR（<0.1 可接受）
        ci:         float              一致性指标 CI
    """
    m = judgment_matrix.shape[0]
    if judgment_matrix.shape != (m, m):
        raise ValueError(f'判断矩阵必须为方阵，当前形状: {judgment_matrix.shape}')

    # 特征值分解
    eigvals, eigvecs = np.linalg.eig(judgment_matrix)
    idx = np.argmax(eigvals.real)
    lambda_max = eigvals[idx].real
    principal = eigvecs[:, idx].real

    # 归一化得权重
    weights = principal / principal.sum()

    # 一致性检验
    ci = (lambda_max - m) / (m - 1) if m > 1 else 0.0
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(m, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return weights, lambda_max, cr, ci


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


# ── CRITIC 打印辅助 ──


def _print_corr_matrix(names, corr):
    """打印相关系数矩阵"""
    header = f'{"":>10s}'
    for name in names:
        short = name[:8]
        header += f' {short:>8s}'
    print(header)
    for i, name_i in enumerate(names):
        row = f'{name_i:>10s}'
        for j in range(len(names)):
            row += f' {corr[i, j]:8.4f}'
        print(row)


def _print_critic_table(names, orig_metrics,
                         w_unc, cont_unc, confl_unc,
                         w_int, cont_int, confl_int,
                         corr_unc, corr_int,
                         n_total, n_int, title):
    """CRITIC 权重详细输出：两套权重 + 相关系数矩阵"""
    print(f'\n  >> {title} CRITIC 权重（无条件, n={n_total}）')
    print(f'  {"指标":<10s} {"CRITIC":>9s} {"对比强度σ":>9s} '
          f'{"冲突度Σ(1-r)":>12s} {"C_j":>9s} {"经验权重":>9s}')
    print(f'  {"-" * 65}')
    for j, name in enumerate(names):
        C_j = cont_unc[j] * confl_unc[j]
        orig_w = orig_metrics[j]['w']
        print(f'  {name:<10s} {w_unc[j]:9.4f} {cont_unc[j]:9.4f} '
              f'{confl_unc[j]:12.4f} {C_j:9.4f} {orig_w:9.4f}')

    if n_int > 0:
        print(f'\n  >> {title} CRITIC 权重（交集子集, n={n_int}）')
        print(f'  {"指标":<10s} {"CRITIC":>9s} {"对比强度σ":>9s} '
              f'{"冲突度Σ(1-r)":>12s} {"C_j":>9s} {"经验权重":>9s}')
        print(f'  {"-" * 65}')
        for j, name in enumerate(names):
            C_j = cont_int[j] * confl_int[j]
            orig_w = orig_metrics[j]['w']
            print(f'  {name:<10s} {w_int[j]:9.4f} {cont_int[j]:9.4f} '
                  f'{confl_int[j]:12.4f} {C_j:9.4f} {orig_w:9.4f}')
    else:
        print('  [交集子集为空，跳过]')

    # 相关系数矩阵（无条件）
    print(f'\n  相关系数矩阵 R（无条件, n={n_total}）:')
    _print_corr_matrix(names, corr_unc)

    # 相关系数矩阵（交集子集）
    if n_int > 0:
        print(f'\n  相关系数矩阵 R（交集子集, n={n_int}）:')
        _print_corr_matrix(names, corr_int)


def _print_weight_comparison(names, orig_metrics,
                              w_ewm_unc, w_ewm_cond,
                              w_critic_unc, w_critic_int,
                              n_int, title):
    """EWM + CRITIC + 经验权重综合对比表"""
    print(f'\n  === {title}：综合权重对比 ===')
    print(f'  {"指标":<10s} {"经验权重":>9s} {"EWM无条件":>9s} '
          f'{"EWM条件":>9s} {"CRITIC无条件":>12s} {"CRITIC交集":>11s}')
    print(f'  {"-" * 65}')
    for j, name in enumerate(names):
        orig_w = orig_metrics[j]['w']
        int_str = f'{w_critic_int[j]:.4f}' if n_int > 0 else '   N/A'
        print(f'  {name:<10s} {orig_w:9.4f} {w_ewm_unc[j]:9.4f} '
              f'{w_ewm_cond[j]:9.4f} {w_critic_unc[j]:12.4f} {int_str:>11s}')


# ── AHP 打印辅助 ──


def _print_ahp_table(names, weights, lambda_max, cr, ci, title, matrix):
    """AHP 权重详细输出（含判断矩阵 + 一致性检验）"""
    print(f'\n  >> {title} AHP 权重')
    print(f'  判断矩阵（顺序: {", ".join(names)}）:')
    header = f'{"":>10s}'
    for name in names:
        header += f' {name:>8s}'
    print(header)
    for i, name_i in enumerate(names):
        row = f'{name_i:>10s}'
        for j in range(len(names)):
            row += f' {matrix[i, j]:8.4f}'
        print(row)

    print(f'\n  {"指标":<10s} {"AHP权重":>8s}')
    print(f'  {"-" * 25}')
    for j, name in enumerate(names):
        print(f'  {name:<10s} {weights[j]:8.4f}')
    print(f'  lambda_max = {lambda_max:.4f}')
    print(f'  CI = {ci:.4f}')
    print(f'  CR = {cr:.4f}', end='')
    if cr < 0.1:
        print('  (OK, CR<0.1)')
    else:
        print('  (WARNING: CR>=0.1, 需调整判断矩阵)')


# ── 组合赋权 ──


def compute_combined_weights(w_dict):
    """组合赋权

    两种策略:
      1) 乘法归一化: w_j = (prod_k w_j^k)^{1/K} / 归一化
      2) 离差最小化: w_j = mean(w_j^k) / 归一化

    参数:
        w_dict: dict, 形如 {'EWM': np.array, 'CRITIC': np.array, 'AHP': np.array}

    返回:
        dict: {'multiplicative': np.array, 'min_deviation': np.array}
    """
    names_list = list(w_dict.keys())
    weight_arrays = [w_dict[n] for n in names_list]
    m = weight_arrays[0].shape[0]

    # 策略 1: 乘法归一化（几何平均后归一化）
    product = np.ones(m)
    for w in weight_arrays:
        product *= w
    w_multi = product ** (1.0 / len(weight_arrays))
    w_multi = w_multi / w_multi.sum()

    # 策略 2: 离差最小化（算术平均后归一化）
    w_min_dev = np.mean(weight_arrays, axis=0)
    w_min_dev = w_min_dev / w_min_dev.sum()

    return {'multiplicative': w_multi, 'min_deviation': w_min_dev}


def _print_full_comparison(names, orig_metrics,
                            w_ewm_unc, w_ewm_cond,
                            w_critic_unc, w_critic_int, n_int,
                            w_ahp, w_multi, w_min_dev, title):
    """全方法权重对比表（9 列：经验 + EWM×2 + CRITIC×2 + AHP + 组合×2）"""
    print(f'\n  === {title}：全方法权重对比 ===')
    print(f'  {"指标":<8s} {"经验":>7s} {"EWM无":>7s} {"EWM条":>7s} '
          f'{"CRITIC无":>9s} {"CRITIC交":>9s} '
          f'{"AHP":>7s} {"乘法组合":>9s} {"均值组合":>9s}')
    print(f'  {"-" * 85}')
    for j, name in enumerate(names):
        orig_w = orig_metrics[j]['w']
        int_str = f'{w_critic_int[j]:.4f}' if n_int > 0 else '   N/A'
        print(f'  {name:<8s} {orig_w:7.4f} {w_ewm_unc[j]:7.4f} {w_ewm_cond[j]:7.4f} '
              f'{w_critic_unc[j]:9.4f} {int_str:>9s} '
              f'{w_ahp[j]:7.4f} {w_multi[j]:9.4f} {w_min_dev[j]:9.4f}')


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


def run_critic_analysis(locs, base_dir='E:/0little'):
    """对变道和跟驰分别计算 CRITIC 权重（无条件 + 交集子集）并打印"""
    print('\n' + '=' * 72)
    print('  CRITIC 法客观权重计算 —— 无条件 vs 交集子集')
    print('=' * 72)

    results = {}

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
        n_lc = X_lc.shape[0]

        # 无条件 CRITIC
        w_unc, cont_unc, confl_unc, corr_unc = compute_critic_weights(X_lc)

        # 交集子集 CRITIC（所有指标同时 >0 的样本，即全部邻车存在）
        mask_int = (X_lc > 0).all(axis=1)
        X_int = X_lc[mask_int]
        n_lc_int = X_int.shape[0]
        if n_lc_int > 0:
            w_int, cont_int, confl_int, corr_int = compute_critic_weights(X_int)
        else:
            w_int = cont_int = confl_int = corr_int = None

        _print_critic_table(lc_names, METRICS,
                            w_unc, cont_unc, confl_unc,
                            w_int, cont_int, confl_int,
                            corr_unc, corr_int,
                            n_lc, n_lc_int, '变道车辆')

        results['lc'] = {
            'names': lc_names, 'metrics': METRICS,
            'w_critic_unc': w_unc, 'w_critic_int': w_int,
            'n_int': n_lc_int,
        }
    else:
        print('\n  [WARN] 未找到变道数据文件')
        results['lc'] = None

    # ── 跟驰车辆 ──
    fl_files = []
    for loc_key in locs:
        fp = os.path.normpath(
            os.path.join(base_dir, loc_key, 'traffic_following_change.csv'))
        if os.path.exists(fp):
            fl_files.append(fp)

    if fl_files:
        X_fl, fl_names = build_activation_matrix(fl_files, FOLLOWING_METRICS)
        n_fl = X_fl.shape[0]

        w_unc, cont_unc, confl_unc, corr_unc = compute_critic_weights(X_fl)

        mask_int = (X_fl > 0).all(axis=1)
        X_int = X_fl[mask_int]
        n_fl_int = X_int.shape[0]
        if n_fl_int > 0:
            w_int, cont_int, confl_int, corr_int = compute_critic_weights(X_int)
        else:
            w_int = cont_int = confl_int = corr_int = None

        _print_critic_table(fl_names, FOLLOWING_METRICS,
                            w_unc, cont_unc, confl_unc,
                            w_int, cont_int, confl_int,
                            corr_unc, corr_int,
                            n_fl, n_fl_int, '跟驰车辆')

        results['fl'] = {
            'names': fl_names, 'metrics': FOLLOWING_METRICS,
            'w_critic_unc': w_unc, 'w_critic_int': w_int,
            'n_int': n_fl_int,
        }
    else:
        print('\n  [WARN] 未找到跟驰数据文件')
        results['fl'] = None

    return results


def run_ahp_analysis():
    """运行 AHP 层次分析法并打印结果"""
    print('\n' + '=' * 72)
    print('  AHP 层次分析法主观权重')
    print('=' * 72)

    results = {}

    # ── 变道车辆（5 指标）──
    lc_names = [m['name'] for m in METRICS]
    w_lc, lam_lc, cr_lc, ci_lc = compute_ahp_weights(LC_AHP_MATRIX, lc_names)
    _print_ahp_table(lc_names, w_lc, lam_lc, cr_lc, ci_lc, '变道车辆', LC_AHP_MATRIX)
    results['lc'] = {'names': lc_names, 'weights': w_lc}

    # ── 跟驰车辆（3 指标）──
    fl_names = [m['name'] for m in FOLLOWING_METRICS]
    w_fl, lam_fl, cr_fl, ci_fl = compute_ahp_weights(FL_AHP_MATRIX, fl_names)
    _print_ahp_table(fl_names, w_fl, lam_fl, cr_fl, ci_fl, '跟驰车辆', FL_AHP_MATRIX)
    results['fl'] = {'names': fl_names, 'weights': w_fl}

    return results


def run_combined_analysis(locs, base_dir='E:/0little',
                           critic_results=None, ahp_results=None):
    """组合赋权 + 全方法对比表"""
    print('\n' + '=' * 72)
    print('  组合赋权')
    print('=' * 72)

    results = {}

    for data_key, metrics_cfg in [('lc', METRICS), ('fl', FOLLOWING_METRICS)]:
        if critic_results is None or critic_results.get(data_key) is None \
                or ahp_results is None or ahp_results.get(data_key) is None:
            continue

        # 构建 EWM 权重
        if data_key == 'lc':
            files = []
            for loc_key in locs:
                for side in ['left', 'right']:
                    fp = os.path.normpath(
                        os.path.join(base_dir, loc_key, f'traffic_{side}_change.csv'))
                    if os.path.exists(fp):
                        files.append(fp)
        else:
            files = []
            for loc_key in locs:
                fp = os.path.normpath(
                    os.path.join(base_dir, loc_key, 'traffic_following_change.csv'))
                if os.path.exists(fp):
                    files.append(fp)

        if not files:
            continue

        X, names = build_activation_matrix(files, metrics_cfg)
        w_ewm_unc, _, _ = compute_ewm_weights(X)
        w_ewm_cond, _, _, _ = compute_conditional_ewm_weights(X)

        cr = critic_results[data_key]
        ar = ahp_results[data_key]
        w_ahp = ar['weights']

        w_dict = {'EWM': w_ewm_unc, 'CRITIC': cr['w_critic_unc'], 'AHP': w_ahp}
        combo = compute_combined_weights(w_dict)

        title = '变道车辆' if data_key == 'lc' else '跟驰车辆'
        _print_full_comparison(
            names, metrics_cfg,
            w_ewm_unc, w_ewm_cond,
            cr['w_critic_unc'], cr['w_critic_int'], cr['n_int'],
            w_ahp, combo['multiplicative'], combo['min_deviation'],
            title)

        results[data_key] = {
            'names': names, 'w_ahp': w_ahp,
            'w_multi': combo['multiplicative'],
            'w_min_dev': combo['min_deviation'],
        }

    return results


# ── 权重对比可视化 ──


def plot_weight_analysis(locs, base_dir='E:/0little'):
    """生成权重对比图（柱状图 + 相关系数热力图 + 权重差异热力图）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'weight_sensitivity')
    os.makedirs(out_dir, exist_ok=True)

    # ── 收集数据 ──
    all_data = {}
    for data_type, cfg, matrix in [
        ('lc', METRICS, LC_AHP_MATRIX),
        ('fl', FOLLOWING_METRICS, FL_AHP_MATRIX)
    ]:
        if data_type == 'lc':
            files = []
            for loc_key in locs:
                for side in ['left', 'right']:
                    fp = os.path.normpath(
                        os.path.join(base_dir, loc_key, f'traffic_{side}_change.csv'))
                    if os.path.exists(fp):
                        files.append(fp)
        else:
            files = []
            for loc_key in locs:
                fp = os.path.normpath(
                    os.path.join(base_dir, loc_key, 'traffic_following_change.csv'))
                if os.path.exists(fp):
                    files.append(fp)

        if not files:
            continue

        X, names = build_activation_matrix(files, cfg)
        w_exp = np.array([m['w'] for m in cfg])
        w_ewm, _, _ = compute_ewm_weights(X)
        w_critic, _, _, corr_m = compute_critic_weights(X)
        w_ahp, _, _, _ = compute_ahp_weights(matrix)
        combo = compute_combined_weights(
            {'EWM': w_ewm, 'CRITIC': w_critic, 'AHP': w_ahp})

        all_data[data_type] = {
            'names': names,
            'weights': {
                'Expert': w_exp, 'EWM': w_ewm, 'CRITIC': w_critic,
                'AHP': w_ahp,
                'Multiplicative': combo['multiplicative'],
                'MeanDev': combo['min_deviation'],
            },
            'corr_matrix': corr_m,
        }

    # ── 1. 分组柱状图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    method_order = ['Expert', 'EWM', 'CRITIC', 'AHP', 'Multiplicative', 'MeanDev']
    method_colors = ['#2c3e50', '#3498db', '#1abc9c', '#e67e22', '#9b59b6', '#e84393']

    for ax_idx, (data_type, sc_name) in enumerate(
            [('lc', 'Lane Change'), ('fl', 'Following')]):
        ax = axes[ax_idx]
        d = all_data.get(data_type)
        if d is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        names = d['names']
        weights = d['weights']
        x = np.arange(len(names))
        n_methods = len(method_order)
        bar_width = 0.12
        offsets = np.linspace(-bar_width * (n_methods - 1) / 2,
                               bar_width * (n_methods - 1) / 2, n_methods)

        for i, method in enumerate(method_order):
            vals = weights[method]
            bars = ax.bar(x + offsets[i], vals, bar_width,
                          color=method_colors[i], alpha=0.85,
                          label=method, edgecolor='white', linewidth=0.5)
            # 在柱子上方标注数值
            for j, v in enumerate(vals):
                ax.text(x[j] + offsets[i], v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=5.5, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Weight', fontsize=11)
        ax.set_title(f'{sc_name} — Weight Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 0.7)
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'weight_comparison_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

    # ── 2. 相关系数矩阵热力图 ──
    for data_type, sc_name in [('lc', 'Lane Change'), ('fl', 'Following')]:
        d = all_data.get(data_type)
        if d is None:
            continue

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        corr = d['corr_matrix']
        names = d['names']
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-0.2, vmax=1.0, aspect='equal')

        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f'{corr[i, j]:.4f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if abs(corr[i, j]) > 0.5 else 'black')

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(f'{sc_name} — Metric Correlation Matrix', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson's r")

        plt.tight_layout()
        path = os.path.join(out_dir, f'weight_corr_{data_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')

    # ── 3. 权重差异热力图（各方法 vs Expert） ──
    for data_type, sc_name in [('lc', 'Lane Change'), ('fl', 'Following')]:
        d = all_data.get(data_type)
        if d is None:
            continue

        names = d['names']
        weights = d['weights']
        w_exp = weights['Expert']

        diff_data = {}
        for method in method_order:
            if method == 'Expert':
                continue
            diff_data[method] = weights[method] - w_exp

        diff_df = np.array([diff_data[m] for m in method_order if m != 'Expert'])

        fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2 + 1),
                                         max(3.5, len(method_order) * 0.7)))
        vmax = max(abs(diff_df.min()), abs(diff_df.max()))
        im = ax.imshow(diff_df, cmap='RdYlBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

        for i in range(len(method_order) - 1):
            for j in range(len(names)):
                val = diff_df[i, j]
                ax.text(j, i, f'{val:+.3f}',
                        ha='center', va='center', fontsize=9,
                        color='white' if abs(val) > vmax * 0.6 else 'black')

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(method_order) - 1))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticklabels([m for m in method_order if m != 'Expert'], fontsize=8)
        ax.set_title(f'{sc_name} — Weight Difference vs Expert', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Δ Weight')

        plt.tight_layout()
        path = os.path.join(out_dir, f'weight_diff_{data_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')

    print(f'\n  所有权重对比图已保存至: {out_dir}')


if __name__ == '__main__':
    locs_test = {f'location{i}': f'E:/0little/location{i}' for i in range(1, 6)}

    # ── EWM 权重 ──
    run_ewm_analysis(locs_test)

    # ── CRITIC 权重 ──
    critic_results = run_critic_analysis(locs_test)

    # ── AHP 权重 ──
    ahp_results = run_ahp_analysis()

    # ── 组合赋权 + 全方法对比 ──
    run_combined_analysis(locs_test,
                           critic_results=critic_results,
                           ahp_results=ahp_results)

    # ── 权重对比图 ──
    plot_weight_analysis(locs_test)
