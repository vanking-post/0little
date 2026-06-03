"""
安全评价评分 — 连续风险版 (exp(-x/k) + 加权)

与 safety_scoring.py 的区别:
  - 不再使用类别打分 (dangerous=2, cautious=1)，改用 exp(-值/k) 连续映射
  - 无对应车辆时贡献=0，不参与权重归一化
  - 暂不设定高风险/中风险/低风险阈值，仅输出连续分

用法:
    from safety_scoring_exp import risk_score, risk_label
"""

import numpy as np
import os
import pandas as pd

B = 1.3  # 速度修正参数

# ---------- 风险等级阈值 ----------
# 中风险 ≥ thresh_mid, 高风险 ≥ thresh_high
THRESH_LANE_CHANGE = {'mid': 0.40, 'high': 0.60}  # 变道车辆
THRESH_FOLLOWING   = {'mid': 0.20, 'high': 0.35}  # 跟驰车辆

# ---------- 指标配置（变道车辆） ----------
METRICS = [
    {'name': 'mTTC',         'col': 'mTTC',         'w': 0.247, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',          'col': 'Time_Headway',  'w': 0.083, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'PET',          'col': 'PET',           'w': 0.131, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
    {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.356, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.183, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
]

# ---------- 指标配置（跟驰车辆） ----------
# 仅前车(mTTC/THW) + 后车(B_mTTC)，权重归一化到和为 1
FOLLOWING_METRICS = [
    {'name': 'mTTC',    'col': 'mTTC',    'w': 0.463, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',     'col': 'Time_Headway', 'w': 0.114, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'B_mTTC',  'col': 'B_mTTC',  'w': 0.423, 'k': 12.0,
     'valid': lambda v: (v > 0)},
]


def _frame_contrib(val, k):
    """单帧的风险贡献: exp(-val/k)，val 越小、贡献越接近 1"""
    return np.exp(-val / k)


def risk_score(grp, v0_kmh=100):
    """计算一辆车的连续风险分 (0~1+，越高越危险)

    参数:
        grp: 单辆车的 50 帧 DataFrame
        v0_kmh: 道路基准速度 (km/h)，location1-4→100, location5→80

    返回:
        float: 风险分（未归一化上限，通常 0~1 之间）
    """
    score = 0.0

    for m in METRICS:
        if m['col'] not in grp.columns:
            continue

        vals = grp[m['col']].values.astype(float)
        valid = m['valid'](vals)

        if not valid.any():
            # 无对应车辆 → 该指标不贡献
            continue

        # 取 50 帧中 exp(-val/k) 的最大值（等同 worst_cat 的取最差思路）
        contrib = np.nanmax(_frame_contrib(vals[valid], m['k']))
        score += m['w'] * contrib

    # 速度修正系数 K
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        k_speed = 1.0
    else:
        v0 = v0_kmh / 3.6
        k_speed = 1.0 if v85 <= v0 else (v85 / v0) ** B

    return score * k_speed


def following_risk_score(grp, v0_kmh=100):
    """计算跟驰车辆的连续风险分（与变道车辆同尺度，可直接对比）

    仅用前车(mTTC/THW) + 后车(B_mTTC)三项，
    权重已归一化到和为 1，乘速度修正后与 risk_score 对齐。
    """
    score = 0.0

    for m in FOLLOWING_METRICS:
        if m['col'] not in grp.columns:
            continue

        vals = grp[m['col']].values.astype(float)
        valid = m['valid'](vals)

        if not valid.any():
            continue

        contrib = np.nanmax(_frame_contrib(vals[valid], m['k']))
        score += m['w'] * contrib

    # 速度修正系数 K（同 risk_score）
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        k_speed = 1.0
    else:
        v0 = v0_kmh / 3.6
        k_speed = 1.0 if v85 <= v0 else (v85 / v0) ** B

    return score * k_speed


# ==================== 熵权法 (EWM) 客观权重计算 ====================
def compute_ewm_weights(X):
    """熵权法计算客观权重

    参数:
        X: np.ndarray, shape (n_samples, n_metrics)
           风险激活值矩阵，X_{ij} ∈ [0, 1]，
           表示第 i 辆车在第 j 个指标上的最大风险激活值

    返回:
        weights:  np.ndarray (n_metrics,)  客观权重，求和为 1
        entropy:  np.ndarray (n_metrics,)  各指标信息熵 E_j
        diversity: np.ndarray (n_metrics,) 各指标差异系数 D_j = 1 - E_j
    """
    n = X.shape[0]          # 车辆数
    m = X.shape[1]          # 指标数

    if n == 0 or m == 0:
        raise ValueError(f'X 不能为空: shape={X.shape}')

    # ── Step 1: 计算特征比重 P_{ij} ──
    # P_{ij} = X_{ij} / Σ_i X_{ij}，Σ_i P_{ij} = 1
    col_sums = X.sum(axis=0)
    P = np.zeros_like(X, dtype=float)
    for j in range(m):
        if col_sums[j] > 0:
            P[:, j] = X[:, j] / col_sums[j]
        # 若第 j 列全为 0（所有车辆均无该指标对应的邻车），
        # 则 P_{·j} 保持全 0 → 后续 0·ln(0) = 0

    # ── Step 2: 计算信息熵 E_j ──
    # E_j = -1/ln(n) · Σ_i P_{ij} · ln(P_{ij})
    # 约定 0·ln(0) = 0（numpy 自动处理，但需显式掩码避免 -inf）
    entropy = np.zeros(m)
    ln_n = np.log(n) if n > 1 else 1.0   # n=1 时退化为均匀权重
    for j in range(m):
        pj = P[:, j]
        pj_pos = pj[pj > 0]              # 只取 >0 的概率
        if len(pj_pos) > 0:
            entropy[j] = -np.sum(pj_pos * np.log(pj_pos)) / ln_n
        # 若某列全为 0：entropy[j] = 0（无信息）

    # ── Step 3: 差异系数与权重 ──
    # D_j = 1 - E_j：信息越集中（熵越小），差异越大，权重越大
    diversity = 1.0 - entropy           # D_j ∈ [0, 1]

    if diversity.sum() > 0:
        weights = diversity / diversity.sum()
    else:
        weights = np.ones(m) / m        # 退化为均匀权重

    return weights, entropy, diversity


def compute_conditional_ewm_weights(X):
    """条件熵权法：每个指标仅在激活值>0的车辆子集上计算

    消除"无对应邻车→激活值=0"的结构性零值对熵的干扰。

    对每个指标 j，定义子集 S_j = {i | X_{ij} > 0}：
      - |S_j| = 0：E_j = 0, D_j = 0（该指标无法评估任何车辆）
      - |S_j| > 0：在 S_j 上计算信息熵，用 ln(|S_j|) 归一化

    参数:
        X: np.ndarray, shape (n_samples, n_metrics)

    返回:
        weights:       np.ndarray (m,)  客观权重，求和为 1
        entropy:       np.ndarray (m,)  条件信息熵 E_j
        diversity:     np.ndarray (m,)  差异系数 D_j = 1 - E_j
        subset_sizes:  list[int]        |S_j|，各指标的有效样本数
    """
    n, m = X.shape
    if n == 0 or m == 0:
        raise ValueError(f'X 不能为空: shape={X.shape}')

    entropy = np.zeros(m)
    subset_sizes = np.zeros(m, dtype=int)

    for j in range(m):
        col = X[:, j]
        mask = col > 0              # 激活值 >0 → 对应邻车存在
        n_j = mask.sum()
        subset_sizes[j] = n_j

        if n_j == 0:
            entropy[j] = 0.0        # 所有车辆均无此邻车 → 无信息
            continue

        # ── P_{ij} = X_{ij} / Σ_{k∈S_j} X_{kj} ──
        col_sum = col[mask].sum()
        P = col[mask] / col_sum if col_sum > 0 else np.zeros(n_j)

        # ── E_j = -1/ln(n_j) · Σ P · ln(P) ──
        P_pos = P[P > 0]
        ln_nj = np.log(n_j) if n_j > 1 else 1.0
        if len(P_pos) > 0:
            entropy[j] = -np.sum(P_pos * np.log(P_pos)) / ln_nj
        # n_j=1 时 entropy=0（单样本无分布信息）

    # ── 差异系数与权重 ──
    diversity = 1.0 - entropy
    if diversity.sum() > 0:
        weights = diversity / diversity.sum()
    else:
        weights = np.ones(m) / m

    return weights, entropy, diversity, list(subset_sizes)


def _metric_activation(grp, col, k, valid_fn):
    """计算单辆车在单个指标上的最大风险激活值

    与 risk_score() 中的逻辑完全一致:
        X_{ij} = max_{帧∈有效} exp(-val / k)
    若无有效值（无邻车或数据缺失），返回 0。
    """
    if col not in grp.columns:
        return 0.0
    vals = grp[col].values.astype(float)
    valid = valid_fn(vals)
    if not valid.any():
        return 0.0
    return float(np.nanmax(_frame_contrib(vals[valid], k)))


def build_activation_matrix(file_paths, metrics_config):
    """从 CSV 文件列表构建风险激活矩阵 X

    遍历所有文件中的所有车辆，对每个指标计算最大风险激活值。

    参数:
        file_paths:      list[str]  CSV 文件路径列表
        metrics_config:  list[dict] 指标配置
                         (METRICS 或 FOLLOWING_METRICS)

    返回:
        X:      np.ndarray (n_vehicles, n_metrics)
        names:  list[str]  对应的指标名称列表
    """
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

    # ── 无条件 EWM ──
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

    # ── 条件 EWM ──
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

    # ── 简洁摘要 ──
    print(f'\n  结论: 无条件 ρ={rho_unc:.4f}, 条件 ρ={rho_cond:.4f}')
    return rho_unc, rho_cond


def run_ewm_analysis(locs, base_dir='E:/0little', loc_v0=None):
    """对变道和跟驰车辆分别运行无条件+条件熵权法并打印对比结果

    参数:
        locs:     dict   {loc_key: data_dir}
        base_dir: str    项目根目录
        loc_v0:   dict   各 location 的 V0（仅用于信息输出，不影响权重计算）
    """
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

        # 无条件 EWM
        w_unc, e_unc, d_unc = compute_ewm_weights(X_lc)
        # 条件 EWM
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

        # 无条件 EWM
        w_unc, e_unc, d_unc = compute_ewm_weights(X_fl)
        # 条件 EWM
        w_cond, e_cond, d_cond, sizes = compute_conditional_ewm_weights(X_fl)

        _print_ewm_table(fl_names, FOLLOWING_METRICS,
                         w_unc, e_unc, d_unc,
                         w_cond, e_cond, d_cond, sizes,
                         '跟驰车辆', X_fl.shape[0])
    else:
        print('\n  [WARN] 未找到跟驰数据文件')


def risk_label(score, scenario='lane_change'):
    """将风险分转为中文标签

    参数:
        score: 连续风险分
        scenario: 'lane_change' 或 'following'，不同场景阈值不同

    阈值由题头 THRESH_LANE_CHANGE / THRESH_FOLLOWING 全局参数控制。
    """
    t = THRESH_LANE_CHANGE if scenario == 'lane_change' else THRESH_FOLLOWING

    if score >= t['high']:
        return '高风险', '#e74c3c'
    elif score >= t['mid']:
        return '中风险', '#f39c12'
    return '低风险', '#27ae60'
