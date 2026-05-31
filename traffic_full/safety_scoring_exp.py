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

B = 1.3  # 速度修正参数

# ---------- 指标配置 ----------
METRICS = [
    {'name': 'mTTC',         'col': 'mTTC',         'w': 0.25, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',          'col': 'Time_Headway',  'w': 0.20, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'PET',          'col': 'PET',           'w': 0.20, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
    {'name': 'F_ETTC',       'col': 'F_ETTC',        'w': 0.20, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'OL_PET',       'col': 'OL_PET',        'w': 0.15, 'k': 12.0,
     'valid': lambda v: ~np.isinf(v) & ~np.isnan(v) & (v > 0)},
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


def risk_label(score):
    """将风险分转为中文标签（阈值待定，暂用占位）"""
    if score >= 0.5:
        return '高风险', '#e74c3c'
    elif score >= 0.2:
        return '中风险', '#f39c12'
    return '低风险', '#27ae60'
