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

# ---------- 指标配置（变道车辆） ----------
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

# ---------- 指标配置（跟驰车辆） ----------
# 仅前车(mTTC/THW) + 后车(B_mTTC)，权重归一化到和为 1
FOLLOWING_METRICS = [
    {'name': 'mTTC',    'col': 'mTTC',    'w': 0.4, 'k': 12.0,
     'valid': lambda v: (v > 0)},
    {'name': 'THW',     'col': 'Time_Headway', 'w': 0.3, 'k': 6.0,
     'valid': lambda v: (v > 0)},
    {'name': 'B_mTTC',  'col': 'B_mTTC',  'w': 0.3, 'k': 12.0,
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


def risk_label(score, scenario='lane_change'):
    """将风险分转为中文标签

    参数:
        score: 连续风险分
        scenario: 'lane_change' 或 'following'，不同场景阈值不同

    阈值依据各场景 P50(中风险下限) / P85(高风险下限) 设定:
      - 变道车辆: P50=0.40, P85=0.61
      - 跟驰车辆: P50=0.16, P85=0.30
    """
    if scenario == 'lane_change':
        thresh_mid, thresh_high = 0.40, 0.60
    else:
        thresh_mid, thresh_high = 0.16, 0.30

    if score >= thresh_high:
        return '高风险', '#e74c3c'
    elif score >= thresh_mid:
        return '中风险', '#f39c12'
    return '低风险', '#27ae60'
