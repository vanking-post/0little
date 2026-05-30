"""
安全评价评分共享模块 — 封装打分逻辑，供各建模/可视化脚本统一调用

用法:
    from safety_scoring import worst_cat, overall_risk, risk_label
"""

import numpy as np
threshold_mid = 1.5
threshold_high = 3.0
B = 1.3 # 速度修正参数

def worst_cat(series):
    """扫描全部帧，返回该指标的最差类别"""
    if (series == 'dangerous').any():
        return 'dangerous'
    if (series == 'cautious').any():
        return 'cautious'
    return 'safe'


def overall_risk(grp, v0_kmh=100):
    """综合风险 Weighted-Scoring + 速度修正
    核心(mTTC, THW): dangerous=2, cautious=1
    辅助(PET, F_ETTC, OL_PET): dangerous=1, cautious=0.5
    基分 = raw_score × K，K=1 或 (V85/V0)^B，V0 按道路类型传入
    修正后 ≥3.0 → 高风险(0),  ≥1.5 → 中风险(1),  <1.5 → 低风险(2)
    """
    f_ettc_w = worst_cat(grp['F_ETTC_cat']) if 'F_ETTC_cat' in grp.columns else 'safe'
    thw_w    = worst_cat(grp['THW_cat'])
    pet_w    = worst_cat(grp['PET_cat'])
    mttc_w   = worst_cat(grp['mTTC_cat'])
    ol_pet_w = worst_cat(grp['OL_PET_cat']) if 'OL_PET' in grp.columns else 'safe'

    raw = 0.0
    for cat, is_core in [(mttc_w, True), (thw_w, True),
                         (pet_w, False), (f_ettc_w, False), (ol_pet_w, False)]:
        if cat == 'dangerous':
            raw += 2 if is_core else 1
        elif cat == 'cautious':
            raw += 1 if is_core else 0.5

    # 速度修正系数 K：取 50 帧 Velocity 的 85% 分位值
    v85 = np.nanpercentile(
        grp['Velocity'].replace([np.inf, -np.inf], np.nan), 85)
    if np.isnan(v85):
        k = 1.0
    else:
        v0 = v0_kmh / 3.6
        k = 1.0 if v85 <= v0 else (v85 / v0) ** B

    score = raw * k

    if score >= threshold_high:
        return 0
    elif score >= threshold_mid:
        return 1
    return 2


def risk_label(score):
    """将 0/1/2 数值标签转为中文和颜色"""
    mapping = {
        0: ('高风险', '#e74c3c'),
        1: ('中风险', '#f39c12'),
        2: ('低风险', '#27ae60'),
    }
    return mapping.get(score, ('未知', '#95a5a6'))
