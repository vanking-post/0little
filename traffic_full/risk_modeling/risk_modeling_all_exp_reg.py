"""
变道风险预测建模（全量 5 个 location）— exp 连续回归版

=== 与 risk_modeling_no5_exp_reg.py 的区别 ===
- 数据范围: location1~5（含高速）
- 评估: 5-Fold 跨 Location + 随机 80/20
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from risk_modeling_utils_exp import load_and_engineer, get_feature_cols, SEED, OUT_DIR

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3_part1': 'E:/0little/location3_part1', 'location4_part1': 'E:/0little/location4_part1',
    'location5': 'E:/0little/location5',
}
LOC_KEYS = ['location1', 'location2', 'location3_part1', 'location4_part1', 'location5']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5']

# ==================== 回归训练函数 ====================
def train_xgb_reg(X_tr, y_tr, X_te, y_te):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=SEED, verbosity=0)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    return model, yp

def train_rf_reg(X_tr, y_tr, X_te, y_te):
    model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   random_state=SEED)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    return model, yp

def train_mlp_reg(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                          max_iter=500, random_state=SEED)
    model.fit(X_tr_s, y_tr)
    yp = model.predict(X_te_s)
    return model, yp

REGRESSORS = {
    'XGBoost': train_xgb_reg,
    'RandomForest': train_rf_reg,
    'MLP': train_mlp_reg,
}


def evaluate_cross_location_reg(df, loc_keys, loc_labels):
    """5-Fold 跨 Location 回归评估"""
    print(f"\n{'='*70}")
    print("  回归评估: 5-Fold 跨 Location 验证")
    print(f"{'='*70}")

    fc = get_feature_cols(df)
    results = {}

    for fi, test_loc in enumerate(loc_keys):
        train_mask = df['location'] != test_loc
        test_mask = df['location'] == test_loc
        X_tr = df[train_mask][fc].values
        y_tr = df[train_mask]['risk_score'].values
        X_te = df[test_mask][fc].values
        y_te = df[test_mask]['risk_score'].values

        print(f"\nFold {fi+1}: 测试={loc_labels[fi]} ({len(X_te)}辆), "
              f"训练={len(X_tr)}辆")

        for name, fn in REGRESSORS.items():
            model, yp = fn(X_tr, y_tr, X_te, y_te)
            mae = mean_absolute_error(y_te, yp)
            r2 = r2_score(y_te, yp)
            if name not in results:
                results[name] = {'mae': [], 'r2': []}
            results[name]['mae'].append(mae)
            results[name]['r2'].append(r2)
            print(f"  {name:15s}: MAE={mae:.4f}  R²={r2:.4f}")

    print(f"\n--- 跨 Location 汇总 (均值±标准差) ---")
    for name, r in results.items():
        mae_mu, mae_sd = np.mean(r['mae']), np.std(r['mae'])
        r2_mu, r2_sd = np.mean(r['r2']), np.std(r['r2'])
        print(f"  {name:15s}: MAE={mae_mu:.4f}±{mae_sd:.4f}  R²={r2_mu:.4f}±{r2_sd:.4f}")

    return results


def evaluate_random_split_reg(df):
    """随机 80/20 回归评估 + 散点图"""
    fc = get_feature_cols(df)
    X = df[fc].values
    y = df['risk_score'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    print(f"\n{'='*70}")
    print("  回归评估: 随机 80/20 划分")
    print(f"  训练: {len(X_tr)}辆, 测试: {len(X_te)}辆")
    print(f"  目标范围: {y.min():.3f}~{y.max():.3f}")
    print(f"{'='*70}")

    results = {}
    fig, axes = plt.subplots(1, len(REGRESSORS), figsize=(6 * len(REGRESSORS), 5))

    for idx, (name, fn) in enumerate(REGRESSORS.items()):
        model, yp = fn(X_tr, y_tr, X_te, y_te)
        mae = mean_absolute_error(y_te, yp)
        r2 = r2_score(y_te, yp)
        results[name] = {'mae': mae, 'r2': r2, 'y_true': y_te, 'y_pred': yp}

        print(f"\n  {name}:")
        print(f"    MAE = {mae:.4f}  (平均误差 {mae:.2f} 分)")
        print(f"    R²  = {r2:.4f}  (1=完美, 0=均值基线)")

        ax = axes[idx] if len(REGRESSORS) > 1 else axes
        ax.scatter(y_te, yp, alpha=0.5, s=20, c='#3498db', edgecolors='white', linewidth=0.3)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.6, label='理想线')
        ax.set_xlabel('真实风险分', fontsize=11)
        ax.set_ylabel('预测风险分', fontsize=11)
        ax.set_title(f'{name}\nMAE={mae:.4f}  R²={r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, '16_all_exp_reg_scatter.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {out}')

    # 汇总
    best = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n最佳模型: {best[0]} (MAE={best[1]['mae']:.4f}, R²={best[1]['r2']:.4f})")

    return results


def main():
    np.random.seed(SEED)
    print("  exp 连续回归版: 全量 5 个 location\n")

    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维")
    print(f"  risk_score 范围: {df['risk_score'].min():.3f} ~ {df['risk_score'].max():.3f}")

    fm = evaluate_cross_location_reg(df, LOC_KEYS, LOC_LABELS)
    rr = evaluate_random_split_reg(df)


if __name__ == '__main__':
    main()
