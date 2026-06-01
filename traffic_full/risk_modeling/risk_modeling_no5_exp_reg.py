"""
变道风险预测建模（排除 location5）— exp 连续回归版

=== 与分类版的区别 ===
- 目标: risk_score（连续值 0~1），而非 0/1/2 分类标签
- 模型: XGBRegressor / RandomForestRegressor / MLPRegressor
- 评估: MAE / R² / 预测值 vs 真实值散点图
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


def evaluate_regression(df, models, test_size=0.2):
    """回归模型评估：随机 80/20 划分，指标为 MAE / R²"""
    fc = get_feature_cols(df)
    X = df[fc].values
    y = df['risk_score'].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED)

    print(f"\n{'='*60}")
    print("  回归评估: 随机 80/20 划分")
    print(f"  训练: {len(X_tr)}辆, 测试: {len(X_te)}辆")
    print(f"  目标: risk_score (连续值, 实际范围 {y.min():.3f}~{y.max():.3f})")
    print(f"{'='*60}")

    results = {}
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))

    for idx, (name, fn) in enumerate(models.items()):
        model, yp = fn(X_tr, y_tr, X_te, y_te)
        mae = mean_absolute_error(y_te, yp)
        r2 = r2_score(y_te, yp)

        results[name] = {'model': model, 'y_true': y_te, 'y_pred': yp,
                         'mae': mae, 'r2': r2}
        print(f"\n  {name}:")
        print(f"    MAE = {mae:.4f}  (平均误差 {mae:.2f} 分)")
        print(f"    R²  = {r2:.4f}  (1=完美, 0=均值基线)")

        # 散点图：预测 vs 真实
        ax = axes[idx] if len(models) > 1 else axes
        ax.scatter(y_te, yp, alpha=0.5, s=20, c='#3498db', edgecolors='white', linewidth=0.3)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.6, label='理想 (y_pred = y_true)')
        ax.set_xlabel('真实风险分', fontsize=11)
        ax.set_ylabel('预测风险分', fontsize=11)
        ax.set_title(f'{name}\nMAE={mae:.4f}  R²={r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, '15_no5_exp_reg_scatter.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {out}')

    # 汇总
    print(f"\n{'='*60}")
    print("  回归汇总")
    print(f"{'='*60}")
    best = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"  最佳模型: {best[0]} (MAE={best[1]['mae']:.4f}, R²={best[1]['r2']:.4f})")
    for name, r in results.items():
        print(f"  {name:15s}: MAE={r['mae']:.4f}  R²={r['r2']:.4f}")

    return results


def main():
    np.random.seed(SEED)
    LOCS = {
        'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
        'location3_part1': 'E:/0little/location3_part1', 'location4_part1': 'E:/0little/location4_part1',
    }
    LOC_KEYS = ['location1', 'location2', 'location3_part1', 'location4_part1']

    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维")
    print(f"  risk_score 范围: {df['risk_score'].min():.3f} ~ {df['risk_score'].max():.3f}")
    print(f"  risk_score 均值: {df['risk_score'].mean():.3f}")

    evaluate_regression(df, REGRESSORS)


if __name__ == '__main__':
    main()
