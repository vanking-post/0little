"""
变道风险预测 — 用 loc1-4 训练，在 loc5 验证（exp 指数版）
=== 配置 ===
- 训练: location1~4（城市快速路/主干道）
- 测试: location5（匝道/高速，V0=80km/h）
- 模型: XGBoost / RandomForest / MLP / LSTM
- 评估: 训练集上训练 → 在 loc5 上做最终验证
"""
from risk_modeling_utils_exp import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==================== 配置 ====================
TRAIN_LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
}
TRAIN_KEYS = ['location1', 'location2', 'location3', 'location4']

TEST_LOCS = {
    'location5': 'E:/0little/location5',
}
TEST_KEYS = ['location5']

MODELS = {**STANDARD_MODELS, **LSTM_MODEL}
SAVE_PREFIX = 'train4_test5'


def main():
    np.random.seed(SEED)

    # ── 1. 加载数据 ──
    print("=" * 70)
    print("  加载训练数据 (loc1-4) & 测试数据 (loc5)")
    print("=" * 70)

    df_train = load_and_engineer(TRAIN_LOCS, TRAIN_KEYS)
    df_test = load_and_engineer(TEST_LOCS, TEST_KEYS, default_v0=100, loc5_v0=80)

    fc = get_feature_cols(df_train)
    # 确保 test 和 train 的特征列对齐
    test_fc = [c for c in fc if c in df_test.columns]
    missing = set(fc) - set(df_test.columns)
    if missing:
        print(f"  [WARN] 测试数据缺少特征: {missing}，填充 0")
        for c in missing:
            df_test[c] = 0.0

    X_train_all = df_train[fc].values
    y_train_all = df_train['risk'].values
    X_test = df_test[test_fc].values
    y_test = df_test['risk'].values

    # 统计
    for name, df_, X_, y_ in [('训练 (loc1-4)', df_train, X_train_all, y_train_all),
                               ('测试 (loc5)',   df_test,  X_test,       y_test)]:
        n_h = (y_ == 0).sum(); n_m = (y_ == 1).sum(); n_l = (y_ == 2).sum()
        print(f"  {name}: {len(df_)} 辆, 特征 {len(fc)} 维, "
              f"高风险{n_h} 中风险{n_m} 低风险{n_l}")

    # ── 2. 加载 LSTM 时序数据 ──
    X_ts_train = X_ts_test = y_ts_train = y_ts_test = None
    if LSTM_MODEL:
        X_ts_train, y_ts_train, meta_tr = load_time_series(TRAIN_LOCS, TRAIN_KEYS, sample_len=75)
        X_ts_test, y_ts_test, meta_te = load_time_series(TEST_LOCS, TEST_KEYS, sample_len=75)
        print(f"  LSTM 训练: {X_ts_train.shape}  测试: {X_ts_test.shape}")

    # ── 3. 训练与评估 ──
    print("\n" + "=" * 70)
    print("  训练 → loc5 验证")
    print("=" * 70)

    results = {}
    for name, fn in MODELS.items():
        if name == 'LSTM' and X_ts_train is not None:
            model, y_pred = fn(X_ts_train, y_ts_train, X_ts_test, y_ts_test)
            y_true_used = y_ts_test
        else:
            model, y_pred = fn(X_train_all, y_train_all, X_test, y_test)
            y_true_used = y_test

        acc = accuracy_score(y_true_used, y_pred)
        f1_w = f1_score(y_true_used, y_pred, average='weighted')
        f1_m = f1_score(y_true_used, y_pred, average='macro')
        results[name] = {'acc': acc, 'f1': f1_w, 'macro_f1': f1_m,
                         'y_true': y_true_used, 'y_pred': y_pred, 'model': model}

        print(f"\n  {name}: Acc={acc:.3f}, F1={f1_w:.3f}, MacroF1={f1_m:.3f}")
        print(classification_report(y_true_used, y_pred,
                                    target_names=['高风险', '中风险', '低风险'],
                                    digits=3))

    # ── 4. 汇总排名 ──
    print("\n" + "=" * 70)
    print("  loc5 验证汇总")
    print("=" * 70)
    ranking = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for name, r in ranking:
        print(f"  {name:<15s}  Acc={r['acc']:.3f}  F1={r['f1']:.3f}  MacroF1={r['macro_f1']:.3f}")
    print(f"\n  最佳模型 (F1): {ranking[0][0]} ({ranking[0][1]['f1']:.3f})")

    # ── 5. 混淆矩阵 ──
    plot_confusion(results, MODELS, f'loc1-4 训练 → loc5 验证', f'{SAVE_PREFIX}_confusion.png')
    print(f"\n  [OK] {SAVE_PREFIX}_confusion.png")

    # ── 6. SHAP 分析 ──
    shap_data = {'X_test': X_test, 'feature_names': fc}
    random_results = {'_data': shap_data}
    for name in results:
        random_results[name] = results[name]
    run_shap_analysis(random_results, MODELS, SAVE_PREFIX)


if __name__ == '__main__':
    main()
