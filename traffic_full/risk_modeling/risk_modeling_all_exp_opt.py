"""
变道风险预测建模（全量 + Optuna 超参数搜索）— exp 指数版
=== 配置 ===
- 数据范围: 全部 5 个 location
- 标签: 基于 safety_scoring_exp 的连续风险分 + 分场景阈值
- Optuna: XGBoost / RandomForest / MLP 各 50 轮搜索
- LSTM: 标准配置（无 Optuna）
- 评估: 5-Fold 跨 Location + 随机 80/20
"""
from risk_modeling_utils_exp import *

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}
LOC_KEYS = ['location1', 'location2', 'location3', 'location4', 'location5']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5']
MODELS = {**OPTUNA_MODELS, **LSTM_MODEL}

N_TRIALS = 50


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")
    print(f"  Optuna 搜索: XGBoost / RandomForest / MLP × {N_TRIALS} 轮")

    X_ts, y_ts, meta_ts = (None, None, None)
    if LSTM_MODEL:
        X_ts, y_ts, meta_ts = load_time_series(LOCS, LOC_KEYS, sample_len=75)
        print(f"  LSTM 输入: {X_ts.shape}")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, X_ts=X_ts, y_ts=y_ts, meta_ts=meta_ts)
    rr = evaluate_random_split(df, MODELS, X_ts=X_ts, y_ts=y_ts)
    run_shap_analysis(rr, MODELS, 'shap_all_exp_opt')
    plot_comparison(fm, rr, MODELS, '全量数据 (Optuna搜索) 模型性能对比', '17_all_exp_opt_comparison.png')
    plot_confusion(rr, MODELS, '全量数据 (Optuna搜索) 混淆矩阵', '18_all_exp_opt_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, '17_all_exp_opt_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, '17_all_exp_opt_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, '17_all_exp_opt_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')


if __name__ == '__main__':
    main()
