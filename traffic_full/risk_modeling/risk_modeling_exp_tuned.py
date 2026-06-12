"""
变道风险预测建模（exp 评分 + 排除 location5 + Optuna 超参调优）
=== 配置 ===
- 评分函数: safety_scoring_exp（exp 版本风险评分）
- 数据范围: location1~4（排除分合流区）
- SMOTE: 无
- Optuna: TPE Sampler, 50 trials/模型
- 模型: XGBoost / RandomForest / MLP（Optuna 寻优版本）
- 评估: 4-Fold 跨 Location + 随机 80/20
"""
from risk_modeling_utils_exp import *

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
}
LOC_KEYS = ['location1', 'location2', 'location3', 'location4']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4']
MODELS = dict(OPTUNA_MODELS)


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS)
    rr = evaluate_random_split(df, MODELS)
    run_shap_analysis(rr, MODELS, 'shap_exp_tuned')
    plot_comparison(fm, rr, MODELS, 'exp 评分 + Optuna 调优模型性能对比', '09_exp_tuned_comparison.png')
    plot_confusion(rr, MODELS, 'exp 评分 + Optuna 调优混淆矩阵 (随机划分)', '10_exp_tuned_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, '09_exp_tuned_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, '09_exp_tuned_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, '09_exp_tuned_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')


if __name__ == '__main__':
    main()
