"""
变道风险预测建模（exp 评分 + 全量 + SMOTE + Optuna 联合优化）
=== 配置 ===
- 评分函数: safety_scoring_exp（exp 版本风险评分）
- 数据范围: 全部 5 个 location
- SMOTE: 训练集过采样 (strategy='not majority')
- Optuna: TPE Sampler, 50 trials/模型
- 模型: XGBoost / RandomForest / MLP（Optuna 寻优版本）
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
MODELS = dict(OPTUNA_MODELS)


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, use_smote=True)
    rr = evaluate_random_split(df, MODELS, use_smote=True)
    run_shap_analysis(rr, MODELS, 'shap_exp_ST')
    plot_comparison(fm, rr, MODELS, 'exp 评分 + SMOTE + Optuna 模型性能对比', '09_exp_ST_comparison.png')
    plot_confusion(rr, MODELS, 'exp 评分 + SMOTE + Optuna 混淆矩阵 (随机划分)', '10_exp_ST_confusion.png')
    print_summary(fm, rr, MODELS)


if __name__ == '__main__':
    main()
