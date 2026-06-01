"""
变道风险预测建模（排除 location5 + SMOTE 过采样）— exp 指数版
=== 配置 ===
- 数据范围: location1~4（排除高速）
- 标签: 基于 safety_scoring_exp 的连续风险分 + 分场景阈值
- SMOTE: 训练集过采样 (strategy='not majority')
- 模型: XGBoost / RandomForest / MLP（不含 LSTM）
- 评估: 4-Fold 跨 Location + 随机 80/20
"""
from risk_modeling_utils_exp import *

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3_part1': 'E:/0little/location3_part1', 'location4_part1': 'E:/0little/location4_part1',
}
LOC_KEYS = ['location1', 'location2', 'location3_part1', 'location4_part1']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4']
MODELS = dict(STANDARD_MODELS)  # 不含 LSTM


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, use_smote=True)
    rr = evaluate_random_split(df, MODELS, use_smote=True)
    run_shap_analysis(rr, MODELS, 'shap_no5_exp_smote')
    plot_comparison(fm, rr, MODELS, '排除 Loc5 + SMOTE (exp版) 模型性能对比', '13_no5_exp_smote_comparison.png')
    plot_confusion(rr, MODELS, '排除 Loc5 + SMOTE (exp版) 混淆矩阵', '14_no5_exp_smote_confusion.png')
    print_summary(fm, rr, MODELS)


if __name__ == '__main__':
    main()
