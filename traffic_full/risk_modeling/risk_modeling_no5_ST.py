"""
变道风险预测建模（排除 location5 + SMOTE + Optuna 超参数搜索）— exp 指数版
=== 配置 ===
- 数据范围: location1~4（排除高速）
- 标签: 基于 safety_scoring_exp 的连续风险分 + 分场景阈值
- SMOTE: 训练集过采样 (strategy='not majority')
- Optuna: XGBoost / RandomForest / MLP 各 50 轮搜索
- 模型: XGBoost / RandomForest / MLP（不含 LSTM）
- 评估: 4-Fold 跨 Location + 随机 80/20
"""
from risk_modeling_utils_exp import *
import risk_modeling_utils_exp as _ut
SCRIPT_DIR = os.path.join(_ut.OUT_DIR, 'risk_modeling_no5_ST')
os.makedirs(SCRIPT_DIR, exist_ok=True)
_ut.OUT_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
}
LOC_KEYS = ['location1', 'location2', 'location3', 'location4']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4']
MODELS = dict(OPTUNA_MODELS)  # Optuna 搜索版，不含 LSTM

N_TRIALS = 50


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")
    print(f"  SMOTE: 训练集过采样 | Optuna: XGBoost/RF/MLP × {N_TRIALS} 轮")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, use_smote=True)
    rr = evaluate_random_split(df, MODELS, use_smote=True)
    run_shap_analysis(rr, MODELS, 'shap_no5_ST')
    plot_comparison(fm, rr, MODELS, 'Excl. Loc5 + SMOTE + Optuna Model Performance', '15_no5_ST_comparison.png')
    plot_confusion(rr, MODELS, 'Excl. Loc5 + SMOTE + Optuna Confusion Matrix', '16_no5_ST_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, '15_no5_ST_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, '15_no5_ST_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, '15_no5_ST_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')

    # ── 保存模型参数与结果报告 ──
    try:
        report_path = os.path.join(OUT_DIR, '15_no5_ST_model_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  排除 Loc5 + SMOTE + Optuna — 模型训练报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"样本: {len(df)} 辆, 特征: {len(feature_cols)} 维\n")
            f.write(f"标签分布: 高风险{n_high} / 中风险{n_mid} / 低风险{n_low}\n")
            f.write(f"SMOTE: 训练集过采样 | Optuna: ×{N_TRIALS} 轮\n\n")

            f.write("-" * 70 + "\n")
            f.write("  各模型最优参数\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name in rr:
                    model = rr[name]['model']
                    f.write(f"\n{name}:\n")
                    params = model.get_params()
                    key_params = [k for k in params if k not in ('random_state', 'n_jobs', 'verbose', 'early_stopping_rounds', 'objective', 'num_class', 'verbosity')]
                    for k in key_params:
                        f.write(f"  {k}: {params[k]}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("  跨 Location 验证\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name not in fm:
                    continue
                d = fm[name]
                f.write(f"\n{name}:\n")
                for fi, loc in enumerate(LOC_KEYS):
                    f.write(f"  Fold {fi+1} ({loc}): Acc={d['acc'][fi]:.3f}, F1={d['f1'][fi]:.3f}, MacroF1={d['macro_f1'][fi]:.3f}\n")
                f.write(f"  均值: Acc={np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f}, F1={np.mean(d['f1']):.3f}±{np.std(d['f1']):.3f}, MacroF1={np.mean(d['macro_f1']):.3f}±{np.std(d['macro_f1']):.3f}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("  随机 80/20 划分\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name not in rr:
                    continue
                d = rr[name]
                f.write(f"\n{name}: Acc={d['acc']:.3f}, F1={d['f1']:.3f}, MacroF1={d['macro_f1']:.3f}\n")

            rankings = sorted([(n, r['f1']) for n, r in rr.items() if n != '_data'], key=lambda x: x[1], reverse=True)
            f.write("\n" + "-" * 70 + "\n")
            f.write("  排名 (随机划分 F1)\n")
            f.write("-" * 70 + "\n")
            for rank, (name, f1) in enumerate(rankings, 1):
                f.write(f"  #{rank} {name}: F1={f1:.3f}\n")
        print(f'\n  [OK] 模型报告: {report_path}')
    except Exception as e:
        print(f'  [WARN] 报告保存失败: {e}')


if __name__ == '__main__':
    main()
