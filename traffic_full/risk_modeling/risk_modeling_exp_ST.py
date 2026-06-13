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
MODELS = {**OPTUNA_MODELS, **LSTM_MODEL}


def main():
    np.random.seed(SEED)
    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")

    X_ts, y_ts, meta_ts = (None, None, None)
    if LSTM_MODEL:
        X_ts, y_ts, meta_ts = load_time_series(LOCS, LOC_KEYS, sample_len=75)
        print(f"  LSTM 输入: {X_ts.shape}")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, use_smote=True,
                                 X_ts=X_ts, y_ts=y_ts, meta_ts=meta_ts)
    rr = evaluate_random_split(df, MODELS, use_smote=True,
                               X_ts=X_ts, y_ts=y_ts)
    run_shap_analysis(rr, MODELS, 'shap_exp_ST')
    plot_comparison(fm, rr, MODELS, 'exp 评分 + SMOTE + Optuna 模型性能对比', '09_exp_ST_comparison.png')
    plot_confusion(rr, MODELS, 'exp 评分 + SMOTE + Optuna 混淆矩阵 (随机划分)', '10_exp_ST_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, '09_exp_ST_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, '09_exp_ST_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, '09_exp_ST_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')

    # ── 保存模型参数与结果报告 ──
    try:
        report_path = os.path.join(OUT_DIR, '09_exp_ST_model_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  exp 评分 + SMOTE + Optuna — 模型训练报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"数据: 全部 5 个 location\n")
            f.write(f"样本: {len(df)} 辆, 特征: {len(feature_cols)} 维\n")
            f.write(f"标签分布: 高风险{n_high} / 中风险{n_mid} / 低风险{n_low}\n")
            f.write(f"SMOTE: 训练集过采样 (strategy='not majority')\n")
            f.write(f"Optuna: TPE Sampler, 50 trials/模型\n\n")

            f.write("-" * 70 + "\n")
            f.write("  各模型最优参数\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name in rr and name != 'LSTM':
                    model = rr[name]['model']
                    f.write(f"\n{name}:\n")
                    params = model.get_params()
                    # 过滤掉默认参数，只输出 Optuna 搜索范围的参数
                    key_params = [k for k in params if k not in ('random_state', 'n_jobs', 'verbose', 'early_stopping_rounds', 'objective', 'num_class', 'verbosity')]
                    for k in key_params:
                        f.write(f"  {k}: {params[k]}\n")
                elif name == 'LSTM' and name in rr:
                    lstm_model = rr[name]['model']
                    f.write(f"\nLSTM:\n")
                    f.write(f"  架构: {lstm_model.count_params():,} 参数\n")
                    for layer in lstm_model.layers:
                        conf = layer.get_config()
                        f.write(f"  Layer: {conf['name']}, units={conf.get('units', 'N/A')}, dropout={conf.get('rate', conf.get('dropout', 'N/A'))}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("  跨 Location 验证\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name not in fm:
                    continue
                d = fm[name]
                f.write(f"\n{name}:\n")
                for fi, loc in enumerate(LOC_KEYS):
                    f.write(f"  Fold {fi+1} ({loc}): Acc={d['acc'][fi]:.3f}, "
                            f"F1={d['f1'][fi]:.3f}, MacroF1={d['macro_f1'][fi]:.3f}\n")
                f.write(f"  均值: Acc={np.mean(d['acc']):.3f}±{np.std(d['acc']):.3f}, "
                        f"F1={np.mean(d['f1']):.3f}±{np.std(d['f1']):.3f}, "
                        f"MacroF1={np.mean(d['macro_f1']):.3f}±{np.std(d['macro_f1']):.3f}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("  随机 80/20 划分\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name not in rr:
                    continue
                d = rr[name]
                f.write(f"\n{name}: Acc={d['acc']:.3f}, F1={d['f1']:.3f}, MacroF1={d['macro_f1']:.3f}\n")

            # 最佳模型排序
            f.write("\n" + "-" * 70 + "\n")
            f.write("  排名 (随机划分 F1)\n")
            f.write("-" * 70 + "\n")
            rankings = sorted(
                [(n, r['f1']) for n, r in rr.items() if n != '_data'],
                key=lambda x: x[1], reverse=True)
            for rank, (name, f1) in enumerate(rankings, 1):
                f.write(f"  #{rank} {name}: F1={f1:.3f}\n")

        print(f'\n  [OK] 模型报告: {report_path}')
    except Exception as e:
        print(f'  [WARN] 报告保存失败: {e}')


if __name__ == '__main__':
    main()
