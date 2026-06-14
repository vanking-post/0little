"""
变道风险预测建模（全量基线）— exp 指数版
=== 配置 ===
- 数据范围: 全部 5 个 location
- 标签: 基于 safety_scoring_exp 的连续风险分 + 分场景阈值
- SMOTE: 无
- Optuna: 无
- 模型: XGBoost / RandomForest / MLP / LSTM
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
MODELS = {**STANDARD_MODELS, **LSTM_MODEL}

 
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

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, X_ts=X_ts, y_ts=y_ts, meta_ts=meta_ts)
    rr = evaluate_random_split(df, MODELS, X_ts=X_ts, y_ts=y_ts)
    run_shap_analysis(rr, MODELS, 'shap_all_exp')
    plot_comparison(fm, rr, MODELS, '全量数据 (exp版) 模型性能对比', '17_all_exp_comparison.png')
    plot_confusion(rr, MODELS, '全量数据 (exp版) 混淆矩阵', '18_all_exp_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, '17_all_exp_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, '17_all_exp_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, '17_all_exp_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')

    # ── 保存模型参数与结果报告 ──
    try:
        report_path = os.path.join(OUT_DIR, '17_all_exp_model_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  全量数据 (exp版) — 模型训练报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"样本: {len(df)} 辆, 特征: {len(feature_cols)} 维\n")
            f.write(f"标签分布: 高风险{n_high} / 中风险{n_mid} / 低风险{n_low}\n\n")

            f.write("-" * 70 + "\n")
            f.write("  各模型参数\n")
            f.write("-" * 70 + "\n")
            for name in MODELS:
                if name in rr and name != 'LSTM':
                    model = rr[name]['model']
                    f.write(f"\n{name}:\n")
                    params = model.get_params()
                    key_params = [k for k in params if k not in ('random_state', 'n_jobs', 'verbose', 'early_stopping_rounds', 'objective', 'num_class', 'verbosity')]
                    for k in key_params:
                        f.write(f"  {k}: {params[k]}\n")
                elif name == 'LSTM' and name in rr:
                    lstm_model = rr[name]['model']
                    f.write(f"\nLSTM: {lstm_model.count_params():,} 参数\n")
                    for layer in lstm_model.layers:
                        conf = layer.get_config()
                        f.write(f"  {conf['name']}: units={conf.get('units', 'N/A')}\n")

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
