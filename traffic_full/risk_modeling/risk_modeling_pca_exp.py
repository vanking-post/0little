"""
变道风险预测建模（exp 评分 + PCA-LSTM 降维版）
=== 配置 ===
- 评分函数: safety_scoring_exp（exp 版本风险评分）
- 数据范围: 全部 5 个 location
- 特殊处理: LSTM 时序特征经 PCA 降维（保留 0.95 方差）
- XGBoost/RF/MLP: 使用 52 维聚合特征，不受 PCA 影响
- 评估: 5-Fold 跨 Location + 随机 80/20
- 定位: 对比 exp 评分体系下时序降维对 LSTM 的影响
"""
from risk_modeling_utils_exp import *
from sklearn.decomposition import PCA

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}
LOC_KEYS = ['location1', 'location2', 'location3', 'location4', 'location5']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5']
MODELS = {**STANDARD_MODELS, **LSTM_MODEL}


def load_pca_time_series(locs, loc_keys, sample_len=75):
    """加载时序数据并做 PCA 降维（exp 版风险评分）"""
    X_list, y_list, meta_list = [], [], []
    all_frames, scaler = [], StandardScaler()

    for loc in loc_keys:
        for side in ['left', 'right']:
            fp = os.path.join(locs[loc], f'traffic_{side}_change.csv')
            if not os.path.exists(fp): continue
            df = pd.read_csv(fp)
            for (vid, src), grp in df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                vals = grp[TS_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).values
                if len(vals) == sample_len:
                    all_frames.append(vals)
                    v0 = v0_for_loc(loc, loc_keys)
                    s = risk_score(grp, v0_kmh=v0)
                    lbl = risk_label(s, 'lane_change')[0]
                    y_list.append(0 if lbl == '高风险' else 1 if lbl == '中风险' else 2)
                    meta_list.append({'location': loc, 'side': side, 'vid': int(vid)})

    flat = np.concatenate(all_frames, axis=0)
    flat_s = scaler.fit_transform(flat)
    pca = PCA(n_components=0.95, random_state=SEED)
    flat_pca = pca.fit_transform(flat_s)
    n_comp = pca.n_components_
    print(f'  PCA: {len(TS_FEATURES)} dim → {n_comp} dim (方差={pca.explained_variance_ratio_.sum():.3f})')

    # 绘制方差累计图（双 Y 轴）
    fig, ax1 = plt.subplots(figsize=(8, 5))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    x = range(1, n_comp + 1)

    # 左轴：单个方差柱状图
    bars = ax1.bar(x, pca.explained_variance_ratio_, alpha=0.7, label='单个方差', color='#3498db')
    ax1.set_xlabel('主成分', fontsize=12)
    ax1.set_ylabel('单个方差占比', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')

    # 右轴：累积方差光滑曲线
    ax2 = ax1.twinx()
    ax2.plot(x, cum_var, color='#e74c3c', linewidth=2.5, marker='o', markersize=4,
             label='累计方差', alpha=0.9)
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_ylabel('累计方差占比', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 1.05)

    # 图例合并
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')

    ax1.set_title(f'PCA 方差累计 (exp版, {len(TS_FEATURES)}→{n_comp}, {cum_var[-1]:.1%})',
                  fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pca_exp_variance.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  [OK] pca_exp_variance.png')

    for vals in all_frames:
        X_list.append(pca.transform(scaler.transform(vals)))

    X = np.array(X_list, dtype=np.float32)
    return X, np.array(y_list), meta_list, n_comp


def main():
    np.random.seed(SEED)
    print("  PCA-LSTM (exp 版): LSTM 时序特征 PCA 降维版本\n")

    df = load_and_engineer(LOCS, LOC_KEYS)
    feature_cols = get_feature_cols(df)
    n_high = (df['risk'] == 0).sum(); n_mid = (df['risk'] == 1).sum(); n_low = (df['risk'] == 2).sum()
    print(f"  样本: {len(df)} 辆, 特征: {len(feature_cols)} 维, "
          f"标签: 高风险{n_high} 中风险{n_mid} 低风险{n_low}")

    X_ts = y_ts = meta_ts = None
    if LSTM_MODEL:
        X_ts, y_ts, meta_ts, n_pc = load_pca_time_series(LOCS, LOC_KEYS, sample_len=75)
        print(f"  PCA-LSTM 输入: ({X_ts.shape[0]}, {X_ts.shape[1]}, {n_pc})")

    fm = evaluate_cross_location(df, MODELS, LOC_KEYS, LOC_LABELS, X_ts=X_ts, y_ts=y_ts, meta_ts=meta_ts)
    rr = evaluate_random_split(df, MODELS, X_ts=X_ts, y_ts=y_ts)
    run_shap_analysis(rr, MODELS, 'shap_pca_exp')
    plot_comparison(fm, rr, MODELS, 'exp 版 PCA-LSTM vs 聚合模型', 'pca_exp_model_comparison.png')
    plot_confusion(rr, MODELS, 'exp 版 PCA-LSTM 混淆矩阵 (随机划分)', 'pca_exp_confusion.png')
    print_summary(fm, rr, MODELS)

    # 雷达图
    try:
        from radar_chart import (plot_cross_location_radar,
                                 plot_random_split_radar,
                                 plot_xgboost_per_location_radar)
        stability = {name: 1 - min(np.std(fm[name]['f1']) / max(np.mean(fm[name]['f1']), 1e-6), 1.0)
                     for name in MODELS if name in fm}
        plot_cross_location_radar(fm, MODELS, 'pca_exp_cross_location_radar.png', out_dir=OUT_DIR)
        plot_random_split_radar(rr, MODELS, 'pca_exp_random_split_radar.png', out_dir=OUT_DIR, stability_vals=stability)
        plot_xgboost_per_location_radar(fm, LOC_KEYS, 'pca_exp_xgboost_radar.png', out_dir=OUT_DIR)
    except Exception as e:
        print(f'  [WARN] 雷达图生成失败: {e}')


if __name__ == '__main__':
    main()
