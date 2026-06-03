"""
变道风险预测建模（PCA-LSTM 降维版）
=== 配置 ===
- 数据范围: 全部 5 个 location
- 特殊处理: LSTM 时序特征经 PCA 降维（保留 0.95 方差）
- XGBoost/RF/MLP: 使用 52 维聚合特征，不受 PCA 影响
- 评估: 5-Fold 跨 Location + 随机 80/20
- 定位: 对比时序降维对 LSTM 的影响
"""
from risk_modeling_utils import *
from sklearn.decomposition import PCA

LOCS = {
    'location1': 'E:/0little/location1', 'location2': 'E:/0little/location2',
    'location3': 'E:/0little/location3', 'location4': 'E:/0little/location4',
    'location5': 'E:/0little/location5',
}
LOC_KEYS = ['location1', 'location2', 'location3', 'location4', 'location5']
LOC_LABELS = ['Loc1', 'Loc2', 'Loc3', 'Loc4', 'Loc5']
MODELS = {**STANDARD_MODELS, **LSTM_MODEL}


def load_pca_time_series(locs, loc_keys, sample_len=50):
    """加载时序数据并做 PCA 降维"""
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
                    y_list.append(overall_risk(grp, v0_kmh=v0))
                    meta_list.append({'location': loc, 'side': side, 'vid': int(vid)})

    flat = np.concatenate(all_frames, axis=0)
    flat_s = scaler.fit_transform(flat)
    pca = PCA(n_components=0.95, random_state=SEED)
    flat_pca = pca.fit_transform(flat_s)
    n_comp = pca.n_components_
    print(f'  PCA: {len(TS_FEATURES)} dim → {n_comp} dim (方差={pca.explained_variance_ratio_.sum():.3f})')

    # 绘制方差累计图
    fig, ax = plt.subplots(figsize=(8, 5))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ax.bar(range(1, n_comp + 1), pca.explained_variance_ratio_, alpha=0.7, label='单个方差')
    ax.step(range(1, n_comp + 1), cum_var, where='mid', color='#e74c3c', linewidth=2, label='累计方差')
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('主成分', fontsize=12); ax.set_ylabel('方差占比', fontsize=12)
    ax.set_title(f'PCA 方差累计 ({len(TS_FEATURES)}→{n_comp}, {cum_var[-1]:.1%})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pca_variance.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('  ✅ pca_variance.png')

    for vals in all_frames:
        X_list.append(pca.transform(scaler.transform(vals)))

    X = np.array(X_list, dtype=np.float32)
    return X, np.array(y_list), meta_list, n_comp


def main():
    np.random.seed(SEED)
    print("  PCA-LSTM: LSTM 时序特征 PCA 降维版本\n")

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
    run_shap_analysis(rr, MODELS, 'shap_pca')
    plot_comparison(fm, rr, MODELS, 'PCA-LSTM vs 聚合模型', 'pca_model_comparison.png')
    plot_confusion(rr, MODELS, 'PCA-LSTM 混淆矩阵 (随机划分)', 'pca_confusion.png')
    print_summary(fm, rr, MODELS)


if __name__ == '__main__':
    main()
