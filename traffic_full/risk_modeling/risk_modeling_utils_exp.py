"""
变道风险预测建模 — 共享工具模块 (exp 指数版)

与 risk_modeling_utils.py 的区别:
  - 标签生成使用 safety_scoring_exp 的 risk_score + risk_label (连续打分)
  - 其余模型训练、评估、SHAP、可视化完全一致

各 risk_modeling_*_exp.py 只需配置 + 串流程，不再各自维护 ~500 行。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, recall_score)
from matplotlib.patches import Patch
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from safety_scoring_exp import risk_score, risk_label

# ==================== TF / LSTM 可选 ====================
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ==================== SHAP（可选） ====================
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
OUT_DIR = 'E:/0little/traffic_full/analysis'
TS_FEATURES = [
    'Velocity', 'long_Vel', 'lat_Vel', 'long_Acc', 'lat_Acc',
    'Following_dist', 'B_Dist', 'LB_Dist', 'RB_Dist', 'LF_Dist', 'RF_Dist',
    'TTC', 'Time_Headway', 'Lateral_Jerk', 'Longitudinal_Jerk'
]

os.makedirs(OUT_DIR, exist_ok=True)


# ==================== 辅助函数 ====================
def get_feature_cols(df):
    exclude = ['vehicle_id', 'location', 'side', 'source', 'risk', 'risk_score']
    return [c for c in df.columns if c not in exclude]


def get_class_weights(y):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {cls: total / (len(unique) * cnt) for cls, cnt in zip(unique, counts)}


def v0_for_loc(loc_key, loc_keys, default_v0=100, loc5_v0=80):
    """判断该 location 使用的 V0 速度修正基准"""
    return loc5_v0 if loc_key == 'location5' and 'location5' in loc_keys else default_v0


# ==================== 数据加载 ====================
def load_and_engineer(locs, loc_keys, ts_features=TS_FEATURES,
                      default_v0=100, loc5_v0=80):
    """从各 location 加载变道数据 → 聚合特征 (每车一行)"""
    all_v = []
    for loc_key in loc_keys:
        for side in ['left', 'right']:
            fp = os.path.join(locs[loc_key], f'traffic_{side}_change.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            for (vid, src), grp in df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                row = {'vehicle_id': int(vid), 'location': loc_key, 'side': side, 'source': src}
                for feat in ts_features:
                    vals = grp[feat].replace([np.inf, -np.inf], np.nan).values
                    row[f'{feat}_mean'] = np.nanmean(vals)
                    row[f'{feat}_std'] = np.nanstd(vals)
                    row[f'{feat}_min'] = np.nanmin(vals)
                    row[f'{feat}_max'] = np.nanmax(vals)
                v0 = v0_for_loc(loc_key, loc_keys, default_v0, loc5_v0)
                score = risk_score(grp, v0_kmh=v0)
                lbl = risk_label(score, 'lane_change')[0]
                row['risk'] = 0 if lbl == '高风险' else 1 if lbl == '中风险' else 2
                row['risk_score'] = round(score, 4)
                all_v.append(row)
    return pd.DataFrame(all_v).fillna(0)


def load_time_series(locs, loc_keys, ts_features=TS_FEATURES,
                     default_v0=100, loc5_v0=80, sample_len=50):
    """加载时序数据 (n_samples, seq_len, n_features) 用于 LSTM"""
    X_list, y_list, meta_list = [], [], []
    scaler = StandardScaler()
    all_vals = []
    for loc_key in loc_keys:
        for side in ['left', 'right']:
            fp = os.path.join(locs[loc_key], f'traffic_{side}_change.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            for (vid, src), grp in df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                vals = grp[ts_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
                if len(vals) == sample_len:
                    all_vals.append(vals)
    all_vals = np.concatenate(all_vals, axis=0)
    scaler.fit(all_vals)

    for loc_key in loc_keys:
        for side in ['left', 'right']:
            fp = os.path.join(locs[loc_key], f'traffic_{side}_change.csv')
            if not os.path.exists(fp):
                continue
            df = pd.read_csv(fp)
            for (vid, src), grp in df.groupby(['ID', 'Source']):
                grp = grp.sort_values('Frame')
                vals = grp[ts_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
                if len(vals) != sample_len:
                    continue
                vals_s = scaler.transform(vals)
                v0 = v0_for_loc(loc_key, loc_keys, default_v0, loc5_v0)
                score = risk_score(grp, v0_kmh=v0)
                lbl = risk_label(score, 'lane_change')[0]
                risk = 0 if lbl == '高风险' else 1 if lbl == '中风险' else 2
                X_list.append(vals_s)
                y_list.append(risk)
                meta_list.append({'location': loc_key, 'side': side, 'vid': int(vid)})
    return np.array(X_list, dtype=np.float32), np.array(y_list), meta_list


# ==================== 模型训练 ====================
def train_xgboost(X_train, y_train, X_test, y_test, seed=SEED):
    cw = get_class_weights(y_train)
    sw = np.array([cw[y] for y in y_train])
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        early_stopping_rounds=20, objective='multi:softmax', num_class=3,
        random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_test, y_test)], verbose=0)
    return model, model.predict(X_test)


def train_rf(X_train, y_train, X_test, y_test, seed=SEED):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        min_samples_split=8, max_features='sqrt',
        class_weight='balanced', oob_score=True,
        random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def train_mlp(X_train, y_train, X_test, y_test, seed=SEED):
    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
        batch_size=16, learning_rate='adaptive', max_iter=500,
        early_stopping=True, validation_fraction=0.15,
        random_state=seed, verbose=False)
    model.fit(X_train_s, y_train)
    return model, model.predict(X_test_s)


def train_lstm(X_train, y_train, X_test, y_test, seed=SEED):
    n_classes = 3
    n_timesteps = X_train.shape[1]
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    cw_w = get_class_weights(y_train)
    cw = {i: cw_w[i] for i in range(n_classes) if i in cw_w}
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(n_timesteps, X_train.shape[2])),
        LSTM(64, return_sequences=True), Dropout(0.25),
        LSTM(32, return_sequences=False), Dropout(0.25),
        Dense(16, activation='relu'), Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    history = model.fit(X_train, y_train_cat, validation_split=0.15, epochs=100,
                        batch_size=16, class_weight=cw, callbacks=[es], verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    model.history_ = history
    return model, y_pred


# ==================== Optuna 训练函数 ====================
def train_xgboost_optuna(X_train, y_train, X_test, y_test, seed=SEED, n_trials=50):
    import optuna
    cw = get_class_weights(y_train)
    sw = np.array([cw[y] for y in y_train])

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }
        model = xgb.XGBClassifier(**params, objective='multi:softmax', num_class=3,
                                   random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_test, y_test)], verbose=0)
        pred = model.predict(X_test)
        return f1_score(y_test, pred, average='weighted')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = xgb.XGBClassifier(**study.best_params, objective='multi:softmax', num_class=3,
                              random_state=seed, n_jobs=-1, verbosity=0)
    best.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_test, y_test)], verbose=0)
    return best, best.predict(X_test)


def train_rf_optuna(X_train, y_train, X_test, y_test, seed=SEED, n_trials=50):
    import optuna

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        }
        model = RandomForestClassifier(**params, class_weight='balanced',
                                        random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        return f1_score(y_test, model.predict(X_test), average='weighted')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = RandomForestClassifier(**study.best_params, class_weight='balanced',
                                   random_state=seed, n_jobs=-1)
    best.fit(X_train, y_train)
    return best, best.predict(X_test)


def train_mlp_optuna(X_train, y_train, X_test, y_test, seed=SEED, n_trials=50):
    import optuna
    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)

    def objective(trial):
        params = {
            'hidden_layer_sizes': trial.suggest_categorical(
                'hidden_layer_sizes', [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)]),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        }
        model = MLPClassifier(**params, max_iter=500, early_stopping=True,
                               validation_fraction=0.15, random_state=seed, verbose=False)
        model.fit(X_train_s, y_train)
        return f1_score(y_test, model.predict(X_test_s), average='weighted')

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = MLPClassifier(**study.best_params, max_iter=500, early_stopping=True,
                          validation_fraction=0.15, random_state=seed, verbose=False)
    best.fit(X_train_s, y_train)
    return best, best.predict(X_test_s)


# ==================== 标准模型字典 ====================
STANDARD_MODELS = {
    'XGBoost': train_xgboost,
    'RandomForest': train_rf,
    'MLP': train_mlp,
}
OPTUNA_MODELS = {
    'XGBoost': train_xgboost_optuna,
    'RandomForest': train_rf_optuna,
    'MLP': train_mlp_optuna,
}
if HAS_TF:
    LSTM_MODEL = {'LSTM': train_lstm}
else:
    LSTM_MODEL = {}


# ==================== 评估 ====================
def _get_lstm_split(meta_ts, test_loc):
    """根据测试 location 切分 LSTM 时序数据"""
    mask = np.array([m['location'] == test_loc for m in meta_ts])
    return mask

def evaluate_cross_location(df, models, loc_keys, loc_labels,
                            use_smote=False, X_ts=None, y_ts=None, meta_ts=None,
                            seed=SEED):
    fc = get_feature_cols(df)
    X_all = df[fc].values
    y_all = df['risk'].values
    n_folds = len(loc_keys)

    print("\n" + "=" * 70)
    print(f"  评估: {n_folds}-Fold 跨 Location 验证")
    print("=" * 70)

    fold_metrics = {name: {'acc': [], 'f1': [], 'macro_f1': [],
                           'high_risk_f1': [], 'high_risk_recall': [], 'test_ratio': []}
                    for name in models}

    for fi, test_loc in enumerate(loc_keys):
        test_mask = df['location'] == test_loc
        X_te_a, y_te = X_all[test_mask], y_all[test_mask]
        X_tr_a, y_tr = X_all[~test_mask], y_all[~test_mask]
        train_lbl = [loc_labels[i] for i in range(n_folds) if loc_keys[i] != test_loc]

        # LSTM split
        X_tr_t = X_te_t = y_tr_t = y_te_t = None
        if models.get('LSTM') and X_ts is not None:
            lstm_mask = _get_lstm_split(meta_ts, test_loc)
            X_tr_t, X_te_t = X_ts[~lstm_mask], X_ts[lstm_mask]
            y_tr_t, y_te_t = y_ts[~lstm_mask], y_ts[lstm_mask]

        # SMOTE
        X_tr_sm = y_tr_sm = None
        if use_smote:
            sm = SMOTE(sampling_strategy='not majority', random_state=seed)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_a, y_tr)

        print(f"\nFold {fi+1}: 测试={loc_labels[fi]} ({sum(test_mask)}辆), "
              f"训练={train_lbl} ({sum(~test_mask)}辆)"
              + (f" → SMOTE后 {len(y_tr_sm)}辆" if use_smote else ""))

        for name, fn in models.items():
            if name == 'LSTM' and X_ts is not None:
                _, yp = fn(X_tr_t, y_tr_t, X_te_t, y_te_t)
                yt = y_te_t
            else:
                X_use, y_use = (X_tr_sm, y_tr_sm) if use_smote else (X_tr_a, y_tr)
                _, yp = fn(X_use, y_use, X_te_a, y_te)
                yt = y_te
            fold_metrics[name]['acc'].append(accuracy_score(yt, yp))
            fold_metrics[name]['f1'].append(f1_score(yt, yp, average='weighted'))
            fold_metrics[name]['macro_f1'].append(f1_score(yt, yp, average='macro'))
            fold_metrics[name]['high_risk_f1'].append(f1_score(yt, yp, labels=[0], average=None)[0])
            fold_metrics[name]['high_risk_recall'].append(recall_score(yt, yp, labels=[0], average=None)[0])
            fold_metrics[name]['test_ratio'].append(sum(test_mask) / sum(~test_mask))
            print(f"  {name:15s}: Acc={fold_metrics[name]['acc'][-1]:.3f}, "
                  f"F1={fold_metrics[name]['f1'][-1]:.3f}, MacroF1={fold_metrics[name]['macro_f1'][-1]:.3f}")

    print("\n--- 跨 Location 汇总 ---")
    for name in models:
        print(f"  {name:15s}: Acc={np.mean(fold_metrics[name]['acc']):.3f}±{np.std(fold_metrics[name]['acc']):.3f}, "
              f"F1={np.mean(fold_metrics[name]['f1']):.3f}±{np.std(fold_metrics[name]['f1']):.3f}, "
              f"MacroF1={np.mean(fold_metrics[name]['macro_f1']):.3f}±{np.std(fold_metrics[name]['macro_f1']):.3f}")

    return fold_metrics


def evaluate_random_split(df, models, use_smote=False, X_ts=None, y_ts=None, seed=SEED):
    fc = get_feature_cols(df)
    X_all = df[fc].values
    y_all = df['risk'].values

    X_tr_a, X_te_a, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=seed)

    # LSTM split
    X_tr_t = X_te_t = y_tr_t = y_te_t = None
    if models.get('LSTM') and X_ts is not None:
        ii = np.arange(len(X_ts))
        it, ie = train_test_split(ii, test_size=0.2, stratify=y_ts, random_state=seed)
        X_tr_t, X_te_t = X_ts[it], X_ts[ie]
        y_tr_t, y_te_t = y_ts[it], y_ts[ie]

    # SMOTE
    X_tr_sm = y_tr_sm = None
    train_lbl = f"{len(X_tr_a)}辆"
    if use_smote:
        sm = SMOTE(sampling_strategy='not majority', random_state=seed)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_a, y_tr)
        train_lbl += f" → SMOTE后 {len(y_tr_sm)}辆"

    print("\n" + "=" * 70)
    print("  评估: 随机 80/20 划分")
    print("=" * 70)
    print(f"  训练: {train_lbl}, 测试: {len(X_te_a)} 辆")

    results = {}
    for name, fn in models.items():
        if name == 'LSTM' and X_ts is not None:
            model, yp = fn(X_tr_t, y_tr_t, X_te_t, y_te_t)
            yr, ypr = y_te_t, yp
        else:
            X_use, y_use = (X_tr_sm, y_tr_sm) if use_smote else (X_tr_a, y_tr)
            model, yp = fn(X_use, y_use, X_te_a, y_te)
            yr, ypr = y_te, yp
        acc = accuracy_score(yr, ypr)
        f1 = f1_score(yr, ypr, average='weighted')
        mf1 = f1_score(yr, ypr, average='macro')
        hf1 = f1_score(yr, ypr, labels=[0], average=None)[0]
        hrc = recall_score(yr, ypr, labels=[0], average=None)[0]
        results[name] = {'acc': acc, 'f1': f1, 'macro_f1': mf1,
                         'high_risk_f1': hf1, 'high_risk_recall': hrc,
                         'y_true': yr, 'y_pred': ypr, 'model': model}
        print(f"\n  {name}: Acc={acc:.3f}, F1={f1:.3f}, MacroF1={mf1:.3f}")
        rpt = classification_report(yr, ypr, target_names=['高风险', '中风险', '低风险'], zero_division=0)
        for line in rpt.split('\n'):
            print(f'    {line}')

    results['_data'] = {'X_test': X_te_a, 'feature_names': fc}
    return results


# ==================== 可视化 ====================
def _model_names_and_colors(models, colors=None):
    names = list(models.keys())
    if colors is None:
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    return names, colors[:len(names)]


def plot_comparison(fold_metrics, random_results, models, title, save_name):
    names, colors = _model_names_and_colors(models)
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.35
    for i, name in enumerate(names):
        ax.bar(i, np.mean(fold_metrics[name]['f1']), w / 2,
               yerr=np.std(fold_metrics[name]['f1']),
               color=colors[i], alpha=0.85, capsize=5, label=name)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Weighted F1', fontsize=12)
    ax.set_title('跨 Location Weighted F1', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1); ax.axhline(y=1/3, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[1]
    for i, name in enumerate(names):
        ax.bar(i, np.mean(fold_metrics[name]['macro_f1']), w / 2,
               yerr=np.std(fold_metrics[name]['macro_f1']),
               color=colors[i], alpha=0.85, capsize=5, label=name)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title('跨 Location Macro F1', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1); ax.axhline(y=1/3, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    ax = axes[2]
    for i, name in enumerate(names):
        ax.bar(i - w/2, random_results[name]['acc'], w, color=colors[i], alpha=0.85)
        ax.bar(i + w/2, random_results[name]['f1'], w, color=colors[i], alpha=0.35)
        ax.bar(i + 3*w/2, random_results[name]['macro_f1'], w, color=colors[i], alpha=0.15, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('随机 80/20 划分', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend([Patch(color='gray', alpha=0.85), Patch(color='gray', alpha=0.35),
               Patch(color='gray', alpha=0.15, hatch='//')],
              ['Acc', 'Weighted F1', 'Macro F1'], fontsize=8)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, save_name), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  [OK] {save_name}')
def plot_confusion(random_results, models, title, save_name):
    names = [m for m in models if m in random_results]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    labels = ['高风险', '中风险', '低风险']

    for i, name in enumerate(names):
        res = random_results[name]
        ax = axes[i]
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, cm[r, c], ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[r, c] > cm.max() / 2 else 'black')
        ax.set_xlabel('预测', fontsize=11); ax.set_ylabel('真实', fontsize=11)
        ax.set_title(f'{name} (F1={res["f1"]:.3f})', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, save_name), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {save_name}')
def print_summary(fold_metrics, random_results, models):
    """打印最佳模型总结"""
    rr = {k: v for k, v in random_results.items() if k != '_data'}
    best_r = max(rr, key=lambda n: rr[n]['f1'])
    best_r_m = max(rr, key=lambda n: rr[n]['macro_f1'])
    best_c = max(fold_metrics, key=lambda n: np.mean(fold_metrics[n]['f1']))
    best_c_m = max(fold_metrics, key=lambda n: np.mean(fold_metrics[n]['macro_f1']))
    print("\n" + "=" * 70)
    print("  建模完成")
    print("=" * 70)
    print(f"  最佳模型 (随机划分):   {best_r} (F1={random_results[best_r]['f1']:.3f})")
    print(f"  最佳 MacroF1 (随机):   {best_r_m} (MacroF1={random_results[best_r_m]['macro_f1']:.3f})")
    print(f"  最佳泛化 (跨Location): {best_c} (F1={np.mean(fold_metrics[best_c]['f1']):.3f})")
    print(f"  最佳 MacroF1 (跨Loc):  {best_c_m} (MacroF1={np.mean(fold_metrics[best_c_m]['macro_f1']):.3f})")


# ==================== SHAP 模型解释 ====================
def shap_analysis(model, X, feature_names, model_name, save_prefix='shap', max_display=15):
    """计算并保存 SHAP 特征重要性图

    支持: XGBoost / RandomForest (TreeExplainer)
    输出: 三分类堆叠柱状图 + 高风险 beeswarm 图, 打印 Top-5
    """
    if not HAS_SHAP:
        print('  SHAP 未安装 (pip install shap)')
        return

    print(f'\n  SHAP 分析: {model_name}')

    # ── 计算 SHAP 值 ──
    if model_name == 'XGBoost' and hasattr(model, 'get_booster'):
        # XGBoost 3.x 原生 SHAP（绕过 shap.TreeExplainer 版本兼容问题）
        import xgboost as xgb
        shap_values = model.get_booster().predict(
            xgb.DMatrix(X), pred_contribs=True)  # (N, n_classes, n_features+1)
        # 去掉最后一个 bias 列, 转置为 (N, F, C)
        shap_3d = np.array(shap_values)[:, :, :-1].transpose(0, 2, 1)  # (N, F, C)
    else:
        # RandomForest: shap.TreeExplainer 工作正常
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # 统一为 3D (N, F, C): list → ndarray
        if isinstance(shap_vals, list):
            shap_3d = np.array(shap_vals).transpose(1, 2, 0)  # (C, N, F) → (N, F, C)
        elif shap_vals.ndim == 3:
            shap_3d = shap_vals
        else:
            shap_3d = shap_vals[:, :, np.newaxis]  # 2D → (N, F, 1)

    n_classes = shap_3d.shape[2]
    class_names = ['高风险', '中风险', '低风险']
    class_colors = ['#e74c3c', '#f39c12', '#3498db']

    # ── 自定义堆叠柱状图 (三分类) ──
    mean_abs = np.abs(shap_3d).mean(axis=0)  # (F, C)
    total_imp = mean_abs.sum(axis=1)          # (F,)
    sort_idx = np.argsort(total_imp)[::-1]
    n_show = min(max_display, len(sort_idx))
    sort_idx = sort_idx[:n_show]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(n_show)
    bar_data = mean_abs[sort_idx]  # (n_show, C)
    feat_labels = [feature_names[i] for i in sort_idx]

    left = np.zeros(n_show)
    for c in range(min(n_classes, 3)):
        ax.barh(y_pos, bar_data[:, c], left=left, height=0.7,
                color=class_colors[c], label=class_names[c] if c < len(class_names) else f'Class {c}')
        left += bar_data[:, c]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=10)
    ax.set_xlabel('mean |SHAP value|', fontsize=12)
    ax.set_title(f'{model_name} — 特征重要性 (按风险等级着色)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.invert_yaxis()
    fig.tight_layout()
    bar_path = os.path.join(OUT_DIR, f'{save_prefix}_{model_name}_importance_bar.png')
    fig.savefig(bar_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {os.path.basename(bar_path)}')

    # ── Beeswarm 图: 高风险类 (class 0) ──
    if n_classes >= 1:
        shap_high = shap_3d[:, :, 0]
    else:
        shap_high = shap_3d[:, :, 0]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_high, X, feature_names=feature_names,
                       max_display=max_display, show=False)
    plt.title(f'{model_name} — SHAP Summary (高风险)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    bee_path = os.path.join(OUT_DIR, f'{save_prefix}_{model_name}_beeswarm.png')
    plt.savefig(bee_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [OK] {os.path.basename(bee_path)}')

    # ── 打印 Top-5 (三分类分别) ──
    print(f'  Top-5 特征 (总影响力排序):')
    for rank, idx in enumerate(sort_idx[:5]):
        parts = ' | '.join(
            f'{class_names[c]}={mean_abs[idx, c]:.3f}'
            for c in range(min(n_classes, 3))
        )
        print(f'    {rank+1}. {feature_names[idx]:25s} [{parts}]')


def run_shap_analysis(random_results, models, save_prefix='shap'):
    """对随机划分结果中的 XGBoost/RF 自动执行 SHAP 分析"""
    if not HAS_SHAP:
        print('  [INFO] SHAP 未安装 (pip install shap)，跳过 SHAP 分析')
        return
    if '_data' not in random_results:
        return
    d = random_results['_data']
    for name in models:
        if name in random_results and name in ('XGBoost', 'RandomForest'):
            try:
                shap_analysis(random_results[name]['model'], d['X_test'],
                             d['feature_names'], name, save_prefix)
            except Exception as e:
                print(f'  [WARN] SHAP 分析失败 ({name}): {e}')
