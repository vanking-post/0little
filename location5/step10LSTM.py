import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 可选 GPU 设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)


def train_lstm(save_dir,
                               sequence_length=50,
                               model_save_name="lstm_traffic_model.keras",
                               encoder_save_name="label_encoder.pkl"):
    """
    从指定目录读取 traffic_flow_train.csv 和 traffic_flow_val.csv，
    将每个车辆的 sequence_length 条轨迹重塑为二维时序样本 (帧数, 主成分数)，
    训练 LSTM 多分类模型，输出评估结果、保存模型和编码器，并绘制损失/准确率曲线和混淆矩阵。

    参数:
        save_dir: 数据文件所在目录
        sequence_length: 每个样本的帧数，默认 50
        model_save_name: 保存的模型文件名 (.keras)
        encoder_save_name: 保存的标签编码器文件名 (.pkl)
    返回:
        history: 训练历史对象
        results: 包含准确率、分类报告、混淆矩阵等信息的字典
    """
    # -------------------- 加载数据 --------------------
    train_path = os.path.join(save_dir, "traffic_flow_train.csv")
    val_path = os.path.join(save_dir, "traffic_flow_val.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证集文件不存在: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"训练集形状: {train_df.shape}")
    print(f"验证集形状: {val_df.shape}")

    # 标签列检测
    label_col = 'Label' if 'Label' in train_df.columns else 'label'
    # if label_col not in train_df.columns:
    #     raise KeyError("数据中未找到 'Label' 或 'label' 列")
    print(f"找到标签列: {label_col}")
    print(f"训练集标签分布:\n{train_df[label_col].value_counts()}")
    print(f"验证集标签分布:\n{val_df[label_col].value_counts()}")

    # 主成分列
    pc_columns = [col for col in train_df.columns if col.startswith('PC')]
    if not pc_columns:
        raise ValueError("未找到以 'PC' 开头的主成分列，请检查数据格式")
    print(f"PCA主成分列: {pc_columns}")
    print(f"主成分数量: {len(pc_columns)}")

    # -------------------- 数据重塑为二维时序样本 --------------------
    def reshape_samples(df, pc_cols, n_frames=50):
        X, y = [], []
        for vid, group in df.groupby('ID'):
            group = group.sort_values('Frame') if 'Frame' in group.columns else group
            feats = group[pc_cols].values
            # 截断或补零
            if len(feats) >= n_frames:
                feats = feats[:n_frames]
            else:
                pad = np.zeros((n_frames - len(feats), len(pc_cols)))
                feats = np.vstack([feats, pad])
            label = group[label_col].iloc[0]
            X.append(feats)
            y.append(label)
        return np.array(X), np.array(y)

    print("\n开始构建 LSTM 二维样本...")
    X_train, y_train = reshape_samples(train_df, pc_columns, sequence_length)
    X_val, y_val = reshape_samples(val_df, pc_columns, sequence_length)
    print(f"训练集 X 形状: {X_train.shape}  → (车辆数, 帧数, 主成分数)")
    print(f"验证集 X 形状: {X_val.shape}")
    print(f"训练集 y 形状: {y_train.shape}")
    print(f"验证集 y 形状: {y_val.shape}")

    # -------------------- 标签编码 --------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    n_classes = len(le.classes_)
    print(f"标签类别: {le.classes_}")

    # -------------------- 构建 LSTM 模型 --------------------
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(sequence_length, len(pc_columns))),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # -------------------- 训练 --------------------
    print("\n开始训练 LSTM...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # -------------------- 预测与评估 --------------------
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_val_enc, y_pred)
    print(f"\n========== 验证集准确率 = {acc:.4f} ==========")
    print("\n分类报告:")
    report_str = classification_report(y_val_enc, y_pred, target_names=le.classes_)
    print(report_str)

    # 保存报告字典
    report_dict = classification_report(y_val_enc, y_pred, target_names=le.classes_, output_dict=True)

    cm = confusion_matrix(y_val_enc, y_pred)

    # -------------------- 绘制混淆矩阵 --------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('LSTM 混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.tight_layout()
    plt.show()

    # -------------------- 绘制训练曲线 --------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('准确率曲线')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- 保存模型与编码器 --------------------
    model_path = os.path.join(save_dir, model_save_name)
    encoder_path = os.path.join(save_dir, encoder_save_name)
    model.save(model_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n模型已保存至: {model_path}")
    print(f"标签编码器已保存至: {encoder_path}")

    # 构造返回结果
    results = {
        'accuracy': acc,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'predicted_labels': y_pred.tolist(),
        'true_labels': y_val_enc.tolist(),
        'class_names': le.classes_.tolist(),
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'feature_shape': X_train.shape[1:]
    }

    print("\n==== LSTM 训练与评估全部完成 ====")
    return history, results

# from step07pca import save_dir
# if __name__ == "__main__":
#     # 示例：直接运行脚本时使用默认路径
#     train_lstm(save_dir)