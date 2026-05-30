from tabnanny import verbose

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 0 = 使用你的4060Ti
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)
# print("✅ 当前运行模式：使用 GPU（RTX4060Ti）")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载路径
save_dir = r"E:\0little\read\CQSkyEyedata5"
# save_dir = r"E:\0little\read\CQSkyEyedata5\location5t"
number_sample = 50 #每个样本的数据行数
# 加载训练集和验证集
train_path = os.path.join(save_dir, "traffic_flow_train.csv")
val_path = os.path.join(save_dir, "traffic_flow_val.csv")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

print(f"训练集形状: {train_df.shape}")
print(f"验证集形状: {val_df.shape}")

# 检查标签列
label_col = 'Label' if 'Label' in train_df.columns else 'label'
if label_col in train_df.columns:
    print(f"找到标签列: {label_col}")
    print(f"训练集标签分布:\n{train_df[label_col].value_counts()}")
    print(f"验证集标签分布:\n{val_df[label_col].value_counts()}")
else:
    print("未找到标签列")
    exit()

# 识别PCA主成分列
pc_columns = [col for col in train_df.columns if col.startswith('PC')]
print(f"PCA主成分列: {pc_columns}")
print(f"主成分数量: {len(pc_columns)}")

# 检查Frame列是否存在
if 'Frame' not in train_df.columns:
    print("警告: 未找到Frame列，按ID排序后假设连续50行为一个样本")
else:
    print("找到Frame列，将按Frame顺序处理样本")
# 1. 按 ID 分组，把每辆车 reshape 成 [50帧, 13主成分] 的二维矩阵
def reshape_samples(df, pc_cols, n_frames=50):
    X = []
    y = []
    for vid, group in df.groupby('ID'):
        group = group.sort_values('Frame') if 'Frame' in group.columns else group
        feats = group[pc_cols].values

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
X_train, y_train = reshape_samples(train_df, pc_columns, number_sample)
X_val, y_val = reshape_samples(val_df, pc_columns, number_sample)

print(f"训练集 X 形状: {X_train.shape}  → (车辆数, 帧数, 主成分数)")
print(f"验证集 X 形状: {X_val.shape}")
print(f"训练集 y 形状: {y_train.shape}")
print(f"验证集 y 形状: {y_val.shape}")

# 2. 标签编码（文字标签 → 数字）
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
n_classes = len(le.classes_)
print(f"\n标签类别: {le.classes_}")

# 3. 构建 LSTM 模型
model = Sequential([
    Masking(mask_value=0.0, input_shape=(number_sample, len(pc_columns))),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. 训练
print("\n开始训练 LSTM...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose = 1
)

# 5. 预测
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

# 6. 评价
acc = accuracy_score(y_val_enc, y_pred)
print(f"\n========== 验证集准确率 = {acc:.4f} ==========")
print("\n分类报告:")
print(classification_report(y_val_enc, y_pred, target_names=le.classes_))

# 7. 混淆矩阵
cm = confusion_matrix(y_val_enc, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('混淆矩阵')
plt.xlabel('预测')
plt.ylabel('真实')
plt.show()

# 8. 训练曲线
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
plt.show()

# 9. 保存模型
model_save_path = os.path.join(save_dir, "lstm_traffic_model.keras")
model.save(model_save_path)
le_save_path = os.path.join(save_dir, "label_encoder.pkl")
with open(le_save_path, 'wb') as f:
    pickle.dump(le, f)

print(f"\n模型已保存至: {model_save_path}")
print(f"标签编码器已保存至: {le_save_path}")
print("\n==== LSTM 训练与评估全部完成 ====")