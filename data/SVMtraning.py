import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载路径
save_dir = r"E:\0little\read\CQSkyEyedata5"

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


# 函数：将50条时间连续序列数据重塑为一个样本
def reshape_to_sequences(df, sequence_length=50):
    """
    将数据重塑为时间序列样本
    每个车辆ID对应50条连续数据作为一个样本
    """
    # 按ID和Frame排序
    if 'Frame' in df.columns:
        df_sorted = df.sort_values(['ID', 'Frame']).reset_index(drop=True)
    else:
        df_sorted = df.sort_values('ID').reset_index(drop=True)

    # 获取唯一的ID
    unique_ids = df_sorted['ID'].unique()

    # 存储重塑后的数据和标签
    X_samples = []
    y_samples = []

    for vehicle_id in unique_ids:
        # 获取该ID的50条数据
        id_data = df_sorted[df_sorted['ID'] == vehicle_id]

        if len(id_data) == 50:  # 确保有50条数据
            # 提取PCA特征
            pc_features = id_data[pc_columns].values  # 形状为 (50, n_features)

            # 获取标签（假设所有50条记录的标签相同）
            label = id_data[label_col].iloc[0]

            # 将50条时间序列数据重塑为一个样本
            # 方法1: 展平为一维向量 (50*n_features,)
            flattened_features = pc_features.flatten()

            # 或者可以使用其他聚合方法，比如均值、最大值等
            # mean_features = pc_features.mean(axis=0)  # 平均值
            # std_features = pc_features.std(axis=0)   # 标准差
            # combined_features = np.concatenate([mean_features, std_features])  # 结合统计特征

            X_samples.append(flattened_features)
            y_samples.append(label)
        else:
            print(f"警告: 车辆ID {vehicle_id} 只有 {len(id_data)} 条数据，跳过")

    return np.array(X_samples), np.array(y_samples)


print("正在将训练数据重塑为时间序列样本...")
X_train, y_train = reshape_to_sequences(train_df)
print(f"训练样本形状: {X_train.shape}")
print(f"训练标签形状: {y_train.shape}")

print("正在将验证数据重塑为时间序列样本...")
X_val, y_val = reshape_to_sequences(val_df)
print(f"验证样本形状: {X_val.shape}")
print(f"验证标签形状: {X_val.shape}")

# 标签编码
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

print(f"标签编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 检查样本数量
print(f"训练样本数量: {len(X_train)}")
print(f"验证样本数量: {len(X_val)}")

# SVM模型参数网格搜索
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

print("开始网格搜索最佳参数...")
svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train_encoded)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 使用最佳参数训练最终模型
best_svm = grid_search.best_estimator_
print("使用最佳参数训练最终模型...")

# 在验证集上进行预测
print("在验证集上进行预测...")
y_pred = best_svm.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val_encoded, y_pred)
print(f"验证集准确率: {accuracy:.4f}")

# 生成分类报告
class_names = le.classes_
report = classification_report(y_val_encoded, y_pred, target_names=class_names, output_dict=True)
print(f"\n分类报告:")
print(classification_report(y_val_encoded, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_val_encoded, y_pred)
print(f"\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('SVM分类混淆矩阵 (基于时间序列样本)')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 特征重要性分析（仅对线性核有效）
if best_svm.kernel == 'linear':
    # 获取线性SVM的系数
    coef = best_svm.coef_
    if coef.ndim > 1:
        # 多类问题，取平均绝对系数
        feature_importance = np.mean(np.abs(coef), axis=0)
    else:
        feature_importance = np.abs(coef)

    # 绘制特征重要性
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title(f'SVM特征重要性 ({best_svm.kernel} kernel)')
    plt.xlabel('特征索引 (Flattened Time Series Features)')
    plt.ylabel('重要性 (系数绝对值)')
    plt.xticks(range(0, len(feature_importance), max(1, len(feature_importance) // 10)),
               [f'F{i + 1}' for i in range(len(feature_importance))][::max(1, len(feature_importance) // 10)],
               rotation=45)
    plt.tight_layout()
    plt.show()

    # 显示最重要的前10个特征
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print(f"\n最重要的10个特征 (Flattened):")
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. Feature_{idx}: {feature_importance[idx]:.4f}")

# 保存模型
model_path = os.path.join(save_dir, "time_series_svm_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_svm, f)

# 保存标签编码器
encoder_path = os.path.join(save_dir, "label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)

print(f"\n模型已保存至: {model_path}")
print(f"标签编码器已保存至: {encoder_path}")

# 模型性能总结
print(f"\n" + "=" * 50)
print("SVM模型性能总结 (基于时间序列样本)")
print("=" * 50)
print(f"模型类型: 支持向量机 (SVM)")
print(f"最优参数: {grid_search.best_params_}")
print(f"验证集准确率: {accuracy:.4f}")
print(f"类别数量: {len(class_names)}")
print(f"特征维度: {X_train.shape[1]} (每个样本包含50条时间序列数据)")
print(f"训练样本数: {X_train.shape[0]} (每个样本为一个车辆ID的50条数据)")
print(f"验证样本数: {X_val.shape[0]} (每个样本为一个车辆ID的50条数据)")
print(f"类别分布: {dict(zip(class_names, np.bincount(y_val_encoded)))}")

# 保存详细结果
results = {
    'model_params': grid_search.best_params_,
    'accuracy': accuracy,
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'predicted_labels': y_pred.tolist(),
    'true_labels': y_val_encoded.tolist(),
    'class_names': class_names.tolist(),
    'feature_count': X_train.shape[1],
    'train_samples': X_train.shape[0],
    'val_samples': X_val.shape[0]
}

results_path = os.path.join(save_dir, "time_series_svm_results.pkl")
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f"\n详细结果已保存至: {results_path}")

# 预测示例
print(f"\n预测示例 (前5个样本):")
for i in range(min(5, len(y_pred))):
    true_label = le.inverse_transform([y_val_encoded[i]])[0]
    pred_label = le.inverse_transform([y_pred[i]])[0]
    print(
        f"样本 {i + 1}: 真实标签={true_label}, 预测标签={pred_label}, 正确={'✓' if true_label == pred_label else '✗'}")

print(f"\nSVM时间序列样本预测模型完成！")
