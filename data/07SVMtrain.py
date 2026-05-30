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

        if len(id_data) == number_sample:  # 确保有50条数据
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
            mix_feature = flattened_features
            X_samples.append(mix_feature)
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
param_grid = {     'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                   'kernel': ['rbf', 'linear', 'poly']}

print("开始网格搜索最佳参数...")
svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train_encoded)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_val) #用得到的最佳参数训练模型SVM模型并预测验证集。
accuracy = accuracy_score(y_val_encoded, y_pred)
print(f"最佳参数在验证集上的准确率: {accuracy:.4f}")

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

# 保存模型
model_path = os.path.join(save_dir, "time_series_svm_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(best_svm, f)

# 保存标签编码器
encoder_path = os.path.join(save_dir, "label_encoder.pkl")
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)

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
print(f"\nSVM时间序列样本预测模型完成！")