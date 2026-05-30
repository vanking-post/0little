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
#from step07pca import save_dir
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def train_svm(save_dir ,
                              sequence_length=50,
                              model_save_name="time_series_svm_model.pkl",
                              encoder_save_name="label_encoder.pkl",
                              results_save_name="time_series_svm_results.pkl"):
    """
    从指定目录下读取 traffic_flows_train.csv 和 traffic_flows_val.csv，
    将每个车辆的 sequence_length 条连续轨迹转换为一个样本，
    训练 SVM 多分类模型，输出评估结果并保存模型、编码器及详细结果。

    参数:
        data_dir: 数据文件所在目录，默认 "E:\\0little\\read\\CQSkyEyedata5\\location5t"
        sequence_length: 每个样本包含的行数（时间步），默认 50
        model_save_name: 保存的模型文件名
        encoder_save_name: 保存的标签编码器文件名
        results_save_name: 保存的结果字典文件名
    """
    # ==================== 加载数据 ====================
    train_path = os.path.join(save_dir, "traffic_flows_train.csv")
    val_path = os.path.join(save_dir, "traffic_flows_val.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"验证集文件不存在: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"训练集形状: {train_df.shape}")
    print(f"验证集形状: {val_df.shape}")

    # 检测标签列
    label_col = 'Label' if 'Label' in train_df.columns else 'label'
    if label_col not in train_df.columns:
        raise KeyError("数据中未找到 'Label' 或 'label' 列")
    print(f"找到标签列: {label_col}")
    print(f"训练集标签分布:\n{train_df[label_col].value_counts()}")
    print(f"验证集标签分布:\n{val_df[label_col].value_counts()}")

    # 识别 PCA 主成分列
    pc_columns = [col for col in train_df.columns if col.startswith('PC')]
    if not pc_columns:
        raise ValueError("未找到以 'PC' 开头的主成分列，请检查数据格式")
    print(f"PCA主成分列: {pc_columns}")
    print(f"主成分数量: {len(pc_columns)}")

    # ==================== 数据重塑为时间序列样本 ====================
    def reshape_to_sequences(df, sequence_length=50):
        """
        将数据重塑为时间序列样本：每个车辆ID对应 sequence_length 条连续数据作为一个样本
        """
        # 按ID和Frame排序
        if 'Frame' in df.columns:
            df_sorted = df.sort_values(['ID', 'Frame']).reset_index(drop=True)
        else:
            df_sorted = df.sort_values('ID').reset_index(drop=True)

        unique_ids = df_sorted['ID'].unique()
        X_samples = []
        y_samples = []

        for vehicle_id in unique_ids:
            id_data = df_sorted[df_sorted['ID'] == vehicle_id]
            if len(id_data) == sequence_length:
                # 提取 PCA 特征序列
                pc_features = id_data[pc_columns].values  # (seq_len, n_features)
                # 展平为一维向量
                flattened_features = pc_features.flatten()
                X_samples.append(flattened_features)
                # 假设同一车辆的所有行标签一致，取第一个
                label = id_data[label_col].iloc[0]
                y_samples.append(label)
            else:
                print(f"警告: 车辆ID {vehicle_id} 只有 {len(id_data)} 条数据（期望 {sequence_length} 条），跳过")

        return np.array(X_samples), np.array(y_samples)

    print("\n正在将训练数据重塑为时间序列样本...")
    X_train, y_train = reshape_to_sequences(train_df, sequence_length)
    print(f"训练样本形状: {X_train.shape}, 训练标签形状: {y_train.shape}")

    print("正在将验证数据重塑为时间序列样本...")
    X_val, y_val = reshape_to_sequences(val_df, sequence_length)
    print(f"验证样本形状: {X_val.shape}, 验证标签形状: {y_val.shape}")

    # ==================== 标签编码 ====================
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    class_names = le.classes_
    print(f"标签编码映射: {dict(zip(class_names, le.transform(class_names)))}")
    print(f"训练样本数量: {len(X_train)}, 验证样本数量: {len(X_val)}")

    # ==================== SVM 网格搜索 ====================
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }
    print("\n开始网格搜索最佳参数...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train_encoded)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

    # 用最佳模型预测验证集
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_val)
    accuracy = accuracy_score(y_val_encoded, y_pred)
    print(f"最佳参数在验证集上的准确率: {accuracy:.4f}")

    # ==================== 评估与报告 ====================
    report = classification_report(y_val_encoded, y_pred, target_names=class_names, output_dict=True)
    print("\n分类报告:")
    print(classification_report(y_val_encoded, y_pred, target_names=class_names))

    cm = confusion_matrix(y_val_encoded, y_pred)
    print("混淆矩阵:")
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

    # ==================== 保存模型与结果 ====================
    # 确保保存目录存在
    model_path = os.path.join(save_dir, model_save_name)
    encoder_path = os.path.join(save_dir, encoder_save_name)
    results_path = os.path.join(save_dir, results_save_name)

    with open(model_path, 'wb') as f:
        pickle.dump(best_svm, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n模型已保存至: {model_path}")
    print(f"标签编码器已保存至: {encoder_path}")

    # 保存详细评估结果
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
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"详细结果已保存至: {results_path}")

    # ==================== 总结输出 ====================
    print("\n" + "=" * 50)
    print("SVM模型性能总结 (基于时间序列样本)")
    print("=" * 50)
    print(f"模型类型: 支持向量机 (SVM)")
    print(f"最优参数: {grid_search.best_params_}")
    print(f"验证集准确率: {accuracy:.4f}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别分布: {dict(zip(class_names, np.bincount(y_val_encoded)))}")
    print(f"特征维度: {X_train.shape[1]} (每个样本包含 {sequence_length} 条时间序列数据)")
    print(f"训练样本数: {X_train.shape[0]}")
    print(f"验证样本数: {X_val.shape[0]}")
    print("\nSVM时间序列样本预测模型完成！")


# if __name__ == "__main__":
#     # 直接运行本脚本时将执行训练流程
#     train_svm()