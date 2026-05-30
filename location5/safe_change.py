import matplotlib
matplotlib.use('Agg')  # 避免 tkinter 警告，且不显示图形窗口，只保存图片
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数配置 ====================
save_dir = r"E:\0little\read\CQSkyEyedata5\location5t"
left_file = os.path.join(save_dir, "traffic_left_change.csv")
right_file = os.path.join(save_dir, "traffic_right_change.csv")

# 安全值替换边界（根据指标含义自定义）
BOUNDARIES = {
    'PET': 10.0,  # inf -> 10秒（安全）
    'TTC': 5.0,   # 0 -> 5秒（安全）
    'mTTC': 5.0,  # 0 -> 5秒（安全）
    'Time_Headway': 5.0,  # 0 -> 5秒（安全）
    'Following_dist': 50.0  # 0 -> 50米（无前车视为安全距离）
}

# 时间段划分（50帧，每10帧为一个阶段，共5段，每段0.4秒）
time_segments = {
    'T1 (-2.0~-1.6s)': slice(0, 10),
    'T2 (-1.6~-1.2s)': slice(10, 20),
    'T3 (-1.2~-0.8s)': slice(20, 30),
    'T4 (-0.8~-0.4s)': slice(30, 40),
    'T5 (-0.4~0.0s)': slice(40, 50)
}


# ==================== 数据加载与预处理 ====================
def load_and_preprocess(file_path, direction):
    """加载CSV，替换安全占位符，添加车辆内帧序号"""
    df = pd.read_csv(file_path)
    df['Direction'] = direction

    # 替换安全占位符
    for col, bound in BOUNDARIES.items():
        if col in df.columns:
            if col == 'PET':
                df[col] = df[col].replace([np.inf, -np.inf], bound)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(bound)
            else:
                df[col] = df[col].replace(0, bound)

    # 确保每个车辆有按时间排序的帧序号（0~49）
    df = df.sort_values(['ID', 'Frame'])
    df['FrameIdx'] = df.groupby('ID').cumcount()
    df = df[df['FrameIdx'] < 50]
    return df


print("加载数据...")
left_df = load_and_preprocess(left_file, '左变道')
right_df = load_and_preprocess(right_file, '右变道')
data = pd.concat([left_df, right_df], ignore_index=True)
print(f"左变道样本数: {left_df['ID'].nunique()}, 右变道样本数: {right_df['ID'].nunique()}")

# ==================== 统计 TTC 和 mTTC 小于2的样本数 ====================
print("\n========== 危险样本统计（<2秒）==========")
# 帧级别统计
ttc_lt2_frames = (data['TTC'] < 2).sum()
mttc_lt2_frames = (data['mTTC'] < 2).sum()
print(f"TTC < 2 帧数: {ttc_lt2_frames} / {len(data)} ({ttc_lt2_frames/len(data)*100:.2f}%)")
print(f"mTTC < 2 帧数: {mttc_lt2_frames} / {len(data)} ({mttc_lt2_frames/len(data)*100:.2f}%)")

# 车辆级别统计：每辆车取最后一帧（变道前瞬间）的值
last_frames = data.groupby('ID').last().reset_index()
ttc_lt2_veh = (last_frames['TTC'] < 2).sum()
mttc_lt2_veh = (last_frames['mTTC'] < 2).sum()
print(f"TTC < 2 车辆数: {ttc_lt2_veh} / {last_frames['ID'].nunique()} ({ttc_lt2_veh/last_frames['ID'].nunique()*100:.2f}%)")
print(f"mTTC < 2 车辆数: {mttc_lt2_veh} / {last_frames['ID'].nunique()} ({mttc_lt2_veh/last_frames['ID'].nunique()*100:.2f}%)")

# ==================== 风险分类（基于 mTTC） ====================
def classify_risk_mttc(row):
    mttc = row['mTTC']
    pet = row['PET']
    if mttc < 2 and pet < 2:
        return '双重危险'
    elif mttc < 2:
        return '前向危险'
    elif pet < 2:
        return '后向危险'
    else:
        return '安全'

# ==================== 风险分类（基于 TTC） ====================
def classify_risk_ttc(row):
    ttc = row['TTC']
    pet = row['PET']
    if ttc < 2 and pet < 2:
        return '双重危险'
    elif ttc < 2:
        return '前向危险'
    elif pet < 2:
        return '后向危险'
    else:
        return '安全'

# 计算基于 mTTC 的风险标签
vehicle_risk_mttc = {}
for vid, group in data.groupby('ID'):
    last_frame = group[group['FrameIdx'] == 49]
    if not last_frame.empty:
        row = last_frame.iloc[0]
        risk = classify_risk_mttc(row)
        vehicle_risk_mttc[vid] = risk
    else:
        vehicle_risk_mttc[vid] = '安全'
data['VehicleRisk_mTTC'] = data['ID'].map(vehicle_risk_mttc)

# 计算基于 TTC 的风险标签
vehicle_risk_ttc = {}
for vid, group in data.groupby('ID'):
    last_frame = group[group['FrameIdx'] == 49]
    if not last_frame.empty:
        row = last_frame.iloc[0]
        risk = classify_risk_ttc(row)
        vehicle_risk_ttc[vid] = risk
    else:
        vehicle_risk_ttc[vid] = '安全'
data['VehicleRisk_TTC'] = data['ID'].map(vehicle_risk_ttc)

# 统计分布（车辆数）
risk_stats_mttc = data.groupby(['Direction', 'VehicleRisk_mTTC']).size().unstack(fill_value=0)
risk_stats_ttc = data.groupby(['Direction', 'VehicleRisk_TTC']).size().unstack(fill_value=0)
print("\n基于 mTTC 的风险分布（车辆数）:")
print(risk_stats_mttc)
print("\n基于 TTC 的风险分布（车辆数）:")
print(risk_stats_ttc)

# ==================== 风险矩阵散点图（mTTC vs PET） ====================
def plot_risk_matrix(df, x_col, y_col, risk_col, title, filename):
    plt.figure(figsize=(10, 8))
    last_frames = df.groupby('ID').last().reset_index().dropna(subset=[x_col, y_col])
    colors = {'双重危险': 'red', '前向危险': 'orange', '后向危险': 'blue', '安全': 'green'}
    for risk, color in colors.items():
        subset = last_frames[last_frames[risk_col] == risk]
        if not subset.empty:
            plt.scatter(subset[x_col], subset[y_col], c=color, label=risk, alpha=0.6, s=30)
    plt.axvline(x=2, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(y=2, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel(f'{x_col} (秒)')
    plt.ylabel(f'{y_col} (秒)')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

plot_risk_matrix(data, 'mTTC', 'PET', 'VehicleRisk_mTTC',
                 '变道风险矩阵 (mTTC vs PET)', 'risk_matrix_scatter_mTTC.png')
plot_risk_matrix(data, 'TTC', 'PET', 'VehicleRisk_TTC',
                 '变道风险矩阵 (TTC vs PET)', 'risk_matrix_scatter_TTC.png')

# ==================== 趋势分析（基于 mTTC 风险分组，保持原有逻辑） ====================
data['RiskGroup'] = data['VehicleRisk_mTTC'].apply(lambda x: '危险' if x != '安全' else '安全')
indicators = ['TTC', 'mTTC', 'Time_Headway', 'Following_dist', 'Velocity', 'lat_Acc']
trend_data = defaultdict(list)

for (risk_group, dir_group), group_df in data.groupby(['RiskGroup', 'Direction']):
    for seg_name, seg_slice in time_segments.items():
        seg_df = group_df[group_df['FrameIdx'].between(seg_slice.start, seg_slice.stop - 1)]
        means = seg_df[indicators].mean()
        for ind in indicators:
            trend_data[(risk_group, dir_group, seg_name, ind)] = means[ind]

trend_index = pd.MultiIndex.from_tuples(trend_data.keys(), names=['RiskGroup', 'Direction', 'Segment', 'Indicator'])
trend_series = pd.Series(trend_data, index=trend_index)
trend_df = trend_series.unstack(level='Indicator').reset_index()
seg_order = list(time_segments.keys())

for ind in indicators:
    plt.figure(figsize=(12, 6))
    for risk in ['安全', '危险']:
        for dir_ in ['左变道', '右变道']:
            subset = trend_df[(trend_df['RiskGroup'] == risk) & (trend_df['Direction'] == dir_)]
            if subset.empty:
                continue
            y_vals = [subset[subset['Segment'] == seg][ind].values[0] for seg in seg_order]
            plt.plot(seg_order, y_vals, marker='o', label=f'{dir_}-{risk}')
    plt.title(f'变道前 {ind} 时序变化 (安全 vs 危险)')
    plt.xlabel('时间段 (变道前)')
    plt.ylabel(ind)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trend_{ind}.png'), dpi=150)
    plt.close()

# ==================== 高风险车辆时序大图（使用 mTTC 双重危险） ====================
danger_vehicles = data[data['VehicleRisk_mTTC'] == '双重危险']['ID'].unique()
print(f"\n基于 mTTC 的双重危险车辆数: {len(danger_vehicles)}")
if len(danger_vehicles) > 0:
    n_plot = min(10, len(danger_vehicles))
    selected_ids = np.random.choice(danger_vehicles, n_plot, replace=False)
    fig, axes = plt.subplots(5, 2, figsize=(20, 11.25))
    axes = axes.flatten()
    for idx, vid in enumerate(selected_ids):
        ax = axes[idx]
        veh_data = data[data['ID'] == vid].sort_values('FrameIdx')
        frames = veh_data['FrameIdx'].values
        ax.plot(frames, veh_data['Velocity'], label='速度 (m/s)', color='blue')
        ax.plot(frames, veh_data['TTC'], label='TTC', color='green')
        ax.plot(frames, veh_data['mTTC'], label='mTTC', color='orange')
        ax.plot(frames, veh_data['Time_Headway'], label='THW', color='purple')
        ax.plot(frames, veh_data['Following_dist'], label='跟驰距 (m)', color='brown')
        pet_val = veh_data['PET'].iloc[0]
        ax.axhline(y=pet_val, color='red', linestyle='--', label=f'PET={pet_val:.1f}s')
        ax.set_title(f'车辆 {vid} (双重危险)')
        ax.set_xlabel('变道前帧序号')
        ax.set_ylabel('指标值')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    for idx in range(n_plot, 10):
        axes[idx].axis('off')
    plt.suptitle('高风险（双重危险）车辆变道前时序指标', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'high_risk_vehicles_traces.png'), dpi=150)
    plt.close()
else:
    print("没有基于 mTTC 的双重危险车辆，跳过高风险时序图。")

# ==================== 统计表格输出 ====================
# 帧级别风险分布（基于 mTTC）
frame_risk_stats = data.groupby(['Direction', 'VehicleRisk_mTTC']).size().unstack(fill_value=0)
print("\n帧级别风险分布（基于 mTTC，帧数）:")
print(frame_risk_stats)

# 基于 TTC 的帧级别分布
frame_risk_stats_ttc = data.groupby(['Direction', 'VehicleRisk_TTC']).size().unstack(fill_value=0)
print("\n帧级别风险分布（基于 TTC，帧数）:")
print(frame_risk_stats_ttc)

# 各风险组平均指标（变道前瞬间，基于 mTTC）
last_frames_all = data.groupby('ID').last().reset_index().dropna(subset=['mTTC', 'PET'])
summary_stats = last_frames_all.groupby('VehicleRisk_mTTC')[indicators + ['PET']].mean().round(2)
print("\n各风险组平均指标（变道前瞬间，基于 mTTC）:")
print(summary_stats)

# 保存到CSV
summary_stats.to_csv(os.path.join(save_dir, 'risk_group_summary_mTTC.csv'))
frame_risk_stats.to_csv(os.path.join(save_dir, 'risk_distribution_mTTC.csv'))
frame_risk_stats_ttc.to_csv(os.path.join(save_dir, 'risk_distribution_TTC.csv'))

print("\n分析完成，结果已保存至:", save_dir)