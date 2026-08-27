"""
变道安全分析 - 交互式 Plotly 仪表盘
读取已有的变道 CSV 数据，生成交互式 HTML 报告
"""
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
@dataclass
class Config:
    """集中管理所有配置常量"""
    # 路径配置
    DATA_DIR: str = r"E:\0little\location1"
    LOC5_DIR: str = r"E:\0little\read\CQSkyEyedata5\location5t"
    OUT_HTML: str = r"E:\0little\location1\safety_dashboard.html"
    
    # 颜色配置
    RISK_COLORS: Dict[str, str] = {
        '高风险': '#E53935',
        '中风险': '#FF9800',
        '低风险': '#4CAF50'
    }
    BLUE: str = '#2196F3'
    RED: str = '#E53935'
    BEHAVIOR_COLORS: Dict[str, str] = {
        '左变道': '#2196F3',
        '右变道': '#E53935'
    }
    
    # 风险等级标记配置
    RISK_MARKERS: Dict[str, Tuple[str, str]] = {
        '高风险': ('circle', '#E53935'),
        '中风险': ('diamond', '#FF9800'),
        '低风险': ('triangle-up', '#4CAF50')
    }
    
    # 安全阈值
    TTC_THRESHOLDS: Tuple[float, float] = (2.0, 5.0)
    PET_THRESHOLDS: Tuple[float, float] = (2.0, 5.0)
    
    # 图表配置
    TEMPLATE: str = 'plotly_white'
    HEIGHT: int = 420
    
    # 文件映射
    FILES: Dict[Tuple[str, str, str], str] = None
    
    def __post_init__(self):
        """初始化文件映射"""
        self.FILES = {
            ('1-1',  '左变道', 'loc1'): os.path.join(self.DATA_DIR, 'traffic_1-1_left_change.csv'),
            ('1-1',  '右变道', 'loc1'): os.path.join(self.DATA_DIR, 'traffic_1-1_right_change.csv'),
            ('1-2',  '左变道', 'loc1'): os.path.join(self.DATA_DIR, 'traffic_1-2_left_change.csv'),
            ('1-2',  '右变道', 'loc1'): os.path.join(self.DATA_DIR, 'traffic_1-2_right_change.csv'),
            ('loc5', '左变道', 'loc5'): os.path.join(self.LOC5_DIR, 'traffic_left_change.csv'),
            ('loc5', '右变道', 'loc5'): os.path.join(self.LOC5_DIR, 'traffic_right_change.csv'),
        }


# ==================== 辅助函数 ====================
def ensure_safety_categories(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """确保 DataFrame 包含安全分类列，若没有则自动计算
    
    Args:
        df: 输入 DataFrame
        config: 配置对象
        
    Returns:
        添加了安全分类列的 DataFrame
    """
    ttc_low, ttc_high = config.TTC_THRESHOLDS
    pet_low, pet_high = config.PET_THRESHOLDS
    
    if 'TTC_cat' not in df.columns:
        df['TTC_cat'] = 'safe'
        df.loc[df['TTC'] == 0, 'TTC_cat'] = 'no_leader'
        df.loc[(df['TTC'] > 0) & (df['TTC'] < ttc_low), 'TTC_cat'] = 'dangerous'
        df.loc[(df['TTC'] >= ttc_low) & (df['TTC'] < ttc_high), 'TTC_cat'] = 'cautious'
    
    if 'PET_cat' not in df.columns:
        df['PET_cat'] = 'safe'
        pi = np.isinf(df['PET'].values) | pd.isna(df['PET'].values)
        df.loc[pi, 'PET_cat'] = 'no_follower'
        df.loc[(df['PET'] > 0) & (df['PET'] < pet_low), 'PET_cat'] = 'dangerous'
        df.loc[(df['PET'] >= pet_low) & (df['PET'] < pet_high), 'PET_cat'] = 'cautious'
    
    if 'OL_PET_cat' not in df.columns and 'OL_PET' in df.columns:
        df['OL_PET_cat'] = 'safe'
        olp_inv = np.isinf(df['OL_PET'].values) | pd.isna(df['OL_PET'].values)
        df.loc[olp_inv, 'OL_PET_cat'] = 'no_follower'
        df.loc[(df['OL_PET'] > 0) & (df['OL_PET'] < pet_low), 'OL_PET_cat'] = 'dangerous'
        df.loc[(df['OL_PET'] >= pet_low) & (df['OL_PET'] < pet_high), 'OL_PET_cat'] = 'cautious'
    
    return df


def classify_risk_vectorized(df: pd.DataFrame) -> pd.Series:
    """向量化风险等级分类，替代逐行 apply
    
    Args:
        df: 包含 TTC_cat 和 PET_cat 列的 DataFrame
        
    Returns:
        风险等级 Series
    """
    conditions = [
        (df['TTC_cat'] == 'dangerous') | (df['PET_cat'] == 'dangerous'),
        (df['TTC_cat'] == 'cautious') | (df['PET_cat'] == 'cautious')
    ]
    choices = ['高风险', '中风险']
    return np.select(conditions, choices, default='低风险')


def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载并预处理所有数据文件
    
    Args:
        config: 配置对象
        
    Returns:
        (df_all, last) 元组：全量数据和每辆车最后一帧数据
    """
    logger.info("加载数据...")
    frames = []
    
    for (section, behavior, src_type), fpath in config.FILES.items():
        if not os.path.exists(fpath):
            logger.warning(f"文件不存在，跳过: {fpath}")
            continue
        
        try:
            df = pd.read_csv(fpath)
            df['Source'] = section
            df['Behavior'] = behavior
            df['SourceType'] = src_type
            df['FrameIdx'] = df.groupby('ID').cumcount()
            df = ensure_safety_categories(df, config)
            frames.append(df)
            logger.info(f"已加载: {os.path.basename(fpath)} ({len(df)} 行)")
        except Exception as e:
            logger.error(f"加载文件失败 {fpath}: {e}")
            continue
    
    if not frames:
        raise ValueError("未能成功加载任何数据文件")
    
    df_all = pd.concat(frames, ignore_index=True)
    df_all['Velocity_ms'] = df_all['Velocity']
    
    # 获取每辆车最后一帧数据
    last = df_all.groupby('ID').last().reset_index()
    
    # 处理无穷值和零值
    for c in ['TTC', 'mTTC', 'PET', 'Time_Headway']:
        if c in last.columns:
            last[c] = last[c].replace([float('inf'), -float('inf')], float('nan'))
    for dc in ['LB_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RF_Dist']:
        if dc in last.columns:
            last[dc] = last[dc].replace(0, float('nan'))
    
    # 向量化风险分类
    last['RiskLevel'] = classify_risk_vectorized(last)
    
    logger.info(f"总车辆: {len(last)}")
    return df_all, last


def create_safety_boxplot(last: pd.DataFrame, config: Config) -> go.Figure:
    """创建安全指标箱线图 (2x2)
    
    Args:
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['PET (s)', 'TTC (s)', 'mTTC (s)', 'THW (s)'],
        vertical_spacing=0.12, horizontal_spacing=0.10
    )
    
    metrics = ['PET', 'TTC', 'mTTC', 'Time_Headway']
    for idx, metric in enumerate(metrics):
        r, c = idx // 2 + 1, idx % 2 + 1
        for beh, color in config.BEHAVIOR_COLORS.items():
            vals = last[last['Behavior'] == beh][metric].dropna()
            cutoff = vals.quantile(0.99)
            vals = vals[vals <= cutoff]
            
            fig.add_trace(go.Box(
                y=vals, name=beh, marker_color=color, line=dict(width=1.2),
                legendgroup=beh, showlegend=(idx == 0),
                hovertemplate=f'{beh}<br>%{{y:.1f}}<extra></extra>'
            ), row=r, col=c)
        
        fig.update_yaxes(gridcolor='#F0F0F0', row=r, col=c)
    
    fig.update_layout(
        title=dict(
            text='<b>安全指标分布对比</b><br><span style="font-size:13px;color:#666">左变道 vs 右变道 | 全部数据</span>',
            x=0.5, font=dict(size=16)
        ),
        height=650, template=config.TEMPLATE, boxmode='group',
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def create_ttc_evolution(df_all: pd.DataFrame, last: pd.DataFrame, config: Config) -> go.Figure:
    """创建 TTC 风险演变图
    
    Args:
        df_all: 全量数据
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['左变道 TTC 风险演变', '右变道 TTC 风险演变'],
        horizontal_spacing=0.10
    )
    
    behaviors = ['左变道', '右变道']
    for idx, beh in enumerate(behaviors):
        beh_df = df_all[df_all['Behavior'] == beh]
        
        for level, (dash, color) in enumerate(['solid', 'dash', 'dot']):
            risk_level = list(config.RISK_COLORS.keys())[level]
            color = config.RISK_COLORS[risk_level]
            
            ids = last[(last['Behavior'] == beh) & (last['RiskLevel'] == risk_level)]['ID']
            pivot = beh_df[beh_df['ID'].isin(ids)].pivot_table(
                index='FrameIdx', columns='ID', values='TTC', aggfunc='first'
            )
            mean_ttc = pivot.mean(axis=1)
            
            fig.add_trace(go.Scatter(
                x=pivot.index, y=mean_ttc, mode='lines',
                name=f'{risk_level} (n={pivot.shape[1]})',
                line=dict(color=color, width=2.5, dash=dash),
                legendgroup=f'{beh}_{risk_level}', showlegend=(idx == 0),
                hovertemplate=f'FrameIdx: %{{x}}<br>TTC: %{{y:.1f}}s<extra>{beh} {risk_level}</extra>'
            ), row=1, col=idx + 1)
        
        fig.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=idx + 1)
        fig.add_hline(y=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=idx + 1)
        fig.update_xaxes(title_text='变道前帧序号', gridcolor='#F0F0F0', row=1, col=idx + 1)
        fig.update_yaxes(title_text='TTC (s)', gridcolor='#F0F0F0', row=1, col=idx + 1)
    
    fig.update_layout(
        title=dict(
            text='<b>变道前 TTC 风险演变</b><br><span style="font-size:13px;color:#666">按风险等级分组 | 均值曲线</span>',
            x=0.5, font=dict(size=16)
        ),
        height=500, template=config.TEMPLATE,
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def create_scatter_and_distribution(last: pd.DataFrame, config: Config) -> go.Figure:
    """创建速度-TTC 散点图 + 风险分布
    
    Args:
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['速度 vs TTC (左变道)', '速度 vs TTC (右变道)', '风险等级分布'],
        column_widths=[0.33, 0.33, 0.34],
        horizontal_spacing=0.10
    )
    
    # 散点图
    for idx, beh in enumerate(['左变道', '右变道']):
        for level, (sym, color) in config.RISK_MARKERS.items():
            sub = last[(last['Behavior'] == beh) & (last['RiskLevel'] == level)]
            if len(sub) > 0:
                fig.add_trace(go.Scatter(
                    x=sub['Velocity_ms'], y=sub['TTC'], mode='markers',
                    name=f'{beh} {level}',
                    marker=dict(color=color, symbol=sym, size=8, line=dict(color='black', width=0.3)),
                    legendgroup=level, showlegend=(idx == 0),
                    customdata=sub['ID'], text=sub['PET'].round(1),
                    hovertemplate='ID: %{customdata}<br>速度: %{x:.1f} m/s<br>TTC: %{y:.1f}s<br>PET: %{text}s<extra></extra>'
                ), row=1, col=idx + 1)
        
        fig.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=idx + 1)
        fig.add_hline(y=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=idx + 1)
        fig.update_xaxes(title_text='速度 (m/s)', gridcolor='#F0F0F0', row=1, col=idx + 1)
        fig.update_yaxes(title_text='TTC (s)', gridcolor='#F0F0F0', row=1, col=idx + 1)
    
    # 风险分布堆叠柱状图
    scene_pairs = [('1-1', '左'), ('1-1', '右'), ('1-2', '左'), ('1-2', '右'), ('loc5', '左'), ('loc5', '右')]
    for level, color in config.RISK_COLORS.items():
        counts, labels = [], []
        for src, beh_short in scene_pairs:
            beh_full = '左变道' if beh_short == '左' else '右变道'
            cnt = len(last[(last['Source'] == src) & (last['Behavior'] == beh_full) & (last['RiskLevel'] == level)])
            counts.append(cnt)
            labels.append(f'{src} {beh_short}')
        
        fig.add_trace(go.Bar(
            name=level, x=labels, y=counts, marker_color=color,
            marker_line=dict(color='white', width=0.5),
            hovertemplate='%{x}<br>%{y} 辆<extra></extra>'
        ), row=1, col=3)
    
    fig.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=3)
    fig.update_xaxes(tickangle=-20, row=1, col=3)
    fig.update_layout(
        title=dict(text='<b>速度-风险关系 & 风险分布</b>', x=0.5, font=dict(size=16)),
        height=500, template=config.TEMPLATE, barmode='stack',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def create_gap_analysis(last: pd.DataFrame, config: Config) -> go.Figure:
    """创建缺口分析 + PET CDF + 高风险占比
    
    Args:
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['周边安全缺口', 'PET 累积分布', '各场景高风险占比'],
        column_widths=[0.40, 0.30, 0.30],
        horizontal_spacing=0.12
    )
    
    # 缺口分析
    dist_cols = ['LB_Dist', 'LF_Dist', 'B_Dist', 'RF_Dist', 'RB_Dist']
    dist_labels = ['左后方', '左前方', '正后方', '右前方', '右后方']
    
    for beh, color in config.BEHAVIOR_COLORS.items():
        means = []
        for c in dist_cols:
            high_risk = last[(last['Behavior'] == beh) & (last['RiskLevel'] == '高风险')]
            vals = high_risk[c].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
        
        fig.add_trace(go.Bar(
            name=f'{beh} 高风险', x=dist_labels, y=means, marker_color=color,
            marker_line=dict(color='white', width=0.5), opacity=0.85,
            hovertemplate=f'{beh} 高风险<br>%{{x}}: %{{y:.0f}} m<extra></extra>'
        ), row=1, col=1)
    
    fig.update_yaxes(title_text='平均距离 (m)', gridcolor='#F0F0F0', row=1, col=1)
    
    # PET CDF
    for beh, color in config.BEHAVIOR_COLORS.items():
        dash = 'solid' if beh == '左变道' else 'dash'
        vals = last[last['Behavior'] == beh]['PET'].dropna()
        vals = vals[vals > 0]
        sorted_v = np.sort(vals)
        cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        
        fig.add_trace(go.Scatter(
            x=sorted_v, y=cum, mode='lines', name=f'{beh} (n={len(vals)})',
            line=dict(color=color, dash=dash, width=2.5),
            hovertemplate=f'PET: %{{x:.1f}}s<br>累积: %{{y:.1%}}<extra>{beh}</extra>'
        ), row=1, col=2)
    
    fig.add_vline(x=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=2)
    fig.add_vline(x=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=2)
    fig.update_xaxes(title_text='PET (s)', gridcolor='#F0F0F0', row=1, col=2)
    fig.update_yaxes(title_text='累积概率', gridcolor='#F0F0F0', row=1, col=2)
    
    # 高风险占比
    pairs_full = [('loc1', '左变道', '#42A5F5'), ('loc1', '右变道', '#EF5350'),
                  ('loc5', '左变道', '#1565C0'), ('loc5', '右变道', '#B71C1C')]
    labels_hr = ['Location1\n左变道', 'Location1\n右变道', 'Location5\n左变道', 'Location5\n右变道']
    pcts, totals = [], []
    
    for st, beh, _ in pairs_full:
        s = last[(last['SourceType'] == st) & (last['Behavior'] == beh)]
        totals.append(len(s))
        pcts.append((s['RiskLevel'] == '高风险').mean() * 100)
    
    fig.add_trace(go.Bar(
        x=labels_hr, y=pcts,
        marker_color=[c for _, _, c in pairs_full],
        marker_line=dict(color='white', width=1),
        text=[f'{v:.1f}%' for v in pcts], textposition='outside',
        textfont=dict(size=13, color='black'),
        customdata=totals,
        hovertemplate='%{x}<br>高风险占比: %{y:.1f}%<br>总车辆: %{customdata}<extra></extra>'
    ), row=1, col=3)
    
    fig.update_yaxes(title_text='高风险占比 (%)', gridcolor='#F0F0F0', row=1, col=3)
    fig.update_layout(
        title=dict(text='<b>安全缺口 & 风险分布摘要</b>', x=0.5, font=dict(size=16)),
        height=480, template=config.TEMPLATE, barmode='group',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def create_location_comparison(last: pd.DataFrame, config: Config) -> go.Figure:
    """创建 Location1 vs Location5 对比图
    
    Args:
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['PET 分布 Location1 vs Location5', '风险等级 Location1 vs Location5'],
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.12
    )
    
    # PET 箱线图
    for st in ['loc1', 'loc5']:
        lbl = 'Location1' if st == 'loc1' else 'Location5'
        for beh, color in config.BEHAVIOR_COLORS.items():
            vals = last[(last['SourceType'] == st) & (last['Behavior'] == beh)]['PET'].dropna()
            cutoff = vals.quantile(0.99)
            vals = vals[vals <= cutoff]
            
            fig.add_trace(go.Box(
                y=vals, name=f'{lbl} {beh}', marker_color=color, line=dict(width=1.2),
                width=0.25, offsetgroup=beh,
                hovertemplate=f'{lbl} {beh}<br>PET: %{{y:.1f}}s<extra></extra>'
            ), row=1, col=1)
    
    fig.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.3, row=1, col=1)
    fig.update_yaxes(title_text='PET (s)', gridcolor='#F0F0F0', row=1, col=1)
    
    # 风险等级堆叠柱状图
    pairs_comp = [('loc1', '左变道'), ('loc1', '右变道'), ('loc5', '左变道'), ('loc5', '右变道')]
    labels_comp = ['loc1 左', 'loc1 右', 'loc5 左', 'loc5 右']
    bar_colors = ['#42A5F5', '#EF5350', '#1565C0', '#B71C1C']
    
    for level, color in config.RISK_COLORS.items():
        counts = []
        for st, beh in pairs_comp:
            cnt = len(last[(last['SourceType'] == st) & (last['Behavior'] == beh) & (last['RiskLevel'] == level)])
            counts.append(cnt)
        
        fig.add_trace(go.Bar(
            name=level, x=labels_comp, y=counts, marker_color=color,
            marker_line=dict(color='white', width=0.5),
            hovertemplate='%{x} %{fullData.name}<br>%{y} 辆<extra></extra>'
        ), row=1, col=2)
    
    fig.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=2)
    fig.update_xaxes(tickangle=-15, row=1, col=2)
    fig.update_layout(
        title=dict(text='<b>Location1 vs Location5 对比</b>', x=0.5, font=dict(size=16)),
        height=480, template=config.TEMPLATE, barmode='stack', boxmode='group',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def create_category_distribution(last: pd.DataFrame, config: Config) -> go.Figure:
    """创建安全指标分类分布图
    
    Args:
        last: 每辆车最后一帧数据
        config: 配置对象
        
    Returns:
        Plotly Figure 对象
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['PET 分类分布', 'TTC 分类分布'],
        horizontal_spacing=0.10
    )
    
    scene_pairs = [('1-1', '左变道'), ('1-1', '右变道'), ('1-2', '左变道'), ('1-2', '右变道'),
                   ('loc5', '左变道'), ('loc5', '右变道')]
    scene_lbls = ['1-1 左', '1-1 右', '1-2 左', '1-2 右', 'loc5 左', 'loc5 右']
    
    pet_order = ['no_follower', 'dangerous', 'cautious', 'safe']
    ttc_order = ['no_leader', 'dangerous', 'cautious', 'safe']
    pet_colors = {'no_follower': '#9E9E9E', 'dangerous': config.RED, 'cautious': '#FF9800', 'safe': '#4CAF50'}
    ttc_colors = {'no_leader': '#9E9E9E', 'dangerous': config.RED, 'cautious': '#FF9800', 'safe': '#4CAF50'}
    
    label_map = {
        'no_follower': '无后车', 'no_leader': '无前车',
        'dangerous': '危险', 'cautious': '谨慎', 'safe': '安全'
    }
    
    for cat_col, cat_order, colors, col_idx in [
        ('PET_cat', pet_order, pet_colors, 1),
        ('TTC_cat', ttc_order, ttc_colors, 2)
    ]:
        for cat in cat_order:
            counts = []
            for src, beh in scene_pairs:
                cnt = len(last[(last['Source'] == src) & (last['Behavior'] == beh) & (last[cat_col] == cat)])
                counts.append(cnt)
            
            fig.add_trace(go.Bar(
                name=label_map[cat], x=scene_lbls, y=counts,
                marker_color=colors[cat], marker_line=dict(color='white', width=0.5),
                hovertemplate=f'%{{x}}<br>{label_map[cat]}: %{{y}} 辆<extra></extra>'
            ), row=1, col=col_idx)
        
        fig.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=col_idx)
        fig.update_xaxes(tickangle=-20, row=1, col=col_idx)
    
    fig.update_layout(
        title=dict(
            text='<b>安全指标分类分布</b><br><span style="font-size:13px;color:#666">按场景分组的 PET / TTC 类别构成 | 灰色=无冲突对象</span>',
            x=0.5, font=dict(size=16)
        ),
        height=500, template=config.TEMPLATE, barmode='stack',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(t=80, b=40)
    )
    
    return fig


def generate_indicator_cards(last: pd.DataFrame, n_total: int) -> str:
    """生成指标卡 HTML
    
    Args:
        last: 每辆车最后一帧数据
        n_total: 总车辆数
        
    Returns:
        指标卡 HTML 字符串
    """
    left_pet_all = last[last['Behavior'] == '左变道']['PET'].dropna()
    right_pet_all = last[last['Behavior'] == '右变道']['PET'].dropna()
    left_hr = (last[last['Behavior'] == '左变道']['RiskLevel'] == '高风险').mean() * 100
    right_hr = (last[last['Behavior'] == '右变道']['RiskLevel'] == '高风险').mean() * 100
    
    return f'''
<div class="cards-row">
  <div class="card card-blue">
    <div class="card-number">{n_total}</div>
    <div class="card-label">总变道车辆</div>
    <div class="card-sub">loc1: {len(last[last.SourceType=="loc1"])} | loc5: {len(last[last.SourceType=="loc5"])}</div>
  </div>
  <div class="card card-red">
    <div class="card-number">{(last["RiskLevel"]=="高风险").mean()*100:.1f}%</div>
    <div class="card-label">总体高风险占比</div>
    <div class="card-sub">PET<2s 或 TTC<2s</div>
  </div>
  <div class="card card-blue">
    <div class="card-number">{left_hr:.1f}% <span style="font-size:16px;color:#666">vs</span> {right_hr:.1f}%</div>
    <div class="card-label">左变道 vs 右变道 高风险占比</div>
    <div class="card-sub">差异: {right_hr-left_hr:+.1f}pp</div>
  </div>
  <div class="card card-red">
    <div class="card-number">{(left_pet_all<2).mean()*100:.1f}% <span style="font-size:16px;color:#666">vs</span> {(right_pet_all<2).mean()*100:.1f}%</div>
    <div class="card-label">左变道 vs 右变道 PET<2s</div>
    <div class="card-sub">右变道 +{((right_pet_all<2).mean()-(left_pet_all<2).mean())*100:.1f}pp</div>
  </div>
</div>
'''


def generate_html_report(figures: List[go.Figure], indicator_cards: str, config: Config) -> None:
    """生成完整的 HTML 报告
    
    Args:
        figures: 图表列表
        indicator_cards: 指标卡 HTML
        config: 配置对象
    """
    charts_html = []
    for fig, div_id in zip(figures, ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']):
        charts_html.append(f'<div class="chart-container" id="{div_id}">')
        charts_html.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id))
        charts_html.append('</div>')
    
    full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>变道安全分析仪表盘</title>
<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: #f5f5f5; color: #333; padding: 20px;
  }}
  .header {{
    text-align: center; padding: 20px 0 10px;
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white; border-radius: 12px; margin-bottom: 20px;
  }}
  .header h1 {{ font-size: 28px; font-weight: 700; }}
  .header p {{ font-size: 14px; opacity: 0.85; margin-top: 4px; }}
  .cards-row {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin-bottom: 20px;
  }}
  .card {{
    background: white; border-radius: 10px; padding: 20px 16px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid #ccc;
  }}
  .card-blue {{ border-top-color: #2196F3; }}
  .card-red {{ border-top-color: #E53935; }}
  .card-number {{ font-size: 32px; font-weight: 700; color: #212121; line-height: 1.2; }}
  .card-label {{ font-size: 13px; font-weight: 600; color: #555; margin-top: 4px; }}
  .card-sub {{ font-size: 11px; color: #999; margin-top: 2px; }}
  .chart-container {{
    background: white; border-radius: 10px; padding: 10px 16px;
    margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  @media (max-width: 1000px) {{
    .cards-row {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>变道安全分析仪表盘</h1>
  <p>Location1 (1-1 / 1-2) + Location5 | 高精度轨迹数据 | 交互式可视化</p>
</div>
{indicator_cards}
{''.join(charts_html)}
</body>
</html>
'''
    
    with open(config.OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    logger.info(f"保存仪表盘到: {config.OUT_HTML}")
    logger.info(f"文件大小: {os.path.getsize(config.OUT_HTML) / 1024:.0f} KB")
    logger.info(f"✅ 完成! 在浏览器中打开: {config.OUT_HTML}")


def main() -> None:
    """主函数"""
    config = Config()
    
    try:
        # 加载数据
        df_all, last = load_data(config)
        n_total = len(last)
        
        # 生成图表
        logger.info("生成图表...")
        figures = [
            create_safety_boxplot(last, config),
            create_ttc_evolution(df_all, last, config),
            create_scatter_and_distribution(last, config),
            create_gap_analysis(last, config),
            create_location_comparison(last, config),
            create_category_distribution(last, config)
        ]
        
        # 生成指标卡
        indicator_cards = generate_indicator_cards(last, n_total)
        
        # 生成 HTML 报告
        generate_html_report(figures, indicator_cards, config)
        
    except Exception as e:
        logger.error(f"生成仪表盘失败: {e}")
        raise


if __name__ == "__main__":
    main()
