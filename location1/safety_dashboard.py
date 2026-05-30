"""
变道安全分析 - 交互式 Plotly 仪表盘
读取已有的变道 CSV 数据，生成交互式 HTML 报告
"""
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== 配置 ====================
DATA_DIR = r"E:\0little\location1"
LOC5_DIR = r"E:\0little\read\CQSkyEyedata5\location5t"
OUT_HTML = r"E:\0little\location1\safety_dashboard.html"

FILES = {
    ('1-1',  '左变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-1_left_change.csv'),
    ('1-1',  '右变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-1_right_change.csv'),
    ('1-2',  '左变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-2_left_change.csv'),
    ('1-2',  '右变道', 'loc1'): os.path.join(DATA_DIR, 'traffic_1-2_right_change.csv'),
    ('loc5', '左变道', 'loc5'): os.path.join(LOC5_DIR, 'traffic_left_change.csv'),
    ('loc5', '右变道', 'loc5'): os.path.join(LOC5_DIR, 'traffic_right_change.csv'),
}

RISK = {'高风险': '#E53935', '中风险': '#FF9800', '低风险': '#4CAF50'}
BLUE, RED = '#2196F3', '#E53935'

# ==================== 辅助函数 ====================
def _ensure_safety_cats(df):
    """确保 DataFrame 包含安全分类列，若没有则自动计算"""
    if 'TTC_cat' not in df.columns:
        df['TTC_cat'] = 'safe'
        df.loc[df['TTC'] == 0, 'TTC_cat'] = 'no_leader'
        df.loc[(df['TTC'] > 0) & (df['TTC'] < 2), 'TTC_cat'] = 'dangerous'
        df.loc[(df['TTC'] >= 2) & (df['TTC'] < 5), 'TTC_cat'] = 'cautious'
    if 'PET_cat' not in df.columns:
        df['PET_cat'] = 'safe'
        pi = np.isinf(df['PET'].values) | pd.isna(df['PET'].values)
        df.loc[pi, 'PET_cat'] = 'no_follower'
        df.loc[(df['PET'] > 0) & (df['PET'] < 2), 'PET_cat'] = 'dangerous'
        df.loc[(df['PET'] >= 2) & (df['PET'] < 5), 'PET_cat'] = 'cautious'
    if 'OL_PET_cat' not in df.columns and 'OL_PET' in df.columns:
        df['OL_PET_cat'] = 'safe'
        olp_inv = np.isinf(df['OL_PET'].values) | pd.isna(df['OL_PET'].values)
        df.loc[olp_inv, 'OL_PET_cat'] = 'no_follower'
        df.loc[(df['OL_PET'] > 0) & (df['OL_PET'] < 2), 'OL_PET_cat'] = 'dangerous'
        df.loc[(df['OL_PET'] >= 2) & (df['OL_PET'] < 5), 'OL_PET_cat'] = 'cautious'
    return df


# ==================== 数据加载 ====================
print("加载数据...")
frames = []
for (section, behavior, src_type), fpath in FILES.items():
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    df['Source'] = section
    df['Behavior'] = behavior
    df['SourceType'] = src_type
    df['FrameIdx'] = df.groupby('ID').cumcount()
    df = _ensure_safety_cats(df)
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
df_all['Velocity_ms'] = df_all['Velocity']

last = df_all.groupby('ID').last().reset_index()
for c in ['TTC', 'mTTC', 'PET', 'Time_Headway']:
    if c in last.columns:
        last[c] = last[c].replace([float('inf'), -float('inf')], float('nan'))
for dc in ['LB_Dist', 'LF_Dist', 'B_Dist', 'RB_Dist', 'RF_Dist']:
    if dc in last.columns:
        last[dc] = last[dc].replace(0, float('nan'))


def classify_risk(row):
    tc = row.get('TTC_cat', 'safe')
    pc = row.get('PET_cat', 'safe')
    if tc == 'dangerous' or pc == 'dangerous':
        return '高风险'
    if tc == 'cautious' or pc == 'cautious':
        return '中风险'
    return '低风险'


last['RiskLevel'] = last.apply(classify_risk, axis=1)
n_total = len(last)

# ==================== 创建各独立图表 ====================
TEMPLATE = 'plotly_white'
HEIGHT = 420

print(f"总车辆: {n_total}, 生成图表...")

# ---- Fig1: 安全指标箱线图 (2x2) ----
fig1 = make_subplots(rows=2, cols=2,
                     subplot_titles=['PET (s)', 'TTC (s)', 'mTTC (s)', 'THW (s)'],
                     vertical_spacing=0.12, horizontal_spacing=0.10)
for idx, metric in enumerate(['PET', 'TTC', 'mTTC', 'Time_Headway']):
    r, c = idx // 2 + 1, idx % 2 + 1
    for beh, color in [('左变道', BLUE), ('右变道', RED)]:
        vals = last[last['Behavior'] == beh][metric].dropna()
        cutoff = vals.quantile(0.99)
        vals = vals[vals <= cutoff]
        fig1.add_trace(go.Box(
            y=vals, name=beh, marker_color=color, line=dict(width=1.2),
            legendgroup=beh, showlegend=(idx == 0),
            hovertemplate=f'{beh}<br>%{{y:.1f}}<extra></extra>'),
            row=r, col=c)
    fig1.update_yaxes(gridcolor='#F0F0F0', row=r, col=c)
fig1.update_layout(
    title=dict(text='<b>安全指标分布对比</b><br><span style="font-size:13px;color:#666">左变道 vs 右变道 | 全部数据</span>',
               x=0.5, font=dict(size=16)),
    height=650, template=TEMPLATE, boxmode='group',
    legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ---- Fig2: TTC 风险演变 ----
fig2 = make_subplots(rows=1, cols=2,
                     subplot_titles=['左变道 TTC 风险演变', '右变道 TTC 风险演变'],
                     horizontal_spacing=0.10)
for idx, beh in enumerate(['左变道', '右变道']):
    beh_df = df_all[df_all['Behavior'] == beh]
    for level, color, dash in [('高风险', RED, 'solid'), ('中风险', '#FF9800', 'dash'), ('低风险', '#4CAF50', 'dot')]:
        ids = last[(last['Behavior'] == beh) & (last['RiskLevel'] == level)]['ID']
        pivot = beh_df[beh_df['ID'].isin(ids)].pivot_table(
            index='FrameIdx', columns='ID', values='TTC', aggfunc='first')
        mean_ttc = pivot.mean(axis=1)
        fig2.add_trace(go.Scatter(
            x=pivot.index, y=mean_ttc, mode='lines', name=f'{level} (n={pivot.shape[1]})',
            line=dict(color=color, width=2.5, dash=dash),
            legendgroup=f'{beh}_{level}', showlegend=(idx == 0),
            hovertemplate=f'FrameIdx: %{{x}}<br>TTC: %{{y:.1f}}s<extra>{beh} {level}</extra>'),
            row=1, col=idx + 1)
    fig2.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=idx + 1)
    fig2.add_hline(y=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=idx + 1)
    fig2.update_xaxes(title_text='变道前帧序号', gridcolor='#F0F0F0', row=1, col=idx + 1)
    fig2.update_yaxes(title_text='TTC (s)', gridcolor='#F0F0F0', row=1, col=idx + 1)
fig2.update_layout(
    title=dict(text='<b>变道前 TTC 风险演变</b><br><span style="font-size:13px;color:#666">按风险等级分组 | 均值曲线</span>',
               x=0.5, font=dict(size=16)),
    height=500, template=TEMPLATE,
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ---- Fig3: 速度-TTC 散点图 + 风险分布 ----
fig3 = make_subplots(rows=1, cols=3,
                     subplot_titles=['速度 vs TTC (左变道)', '速度 vs TTC (右变道)', '风险等级分布'],
                     column_widths=[0.33, 0.33, 0.34],
                     horizontal_spacing=0.10)
markers = {'高风险': ('circle', RED), '中风险': ('diamond', '#FF9800'), '低风险': ('triangle-up', '#4CAF50')}
for idx, beh in enumerate(['左变道', '右变道']):
    for level, (sym, color) in markers.items():
        sub = last[(last['Behavior'] == beh) & (last['RiskLevel'] == level)]
        if len(sub) > 0:
            fig3.add_trace(go.Scatter(
                x=sub['Velocity_ms'], y=sub['TTC'], mode='markers',
                name=f'{beh} {level}', marker=dict(color=color, symbol=sym, size=8,
                                                   line=dict(color='black', width=0.3)),
                legendgroup=level, showlegend=(idx == 0),
                customdata=sub['ID'], text=sub['PET'].round(1),
                hovertemplate='ID: %{customdata}<br>速度: %{x:.1f} m/s<br>TTC: %{y:.1f}s<br>PET: %{text}s<extra></extra>'),
                row=1, col=idx + 1)
    fig3.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=idx + 1)
    fig3.add_hline(y=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=idx + 1)
    fig3.update_xaxes(title_text='速度 (m/s)', gridcolor='#F0F0F0', row=1, col=idx + 1)
    fig3.update_yaxes(title_text='TTC (s)', gridcolor='#F0F0F0', row=1, col=idx + 1)

# 风险分布堆叠柱状图
scene_pairs = [('1-1', '左'), ('1-1', '右'), ('1-2', '左'), ('1-2', '右'), ('loc5', '左'), ('loc5', '右')]
for level, color in [('高风险', RED), ('中风险', '#FF9800'), ('低风险', '#4CAF50')]:
    counts, labels = [], []
    for src, beh_short in scene_pairs:
        beh_full = '左变道' if beh_short == '左' else '右变道'
        cnt = len(last[(last['Source'] == src) & (last['Behavior'] == beh_full) & (last['RiskLevel'] == level)])
        counts.append(cnt)
        labels.append(f'{src} {beh_short}')
    fig3.add_trace(go.Bar(
        name=level, x=labels, y=counts, marker_color=color,
        marker_line=dict(color='white', width=0.5),
        hovertemplate='%{x}<br>%{y} 辆<extra></extra>'),
        row=1, col=3)
fig3.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=3)
fig3.update_xaxes(tickangle=-20, row=1, col=3)
fig3.update_layout(
    title=dict(text='<b>速度-风险关系 & 风险分布</b>',
               x=0.5, font=dict(size=16)),
    height=500, template=TEMPLATE, barmode='stack',
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ---- Fig4: 缺口分析 + PET CDF + 高风险占比 ----
fig4 = make_subplots(rows=1, cols=3,
                     subplot_titles=['周边安全缺口', 'PET 累积分布', '各场景高风险占比'],
                     column_widths=[0.40, 0.30, 0.30],
                     horizontal_spacing=0.12)

dist_cols = ['LB_Dist', 'LF_Dist', 'B_Dist', 'RF_Dist', 'RB_Dist']
dist_labels = ['左后方', '左前方', '正后方', '右前方', '右后方']
for beh, color, hatch in [('左变道', BLUE, '/'), ('右变道', RED, '\\')]:
    means = [last[(last['Behavior'] == beh) & (last['RiskLevel'] == '高风险')][c].dropna().mean()
             if len(last[(last['Behavior'] == beh) & (last['RiskLevel'] == '高风险')][c].dropna()) > 0 else 0
             for c in dist_cols]
    fig4.add_trace(go.Bar(
        name=f'{beh} 高风险', x=dist_labels, y=means, marker_color=color,
        marker_line=dict(color='white', width=0.5), opacity=0.85,
        hovertemplate=f'{beh} 高风险<br>%{{x}}: %{{y:.0f}} m<extra></extra>'),
        row=1, col=1)
fig4.update_yaxes(title_text='平均距离 (m)', gridcolor='#F0F0F0', row=1, col=1)

for beh, color, dash in [('左变道', BLUE, 'solid'), ('右变道', RED, 'dash')]:
    vals = last[last['Behavior'] == beh]['PET'].dropna()
    vals = vals[vals > 0]
    sorted_v = np.sort(vals)
    cum = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    fig4.add_trace(go.Scatter(
        x=sorted_v, y=cum, mode='lines', name=f'{beh} (n={len(vals)})',
        line=dict(color=color, dash=dash, width=2.5),
        hovertemplate=f'PET: %{{x:.1f}}s<br>累积: %{{y:.1%}}<extra>{beh}</extra>'),
        row=1, col=2)
fig4.add_vline(x=2, line_dash='dash', line_color='red', opacity=0.25, row=1, col=2)
fig4.add_vline(x=5, line_dash='dash', line_color='orange', opacity=0.25, row=1, col=2)
fig4.update_xaxes(title_text='PET (s)', gridcolor='#F0F0F0', row=1, col=2)
fig4.update_yaxes(title_text='累积概率', gridcolor='#F0F0F0', row=1, col=2)

pairs_full = [('loc1', '左变道', '#42A5F5'), ('loc1', '右变道', '#EF5350'),
              ('loc5', '左变道', '#1565C0'), ('loc5', '右变道', '#B71C1C')]
labels_hr = ['Location1\n左变道', 'Location1\n右变道', 'Location5\n左变道', 'Location5\n右变道']
pcts, totals = [], []
for st, beh, _ in pairs_full:
    s = last[(last['SourceType'] == st) & (last['Behavior'] == beh)]
    totals.append(len(s))
    pcts.append((s['RiskLevel'] == '高风险').mean() * 100)
fig4.add_trace(go.Bar(
    x=labels_hr, y=pcts,
    marker_color=[c for _, _, c in pairs_full],
    marker_line=dict(color='white', width=1),
    text=[f'{v:.1f}%' for v in pcts], textposition='outside',
    textfont=dict(size=13, color='black'),
    customdata=totals,
    hovertemplate='%{x}<br>高风险占比: %{y:.1f}%<br>总车辆: %{customdata}<extra></extra>'),
    row=1, col=3)
fig4.update_yaxes(title_text='高风险占比 (%)', gridcolor='#F0F0F0', row=1, col=3)

fig4.update_layout(
    title=dict(text='<b>安全缺口 & 风险分布摘要</b>',
               x=0.5, font=dict(size=16)),
    height=480, template=TEMPLATE, barmode='group',
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ---- Fig5: Location1 vs Location5 对比 ----
fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=['PET 分布 Location1 vs Location5', '风险等级 Location1 vs Location5'],
                     column_widths=[0.45, 0.55],
                     horizontal_spacing=0.12)

for i, st in enumerate(['loc1', 'loc5']):
    lbl = 'Location1' if st == 'loc1' else 'Location5'
    for beh, color, offset in [('左变道', BLUE, -0.15), ('右变道', RED, 0.15)]:
        vals = last[(last['SourceType'] == st) & (last['Behavior'] == beh)]['PET'].dropna()
        cutoff = vals.quantile(0.99)
        vals = vals[vals <= cutoff]
        fig5.add_trace(go.Box(
            y=vals, name=f'{lbl} {beh}', marker_color=color, line=dict(width=1.2),
            width=0.25, offsetgroup=beh,
            hovertemplate=f'{lbl} {beh}<br>PET: %{{y:.1f}}s<extra></extra>'),
            row=1, col=1)
fig5.add_hline(y=2, line_dash='dash', line_color='red', opacity=0.3, row=1, col=1)
fig5.update_yaxes(title_text='PET (s)', gridcolor='#F0F0F0', row=1, col=1)

pairs_comp = [('loc1', '左变道'), ('loc1', '右变道'), ('loc5', '左变道'), ('loc5', '右变道')]
labels_comp = ['loc1 左', 'loc1 右', 'loc5 左', 'loc5 右']
bar_colors = ['#42A5F5', '#EF5350', '#1565C0', '#B71C1C']
for level, color in [('高风险', RED), ('中风险', '#FF9800'), ('低风险', '#4CAF50')]:
    counts = []
    for st, beh in pairs_comp:
        cnt = len(last[(last['SourceType'] == st) & (last['Behavior'] == beh) & (last['RiskLevel'] == level)])
        counts.append(cnt)
    fig5.add_trace(go.Bar(
        name=level, x=labels_comp, y=counts, marker_color=color,
        marker_line=dict(color='white', width=0.5),
        hovertemplate='%{x} %{fullData.name}<br>%{y} 辆<extra></extra>'),
        row=1, col=2)
fig5.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=2)
fig5.update_xaxes(tickangle=-15, row=1, col=2)

fig5.update_layout(
    title=dict(text='<b>Location1 vs Location5 对比</b>',
               x=0.5, font=dict(size=16)),
    height=480, template=TEMPLATE, barmode='stack', boxmode='group',
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ---- Fig6: 安全指标分类分布 ----
fig6 = make_subplots(rows=1, cols=2,
                     subplot_titles=['PET 分类分布', 'TTC 分类分布'],
                     horizontal_spacing=0.10)
scene_pairs = [('1-1', '左变道'), ('1-1', '右变道'), ('1-2', '左变道'), ('1-2', '右变道'),
               ('loc5', '左变道'), ('loc5', '右变道')]
scene_lbls = ['1-1 左', '1-1 右', '1-2 左', '1-2 右', 'loc5 左', 'loc5 右']
pet_order = ['no_follower', 'dangerous', 'cautious', 'safe']
ttc_order = ['no_leader', 'dangerous', 'cautious', 'safe']
pet_colors = {'no_follower': '#9E9E9E', 'dangerous': RED, 'cautious': '#FF9800', 'safe': '#4CAF50'}
ttc_colors = {'no_leader': '#9E9E9E', 'dangerous': RED, 'cautious': '#FF9800', 'safe': '#4CAF50'}

for cat_col, cat_order, colors, col_idx in [('PET_cat', pet_order, pet_colors, 1),
                                              ('TTC_cat', ttc_order, ttc_colors, 2)]:
    for cat in cat_order:
        counts = []
        for src, beh in scene_pairs:
            cnt = len(last[(last['Source'] == src) & (last['Behavior'] == beh) & (last[cat_col] == cat)])
            counts.append(cnt)
        label_map = {'no_follower': '无后车', 'no_leader': '无前车',
                     'dangerous': '危险', 'cautious': '谨慎', 'safe': '安全'}
        fig6.add_trace(go.Bar(
            name=label_map[cat], x=scene_lbls, y=counts,
            marker_color=colors[cat], marker_line=dict(color='white', width=0.5),
            hovertemplate=f'%{{x}}<br>{label_map[cat]}: %{{y}} 辆<extra></extra>'),
            row=1, col=col_idx)
    fig6.update_yaxes(title_text='车辆数', gridcolor='#F0F0F0', row=1, col=col_idx)
    fig6.update_xaxes(tickangle=-20, row=1, col=col_idx)
fig6.update_layout(
    title=dict(text='<b>安全指标分类分布</b><br><span style="font-size:13px;color:#666">按场景分组的 PET / TTC 类别构成 | 灰色=无冲突对象</span>',
               x=0.5, font=dict(size=16)),
    height=500, template=TEMPLATE, barmode='stack',
    legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
    margin=dict(t=80, b=40))

# ==================== 组装为完整 HTML ====================
# 用 plotly.io 直接输出所有图表到单个 HTML
import plotly.io as pio

charts_html = []
for fig, div_id in [(fig1, 'fig1'), (fig2, 'fig2'), (fig3, 'fig3'), (fig4, 'fig4'), (fig5, 'fig5'), (fig6, 'fig6')]:
    charts_html.append(f'<div class="chart-container" id="{div_id}">')
    charts_html.append(pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id))
    charts_html.append('</div>')

# ---- 指标卡 HTML ----
left_pet_all = last[last['Behavior'] == '左变道']['PET'].dropna()
right_pet_all = last[last['Behavior'] == '右变道']['PET'].dropna()
left_hr = (last[last['Behavior'] == '左变道']['RiskLevel'] == '高风险').mean() * 100
right_hr = (last[last['Behavior'] == '右变道']['RiskLevel'] == '高风险').mean() * 100

cards_html = f'''
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
{cards_html}
{''.join(charts_html)}
</body>
</html>
'''

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"保存仪表盘到: {OUT_HTML}")
print(f"文件大小: {os.path.getsize(OUT_HTML) / 1024:.0f} KB")
print(f"✅ 完成! 在浏览器中打开: {OUT_HTML}")
