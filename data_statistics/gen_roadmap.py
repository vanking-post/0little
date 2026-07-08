"""
Research Roadmap - Compact layout, large bold text.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(17, 22), facecolor='white')
ax.set_xlim(0, 17)
ax.set_ylim(0, 22)
ax.axis('off')

C1 = '#e3f2fd'
C2 = '#fff3e0'
C3 = '#e8f5e9'
C4 = '#fce4ec'
C_TITLE = '#0d47a1'

def box(ax, x, y, w, h, text, color, fs=14, lw=2):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                    facecolor=color, edgecolor='#37474f', lw=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, color='#212121', fontweight='bold')

def arrow(ax, x1, y1, x2, y2, lw=4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#455a64', lw=lw, shrinkA=8, shrinkB=8))

def lbl(ax, x, y, text):
    ax.text(x, y, text, ha='left', va='center', fontsize=15, fontweight='bold',
            color='white', bbox=dict(boxstyle='round,pad=0.25', facecolor=C_TITLE, lw=1.5))

# Title
ax.text(8.5, 21.6, 'Research Roadmap: Cross-Location Lane Change Risk Prediction',
        ha='center', va='bottom', fontsize=20, fontweight='bold', color=C_TITLE)

# ─── Stage 1 ───
lbl(ax, 0.3, 20.6, '1 Data Acquisition')
box(ax, 1, 19.6, 15, 0.85,
    "CQSkyEyeX Drone Dataset | 5 Scenarios | Subgrade/Bridge/Merge (100-120km/h)\n"
    "30Hz->5Hz | 75-frame window | 1,574 lane change samples",
    C1, 12)

# ─── Stage 2 ───
lbl(ax, 0.3, 18.2, '2 SSM Design & Risk Labeling')
box(ax, 0.5, 16.3, 7.5, 1.6,
    "Five SSM Indicators: mTTC | THW | PET | F_ETTC | OL_PET\n"
    "Exponential Decay: exp(-v/k)  |  EWM Objective Weights",
    C2, 13)
box(ax, 9, 16.3, 7.5, 1.6,
    "Risk Score S_LC -> Three-Tier Labels\n"
    "Correction K = (v85/v0)^1.3\n"
    "High=378  Mid=561  Low=635",
    C1, 13)

# ─── Stage 3 ───
lbl(ax, 0.3, 14.0, '3 Feature Engineering')
box(ax, 1, 13.2, 15, 0.65,
    "15 Indicators x 4 Aggregates (mean/std/min/max) = 60-d Vector | Velocity Accel Distance Jerk TTC Headway",
    C3, 12)

# ─── Stage 4 ───
lbl(ax, 0.3, 11.4, '4 Model Training & Evaluation')
box(ax, 0.5, 8.9, 4.8, 2.2,
    "Four Models\n"
    "XGBoost (Tree) | RF (Tree)\n"
    "MLP (NN) | LSTM (RNN)\n\n"
    "SMOTE + Optuna (50 trials)",
    C4, 13)
box(ax, 5.8, 8.9, 5.0, 2.2,
    "Evaluation Protocol\n\n"
    "Cross-Location CV: 5-fold leave-one-out\n"
    "(real deployment simulation)\n\n"
    "Random 80/20 Split\n"
    "(same-distribution upper bound)",
    C2, 12)
box(ax, 11.3, 8.9, 5.2, 2.2,
    "Cross-Location F1 Results\n\n"
    "XGBoost  0.764 | RF  0.732\n"
    "MLP  0.660 | LSTM  0.463\n\n"
    "Subgrade F1~0.875\n"
    "Merge Area F1~0.630",
    C1, 12)

# ─── Stage 5 ───
lbl(ax, 0.3, 6.0, '5 Visualization & Enhancement Analysis')
box(ax, 0.5, 4.0, 7.5, 1.65,
    "Visualizations\n\n"
    "Aerial screenshots | Speed distributions\n"
    "SSM histograms | Ridgeline density\n"
    "Risk heatmap | F1 line chart | Radar\n"
    "PCA variance curve",
    C3, 13)
box(ax, 9, 4.0, 7.5, 1.65,
    "Enhancement Comparison (XGBoost)\n\n"
    "Baseline 0.734 | SMOTE 0.741\n"
    "Optuna 0.762 | ST Joint 0.764 << best\n\n"
    "PCA: LSTM drops 0.598 -> 0.509",
    C4, 13)

# ─── SHAP summaries ───
lbl(ax, 0.3, 2.0, '6 SHAP Interpretability')
box(ax, 0.5, 0.3, 5.0, 1.4,
    "Global Importance\n"
    "Time_Headway_min dominates\n"
    "(+1.251, 10x above #2)\n"
    "B_Dist series = second tier",
    C1, 12)
box(ax, 6.0, 0.3, 5.0, 1.4,
    "Single-Sample Waterfall\n"
    "XGBoost -> TTC (transient)\n"
    "RF -> Time_Headway (steady)\n"
    "=> Complementary analysis",
    C4, 12)
box(ax, 11.5, 0.3, 5.0, 1.4,
    "Cross-Location Stability\n"
    "Universal: Time_Headway (all folds)\n"
    "Conditional: B_Dist (drops Loc5)\n"
    "Speed gains importance at merge",
    C3, 11)

# ── Arrows ──
arrow(ax, 8.5, 19.4, 8.5, 18.6)
arrow(ax, 8.5, 16.0, 8.5, 14.4)
arrow(ax, 8.5, 13.0, 8.5, 11.8)
arrow(ax, 8.5, 8.6, 8.5, 6.4)
arrow(ax, 8.5, 3.8, 8.5, 2.4)

plt.tight_layout()
import os; os.makedirs('E:/0little/data_statistics/gen_roadmap_output', exist_ok=True)
fig.savefig(os.path.join('E:/0little/data_statistics', 'gen_roadmap_output', 'research_roadmap.png'), dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print('Done')
