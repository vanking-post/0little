"""
Vehicle Trajectory Map — XY trajectory plots with lane lines for 5 locations
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, ast, re

DATA_DIR = 'E:/0little'
OUT_DIR = os.path.join(DATA_DIR, 'data_statistics', 'plot_trajectories_output')
os.makedirs(OUT_DIR, exist_ok=True)

LOCATIONS = [f'location{i}' for i in range(1, 6)]

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_lane_coeffs_from_xlsx(xlsx_path):
    """Load lane coefficients from lane_coeffs.xlsx for locations 1-4."""
    df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    coeffs = {}
    for _, row in df.iterrows():
        src = row['where']  # e.g. '1-1', '3-2'
        raw_str = row['lane_coeffs']
        # Parse the multi-line string of coefficient lists
        # Format: each line is [a5,a4,a3,a2,a1,a0], separated by ",\n"
        lines = [l.strip() for l in raw_str.replace('\n', ',').split(',') if l.strip().startswith('[')]
        # Actually parse more carefully: use regex to find all [...] groups
        matches = re.findall(r'\[[^\]]+\]', raw_str)
        parsed = []
        for m in matches:
            parsed.append(ast.literal_eval(m))
        coeffs[src] = np.array(parsed)  # shape (n_curves, 6)
    return coeffs


def get_lane_coeffs_for_location(loc_key, xlsx_coeffs):
    """Get lane coefficients for a location, averaged across all sources."""
    if loc_key == 'location5':
        csv_path = os.path.join(DATA_DIR, loc_key, 'lane_coeffs.csv')
        if not os.path.exists(csv_path):
            return None, None
        df_lc = pd.read_csv(csv_path)
        first_src = df_lc['where'].iloc[0]
        src_df = df_lc[df_lc['where'] == first_src]
        half = len(src_df) // 2
        dir1 = src_df[src_df['direction'] == 1][['a5','a4','a3','a2','a1','a0']].values
        dir2 = src_df[src_df['direction'] == 2][['a5','a4','a3','a2','a1','a0']].values
        return dir1, dir2
    else:
        # Average coefficients across all sources for this location
        loc_num = loc_key.replace('location', '')
        src_keys = [k for k in xlsx_coeffs if k.startswith(f'{loc_num}-')]
        if not src_keys:
            return None, None
        # Collect all dir1 and dir2 curves from all sources
        all_dir1, all_dir2 = [], []
        for sk in src_keys:
            arr = xlsx_coeffs[sk]  # (8, 6) or (n_curves, 6)
            if len(arr) < 2:
                continue
            half = len(arr) // 2
            all_dir1.append(arr[:half])
            all_dir2.append(arr[half:])
        if not all_dir1 or not all_dir2:
            return None, None
        # Average across sources per curve index
        dir1 = np.mean(np.array(all_dir1), axis=0)  # (n_curves_per_dir, 6)
        dir2 = np.mean(np.array(all_dir2), axis=0)
        return dir1, dir2


def plot_trajectories_for_location(loc_key, df_smooth, dir1_coeffs, dir2_coeffs, max_vehicles=150):
    """Plot vehicle trajectories for one location with lane lines using smoothed data."""
    import warnings
    warnings.filterwarnings('ignore')

    if df_smooth is None or len(df_smooth) == 0:
        return

    fig, ax = plt.subplots(figsize=(20, 10))

    # Collect XY ranges
    all_x, all_y = df_smooth['X'].values, df_smooth['Y'].values
    for direction, cmap_name in [(1, 'Reds'), (2, 'Blues')]:
        dir_df = df_smooth[df_smooth['Direction'] == direction]
        vids = dir_df['ID'].unique()
        if len(vids) > max_vehicles:
            vids = np.random.choice(vids, max_vehicles, replace=False)
        cmap = plt.cm.get_cmap(cmap_name)
        for i, vid in enumerate(vids):
            grp = dir_df[dir_df['ID'] == vid].sort_values('Frame')
            xs, ys = grp['X'].values, grp['Y'].values
            if len(xs) <= 1:
                continue
            # Stay in saturated range (0.5-0.95) to avoid pale/washed-out lines
            shade = 0.5 + 0.45 * (i / max(len(vids) - 1, 1))
            ax.plot(xs, ys, color=cmap(shade), linewidth=0.4, alpha=0.5, zorder=2)

    # Axis limits from data
    x_min, x_max = np.percentile(all_x, [0.5, 99.5])
    y_min, y_max = np.percentile(all_y, [0.5, 99.5])
    pad_x = (x_max - x_min) * 0.03
    pad_y = (y_max - y_min) * 0.05

    # Plot lane lines (solid dark lines, under trajectories)
    lane_kw = dict(color='#2c2c2c', linewidth=2.0, linestyle='-', alpha=0.8, zorder=1)
    if dir1_coeffs is not None and len(dir1_coeffs) > 0:
        x_lane = np.linspace(max(x_min, 0), min(x_max, 420), 500)
        for coeff in dir1_coeffs:
            y_line = np.polyval(coeff, x_lane)
            ax.plot(x_lane, y_line, **lane_kw)

    if dir2_coeffs is not None and len(dir2_coeffs) > 0:
        x_lane = np.linspace(max(x_min, 0), min(x_max, 420), 500)
        for coeff in dir2_coeffs:
            y_line = np.polyval(coeff, x_lane)
            ax.plot(x_lane, y_line, **lane_kw)

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_max + pad_y, y_min - pad_y)  # Y inverted: smaller Y at top

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=plt.cm.Reds(0.6), alpha=0.7, label='Upstream'),
                      Patch(color=plt.cm.Blues(0.6), alpha=0.7, label='Downstream'),
                      Line2D([0], [0], color='#2c2c2c', linewidth=2.5, label='Lane')]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=11, framealpha=0.9)

    loc_label = loc_key.replace('location', 'Loc')
    ax.set_title(f'{loc_label} — Vehicle Trajectories', fontsize=15, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=13)
    ax.set_ylabel('Y (m)', fontsize=13)
    ax.set_box_aspect(0.5)
    ax.grid(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, f'trajectory_{loc_key}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] trajectory_{loc_key}.png')


def main():
    print("Loading lane coefficients from xlsx...")
    xlsx_path = os.path.join(DATA_DIR, 'lane_coeffs.xlsx')
    xlsx_coeffs = load_lane_coeffs_from_xlsx(xlsx_path) if os.path.exists(xlsx_path) else {}

    for loc in LOCATIONS:
        print(f"\nProcessing {loc}...")

        # Load smoothed full trajectory data
        smooth_path = os.path.join(DATA_DIR, loc, 'trajectory_full_smoothed.csv')
        if not os.path.exists(smooth_path):
            print(f'  [SKIP] No smoothed data found')
            continue
        df = pd.read_csv(smooth_path, low_memory=False)
        print(f'  Vehicles: {df["ID"].nunique()}, Rows: {len(df)},'
              f' Dir1={len(df[df["Direction"]==1])}, Dir2={len(df[df["Direction"]==2])}')

        # Get lane coefficients
        dir1_c, dir2_c = get_lane_coeffs_for_location(loc, xlsx_coeffs)

        # Plot (fewer trajectories for loc1-4 to reduce visual clutter)
        n_max = 75 if loc != 'location5' else 150
        plot_trajectories_for_location(loc, df, dir1_c, dir2_c, max_vehicles=n_max)

    print(f"\nAll done! Charts saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
