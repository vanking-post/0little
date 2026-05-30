"""
合并 location1 子文件变道数据 → traffic_left/right_change.csv
"""
import pandas as pd
import os

BASE = 'E:/0little/location1'
PARTS = {
    'left':  ['1-1_left', '1-2_left'],
    'right': ['1-1_right', '1-2_right'],
}

for side, parts in PARTS.items():
    dfs = []
    for p in parts:
        fp = os.path.join(BASE, f'traffic_{p}_change.csv')
        df = pd.read_csv(fp)
        print(f'  {p}: {len(df)} 行, {df["ID"].nunique()} 辆车, OL_PET={"OL_PET" in df.columns}')
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    out = os.path.join(BASE, f'traffic_{side}_change.csv')
    merged.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'  → traffic_{side}_change.csv: {len(merged)} 行, {merged["ID"].nunique()} 辆车\n')
