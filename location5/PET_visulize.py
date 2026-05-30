#对PET进行可视化分析
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import step06change

left_df,right_df = step06change.extract_lane_change_samples()
left_valid = left_df[left_df['PET'] != np.inf]
right_valid = right_df[right_df['PET'] != np.inf]

print(f"左变道 PET < 2秒比例: {(left_valid['PET'] < 2).mean():.1%}")
print(f"右变道 PET < 2秒比例: {(right_valid['PET'] < 2).mean():.1%}")
print(f"左变道 PET 最大值: {left_valid['PET'].max():.1f}")
print(f"右变道 PET 最大值: {right_valid['PET'].max():.1f}")
