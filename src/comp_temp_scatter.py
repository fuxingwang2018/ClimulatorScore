import matplotlib
matplotlib.use('Agg') # Fixes RuntimeError: Invalid DISPLAY variable

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Paths
path_ds = '/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_withSM_whus/'
dir_fuxing_org = '/nobackup/rossby27/users/sm_fuxwa/AI_data/Emilia_Romagna/3km/6hr/tas/'

# 2. Load Datasets
ds_ref_full = xr.open_dataset(dir_fuxing_org + 'tas_3km_6hr_200001010000-200912311800.nc')
ds_comp_full = xr.open_dataset(path_ds + 'simple_cnn_prediction_normalized_normal2009.nc')

# 3. Filter for 2009
# Reference file has 10 years; we extract only 2009 to match the prediction file
ds_ref = ds_ref_full.sel(time='2009')
ds_comp = ds_comp_full.sel(time='2009')

# 4. Extract and Flatten
# Prediction files often use different variable names (e.g., 'test' or 'tas')
# Ensure 'test' is the correct variable name in both files
ref_vals = ds_ref['tas'].values.flatten()
comp_vals = ds_comp['test'].values.flatten()

# Calculate Bias
bias_vals = ref_vals - comp_vals

# 5. Create DataFrame for processing
df = pd.DataFrame({
    'ref': ref_vals,
    'comp': comp_vals,
    'bias': bias_vals
}).dropna()

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot A: Reference vs. Comparison (1:1 Density Plot)
# --------------------------------------------------
hb = ax1.hexbin(df['ref'], df['comp'], gridsize=50, cmap='YlGnBu', mincnt=1)
fig.colorbar(hb, ax=ax1, label='Count')

# Add 1:1 Line
limits = [
    min(df['ref'].min(), df['comp'].min()),
    max(df['ref'].max(), df['comp'].max())
]
ax1.plot(limits, limits, color='red', linestyle='--', label='1:1 Line')

ax1.set_xlabel('Reference Temperature (tas) [K]')
ax1.set_ylabel('CNN Prediction [K]')
ax1.set_title('Scatter Plot: Ref vs. Comp (2009)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot B: Stratified Bias Boxplot
# --------------------------------------------------
bins = np.arange(np.floor(df['ref'].min()), np.ceil(df['ref'].max()), 5)
df['temp_bin'] = pd.cut(df['ref'], bins=bins)

# Group bias by bins
bin_groups = [group['bias'].values for name, group in df.groupby('temp_bin', observed=True)]
labels = [str(name) for name, group in df.groupby('temp_bin', observed=True)]

ax2.boxplot(bin_groups, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'))

ax2.axhline(0, color='black', linestyle='--')
ax2.set_xlabel('Reference Temperature Bin [K]')
ax2.set_ylabel('Bias (Ref - Comp) [K]')
ax2.set_title('Bias Distribution Stratified by Temperature')
plt.setp(ax2.get_xticklabels(), rotation=45)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_validation_2009.png', dpi=300)
print("Analysis complete. Figure saved as 'temperature_validation_2009.png'")
