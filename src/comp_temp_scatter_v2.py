import matplotlib
matplotlib.use('Agg') # Fixes DISPLAY variable error

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Crucial for expanding the scale

# 1. Load and Filter
path_ds = '/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_withSM_whus/'
dir_fuxing_org = '/nobackup/rossby27/users/sm_fuxwa/AI_data/Emilia_Romagna/3km/6hr/tas/'

ds_ref = xr.open_dataset(dir_fuxing_org + 'tas_3km_6hr_200001010000-200912311800.nc').sel(time='2009')
ds_comp = xr.open_dataset(path_ds + 'simple_cnn_prediction_normalized_normal2009.nc').sel(time='2009')

# 2. Extract, Flatten, and Clean
ref = ds_ref['tas'].values.flatten()
comp = ds_comp['test'].values.flatten()

mask = ~np.isnan(ref) & ~np.isnan(comp)
ref, comp = ref[mask], comp[mask]

# 3. Plotting with Expanded Log Scale
plt.figure(figsize=(9, 8))

# Use LogNorm to make low-density outliers much more visible
# mincnt=1 ensures we don't color empty hexagonal bins
hb = plt.hexbin(ref, comp, gridsize=100, cmap='viridis', 
                norm=LogNorm(), mincnt=1)

# Add Colorbar with Log labels
cb = plt.colorbar(hb)
cb.set_label('Point Density (Log Scale)', fontsize=12)

# 4. Add 1:1 Line
limits = [min(ref.min(), comp.min()), max(ref.max(), comp.max())]
plt.plot(limits, limits, color='red', linestyle='--', linewidth=2, label='Perfect Agreement (1:1)')

# 5. Labels and Formatting
plt.xlabel('Reference Temperature (tas) [K]', fontsize=12)
plt.ylabel('CNN Prediction [K]', fontsize=12)
plt.title('1:1 Density Plot: Expanded Outlier Visibility (Year 2009)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('density_1to1_expanded.png', dpi=300)
print("Plot saved as density_1to1_expanded.png")

