import matplotlib
matplotlib.use('Agg') # Fixes RuntimeError: Invalid DISPLAY variable

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Crucial for expanding the scale
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#  bias stratification plot with temperature
MLMODEL = 'SRGAN' #'SRGAN' #-'CNN'
VARIABLE =  'pr' #'pr', 'tas'
YEAR = '2009'
title_number = {'SRGAN': ['(c)', '(d)'], 'CNN': ['(a)', '(b)']}
step_bin = {'tas': 5, 'pr': 50}
unit = {'tas': 'K', 'pr': 'mm/day'}
#SRGAN_SETUP='WTWPWOROG_TEST2YR' #'WTWPWOROG_TEST2YR' #'NTNPNOROG'  #'WTWPWOROG'
#ML_SETUP='WTWPNOROG' #'WTWPWOROG_TEST2YR' #'NTNPNOROG'  #'WTWPWOROG'
#ML_SETUP='NTNPNOROG_TEST2YR' 
#ML_SETUP='WTWPNOROG_TEST2YR' 
ML_SETUP='WTNPNOROG_TEST2YR' 

fontsize_def = 18
xlabel_def = {'tas': {'a': 'HCLIM3 temperature (K)', 'b': 'HCLIM3 temperature bin (K)'}, \
    'pr': {'a': 'HCLIM3 precipitation (mm/day)', 'b': 'HCLIM3 precipitation bin (mm/day)'}}
ylabel_def = {'tas': {'a': f'{MLMODEL} temperature (K)', 'b': f'Bias {MLMODEL} - HCLIM3 (K)'}, \
    'pr': {'a': f'{MLMODEL} precipitation (mm/day)', 'b': f'Bias {MLMODEL} - HCLIM3 (mm/day)'}}
threshold = {'tas': {'min': 200, 'max': 350}, \
    'pr': {'min': 0, 'max': 2000}}

# 1. Define Paths

VARNAME = {'tas': {'HCLIM': 'tas', 'CNN': 'test', 'SRGAN': 'tas'}, \
    'pr': {'HCLIM': 'pr', 'CNN': 'pr', 'SRGAN': 'pr'}}
#    'pr': {'HCLIM': 'pr', 'CNN': 'test', 'SRGAN': 'pr'}}
path_ref = {'tas' : '/nobackup/rossby27/users/sm_fuxwa/AI_data/Emilia_Romagna/3km/6hr/tas/', \
    'pr': '/nobackup/rossby27/users/sm_fuxwa/AI_data/Emilia_Romagna/3km/6hr/pr/'}
file_ref = {'tas': 'tas_3km_6hr_200001010000-200912311800.nc', \
    'pr': 'pr_3km_6hr_200001010300-200912312100.nc'}
unit_convert = {'tas': {'HCLIM': 1, 'CNN': 1, 'SRGAN': 1}, \
    'pr': {'HCLIM': 86400, 'CNN': 1, 'SRGAN': 86400} }

if MLMODEL == 'CNN':
    if ML_SETUP == 'NTNPNOROG':
        path_ml = {'tas': '/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_withSM_whus/', \
            'pr': '/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/'}
        file_ml = {'tas': 'simple_cnn_prediction_normalized_normal2009.nc', \
            'pr': 'cnn_prediction_pr_2009.nc'}
    elif ML_SETUP == 'WTWPNOROG':
        path_ml = {'tas': '/nobackup/rossby27/users/sm_yicwa/DATA_shared/AIES_revision_aug2026/TAS_withT/', \
            'pr': '/nobackup/rossby27/users/sm_yicwa/DATA_shared/AIES_revision_aug2026/PR_ERAINT/'}
        file_ml = {'tas': 'simple_cnn_prediction_normalized_normal2009_withT.nc', \
            'pr': 'simple_cnn_prediction_withP.nc'}

elif MLMODEL == 'SRGAN':
    file_ml = {'tas': 'predictant_ypred_1.nc', \
        'pr': 'predictant_ypred_1.nc'}
    if ML_SETUP == 'NTNPNOROG':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_scale_time_stdscaler_norog_gpufix_lnoise0.1_bs50_ERAI_atos_v2/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_pr_scale_time_stdscaler_np_norog_gpufix_bs50_ERAI_atos_v2/'}
    elif ML_SETUP == 'WTWPWOROG':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_scale_time_stdscaler_wt_worog_gpufix_bs50_ERAI_atos/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_pr_scale_time_stdscaler_wp_worog_gpufix_bs50_ERAI_atos/'}
    elif ML_SETUP == 'WTWPWOROG_TEST2YR':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna//SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsmto_ERAI_2003_2009_arrhenius/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_pr_wsmpo_v2_ERAI_2003_2009_arrhenius/'}
        file_ml = {'tas': 'predictant_ypred_2009.nc', \
            'pr': 'predictant_ypred_2009.nc'}
    elif ML_SETUP == 'WTWPWOROG_TILE_TEST2YR':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna//SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsmto_tile_ERAI_2003_2009_arrhenius/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_pr_wsmpo_tile_ERAI_2003_2009_arrhenius/'}
        file_ml = {'tas': 'predictant_ypred_2009.nc', \
            'pr': 'predictant_ypred_2009.nc'}
    elif ML_SETUP == 'NTNPNOROG_TEST2YR':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna//SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsm_ERAI_2003_2009_arrhenius/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_pr_wsm_tile_ERAI_2003_2009_arrhenius/'}
        file_ml = {'tas': 'predictant_ypred_2.nc', \
            'pr': 'predictant_ypred_2009.nc'}
    elif ML_SETUP == 'WTWPNOROG_TEST2YR':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna//SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsmt_ERAI_2003_2009_arrhenius/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_pr_wsmp_tile_ERAI_2003_2009_arrhenius/'}
        file_ml = {'tas': 'predictant_ypred_2.nc', \
            'pr': 'predictant_ypred_2009.nc'}
    elif ML_SETUP == 'WTNPNOROG_TEST2YR':
        path_ml = {'tas': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna//SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsmt_ERAI_2003_2009_arrhenius/', \
            'pr': '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_pr_wsm_corr_ERAI_2003_2009_arrhenius/'}
        file_ml = {'tas': 'predictant_ypred_2.nc', \
            'pr': 'predictant_ypred_2.nc'}


# 2. Load Datasets
ds_ref_full = xr.open_dataset(f'{path_ref[VARIABLE]}/{file_ref[VARIABLE]}')
ds_comp_full = xr.open_dataset(f'{path_ml[VARIABLE]}/{file_ml[VARIABLE]}')

print(ds_comp_full.time.min().values, ds_comp_full.time.max().values)

# 3. Filter for 2009
# Reference file has 10 years; we extract only 2009 to match the prediction file
if VARIABLE == 'tas':
    ds_ref_origi = ds_ref_full.sel(time=YEAR)
    ds_comp_origi = ds_comp_full.sel(time=YEAR)
elif VARIABLE == 'pr':
    ds_ref_origi = ds_ref_full.sel(time=YEAR).resample(time='1D').mean()
    ds_comp_origi = ds_comp_full.sel(time=YEAR).resample(time='1D').mean()

var_ref_min = ds_ref_origi[VARNAME[VARIABLE]['HCLIM']].min().values * unit_convert[VARIABLE]['HCLIM']
var_ref_max = ds_ref_origi[VARNAME[VARIABLE]['HCLIM']].max().values * unit_convert[VARIABLE]['HCLIM']
var_comp_min = ds_comp_origi[VARNAME[VARIABLE][MLMODEL]].min().values * unit_convert[VARIABLE][MLMODEL]
var_comp_max = ds_comp_origi[VARNAME[VARIABLE][MLMODEL]].max().values * unit_convert[VARIABLE][MLMODEL]
print ('ref min max=', var_ref_min, var_ref_max)
print ('comp min max=', var_comp_min, var_comp_max)
print ('unit_convert=', VARIABLE, MLMODEL, unit_convert[VARIABLE][MLMODEL])

# Coarsen from 3km to e.g. 15km (5x5 pixel blocks) using mean
ds_ref = ds_ref_origi #.coarsen(y=5, x=5, boundary='trim').mean()
ds_comp = ds_comp_origi #.coarsen(y=5, x=5, boundary='trim').mean()

print("ref shape:", ds_ref[VARNAME[VARIABLE]['HCLIM']].shape, ds_ref[VARNAME[VARIABLE]['HCLIM']].dims)
print("comp shape:", ds_comp[VARNAME[VARIABLE][MLMODEL]].shape, ds_comp[VARNAME[VARIABLE][MLMODEL]].dims)
print("ref time:", ds_ref.time.values[:5], ds_ref.time.values[-5:], len(ds_ref.time))
print("comp time:", ds_comp.time.values[:5] if 'time' in ds_comp.coords else "NO TIME COORD", 
      len(ds_comp.time) if 'time' in ds_comp.coords else ds_comp.dims)

print("ref y:", ds_ref.y.values[:5], ds_ref.y.values[-5:])
print("comp y:", ds_comp.y.values[:5], ds_comp.y.values[-5:])
print("ref x:", ds_ref.x.values[:5], ds_ref.x.values[-5:])
print("comp x:", ds_comp.x.values[:5], ds_comp.x.values[-5:])

# 4. Extract and Flatten
# Prediction files often use different variable names (e.g., 'test' or 'tas')
# Ensure 'test' is the correct variable name in both files
print('unit_convert HCLIM:', VARIABLE, MLMODEL, unit_convert[VARIABLE]['HCLIM'])
print('unit_convert:', VARIABLE, MLMODEL, unit_convert[VARIABLE][MLMODEL])

ref_vals = ds_ref[VARNAME[VARIABLE]['HCLIM']].values.flatten() * unit_convert[VARIABLE]['HCLIM']
comp_vals = ds_comp[VARNAME[VARIABLE][MLMODEL]].values.flatten() * unit_convert[VARIABLE][MLMODEL]

mask = ~np.isnan(ref_vals) & ~np.isnan(comp_vals) #& \
#    (ref_vals > threshold[VARIABLE]['min']) & \
#    (ref_vals < threshold[VARIABLE]['max']) & \
#    (comp_vals > threshold[VARIABLE]['min']) & \
#    (comp_vals < threshold[VARIABLE]['max'])

ref_vals, comp_vals = ref_vals[mask], comp_vals[mask]

# Calculate Bias
bias_vals = comp_vals - ref_vals

# 5. Create DataFrame for processing
df = pd.DataFrame({
    'ref': ref_vals,
    'comp': comp_vals,
    'bias': bias_vals
}).dropna()

if VARIABLE == 'pr':
    limits = [
        #threshold[VARIABLE]['min'], #min(df['ref'].min(), df['comp'].min()),
        #threshold[VARIABLE]['max'], #max(df['ref'].max(), df['comp'].max())
        min(df['ref'].min(), df['comp'].min()),
        max(df['ref'].max(), df['comp'].max())
    ]
elif VARIABLE == 'tas':
    limits = [
        min(df['ref'].min(), df['comp'].min()),
        max(df['ref'].max(), df['comp'].max())
    ]

print('limits:', df['ref'].min(), df['comp'].min(), df['ref'].max(), df['comp'].max())
print('df ref.min()', df['ref'].min())
print('df ref.max()', df['ref'].max())
print('df comp.min()', df['comp'].min())
print('df comp.max()', df['comp'].max())

#bins = np.arange(np.floor(df['ref'].min()), np.ceil(df['ref'].max()), step_bin[VARIABLE])
if VARIABLE == 'pr':
    # Custom non-linear bins tailored for precipitation intensity
    bins = [0, 0.1, 1, 2.5, 5, 10, 20, 35, 50, 75, 100, 150, 250, 500, 1000]
else:
    bins = np.arange(np.floor(df['ref'].min()), np.ceil(df['ref'].max()), step_bin[VARIABLE])

df['bin'] = pd.cut(df['ref'], bins=bins)

# Group bias by bins
bin_groups = [group['bias'].values for name, group in df.groupby('bin', observed=True)]
labels = [str(name) for name, group in df.groupby('bin', observed=True)]

# Print list lengths
print(df['bin'])
print(f"bin_groups: {bin_groups}")
print(f"labels:     {labels}")
print(f"Length of bin_groups: {len(bin_groups)}")
print(f"Length of labels:     {len(labels)}")

# Check individual group array shapes if needed
for i, (lbl, grp) in enumerate(zip(labels, bin_groups)):
    print(
        f"Bin {i} [{lbl}]: {len(grp)} samples | Type: {type(grp)} | Shape:"
        f" {getattr(grp, 'shape', 'N/A')}"
    )

# Add statistics.
r2 = r2_score(df['ref'], df['comp'])
rmse = np.sqrt(mean_squared_error(df['ref'], df['comp']))
mae = mean_absolute_error(df['ref'], df['comp'])
mean_bias = (df['comp'] - df['ref']).mean()   # or ref - comp, just be consistent with your boxplot sign convention
print(f"R2 = {r2:.4f}", f"RMSE = {rmse:.4f}", f"MAE = {mae:.4f}", f"Mean Bias = {mean_bias:.4f}")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot A: Reference vs. Comparison (1:1 Density Plot)
# --------------------------------------------------
#hb = ax1.hexbin(df['ref'], df['comp'], gridsize=50, cmap='YlGnBu', mincnt=1)
hb = ax1.hexbin(df['ref'], df['comp'], gridsize=100, cmap='viridis', 
                norm=LogNorm(), \
                mincnt=1)
cbar = fig.colorbar(hb, ax=ax1) #, label='Count')
cbar.set_label('Point Density (Log Scale)', fontsize=int(fontsize_def - 4))

"""
ax1.scatter(
    df['ref'], df['comp'],
    s=1,  # Set small marker size to prevent huge overlapping dots
    alpha=0.1,  # Set transparency so dense regions appear darker
    color='blue',  # Choose a scatter color
)
"""
# Add 1:1 Lin8
ax1.plot(limits, limits, color='red', linestyle='--', label='Perfect agreement (1:1)')
ax1.set_xlabel(xlabel_def[VARIABLE]['a'], fontsize=fontsize_def)
ax1.set_ylabel(ylabel_def[VARIABLE]['a'], fontsize=fontsize_def)
#ax1.set_title('Scatter Plot: Ref vs. Comp (2009)')
#ax1.set_title(f'{title_number[MLMODEL][0]} 1:1 Density Plot: Expanded Outlier Visibility (Year {YEAR})', fontsize=fontsize_def)
ax1.set_title(f'{title_number[MLMODEL][0]} 1:1 Density plot for {MLMODEL}', fontsize=fontsize_def)
#ax1.legend(fontsize=int(fontsize_def - 4))
ax1.legend(
    loc='upper right',
    bbox_to_anchor=(0.97, 0.98),  # (x_coordinate, y_coordinate)
    fontsize=int(fontsize_def - 4),
    framealpha=0.9,  # Slightly opaque background so plot points don't obscure text
)

#stats_text = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} {unit[VARIABLE]}\nMAE = {mae:.2f} {unit[VARIABLE]}'
stats_text = f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} {unit[VARIABLE]}'
ypos = {'tas': 0.03, 'pr': 0.80}
ax1.text(0.94, ypos[VARIABLE], stats_text, transform=ax1.transAxes,
         fontsize=int(fontsize_def - 4), verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
)

ax1.tick_params(axis='both', which='major', labelsize=int(fontsize_def - 4)) 
ax1.grid(True, alpha=0.3)

"""
ax1.set_xscale('symlog', linthresh=1)
ax1.set_yscale('symlog', linthresh=1)
custom_ticks = [0, 0.1, 1, 2.5, 5, 10, 20, 35, 50, 75, 100, 150, 250, 500, 800]
ax1.set_xticks(custom_ticks)
ax1.set_yticks(custom_ticks)
ax1.set_xticklabels([str(t) for t in custom_ticks], rotation=45, fontsize=int(fontsize_def-6))
ax1.set_yticklabels([str(t) for t in custom_ticks], fontsize=int(fontsize_def-6))
ax1.set_xlim(left=-0.5)
ax1.set_ylim(bottom=-0.5)
"""

# Plot B: Stratified Bias Boxplot
# --------------------------------------------------
ax2.boxplot(bin_groups, tick_labels=labels, patch_artist=True,
            #whis=(5, 95),   # whiskers at 5th/95th percentile instead of 1.5*IQR
            #showfliers=False,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'),
            flierprops=dict(marker='o', markersize=2, markerfacecolor='gray',
                             markeredgecolor='none', alpha=0.15))

ax2.axhline(0, color='black', linestyle='--')

if VARIABLE == 'pr':
    # symlog y-axis =====
    ax2.set_yscale('symlog', linthresh=1)   # linear within ±1, log beyond that

    #  custom y-ticks for readable symlog labels =====
    all_ticks = [-4000, -2000, -1000, -500, -200, -100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000 ]
    # Keep only ticks that fall within [-max_val, max_val]
    custom_yticks = [t for t in all_ticks if abs(t) <= df['comp'].max()]

    #if MLMODEL == 'CNN':
    #    custom_yticks = [ -1000, -500, -200, -100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100, 200, 500, 1000 ]
    #elif MLMODEL == 'SRGAN':
    #    custom_yticks = [  -500, -200, -100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100, 200, 500]
    ax2.set_yticks(custom_yticks)
    ax2.set_yticklabels([str(t) for t in custom_yticks])

ax2.set_xlabel(xlabel_def[VARIABLE]['b'], fontsize=fontsize_def)
ax2.set_ylabel(ylabel_def[VARIABLE]['b'], fontsize=fontsize_def)
#3ax2.set_title(f'{title_number[MLMODEL][1]} Bias Distribution Stratified by Temperature', fontsize=fontsize_def)
ax2.set_title(f'{title_number[MLMODEL][1]} Bias distribution for {MLMODEL}', fontsize=fontsize_def)
ax2.tick_params(axis='y', which='major', labelsize=int(fontsize_def-4)) #
plt.setp(ax2.get_xticklabels(), rotation=25)
ax2.grid(axis='y', alpha=0.3)

fig_name = f'bias_stratification_{VARIABLE}_{MLMODEL}_{ML_SETUP}_{YEAR}.png'

plt.tight_layout()
fig_outdir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/bias_stratification/'
plt.savefig(f'{fig_outdir}/{fig_name}', dpi=300) #bias_stratification_SRGAN_NTNOROG_{VARIABLE}_{MLMODEL}_2009.png', dpi=300)
print(f"Analysis complete. Figure saved as 'bias_stratification_2009.pn10'")
