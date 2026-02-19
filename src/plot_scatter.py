import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import stats_tools
import crop_domain
matplotlib.use('Agg')  # use non-interactive backend


# Load the NetCDF files

#test_date = "20050601T1200"
#test_date = "20050801T1200"
#test_date = "20050601T0000"
#test_date = "20050601"
#test_date = 'JJA 2005'
#test_date = 'JJA 2003'
test_date = '20030815T1200' 
#experiment = 'HCLIM'
experiment = 'CNN'
#experiment = 'SRGAN'
#experiment = 'SRGAN_TAS_WSM_SCALETIME_GPUFIX_NSTD0.03_BS50_ERAI'
#experiment = 'SRGAN_TAS_WSM_SCALETIMESAVED_GPUFIX_BS50_DLR1E-5_ERAI'
#experiment = 'SRGAN_T_WSMT_SCALETIME_BS50_LAMB01_ERAI_pred_2009_v2'

# for paper
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_stdscaler_gpufix_bs50_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_stdscaler_gpufix_bs50_ERAI_atos_v2/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_stdscaler_gpufix_bs50_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_save_bs50_val0.1_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_val0.1_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_stdscaler_gpufix_lnoise0.1_bs50_2003_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_stdscaler_gpufix_lnoise0.1_bs50_ERAI_pred_atos/"

# other tests
#basedir = f"/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_ECEARTH_T_withSM_whus/"
#tas_ds = xr.open_dataset(basedir+f"03-inference_comp/simple_cnn_prediction_normalized_{test_date}.nc")

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_val0.1_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_with_sm_val0.1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_val0.1_pred_ERAI_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_with_sm_ERAI_corrlamda0_val0.1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_with_sm_ERAI_corrlamda0_val0.1_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_scale_time_bs50_wsm_ERAI_val0.1_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_scale_time_bs50_wsm_ERAI_val0.1_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_ERAI_val0.1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_space_bs50_lamb0.1_ERAI_val0.1_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_ERAI_val0.1_v1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_space_bs50_lamb0.1_ERAI_val0.1_v1_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_ERAI_val0.1_v2_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_space_bs50_lamb0.1_ERAI_val0.1_v2_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_ERAI_val0.1_v2_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_ERAI_val0.1_v2_pred_2009_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_bs50_lamb0.1_dr0.1_ERAI_val0.1_atos/"
#basedir1 = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_scale_time_val0.1_batch50_lamb0.1_wtm_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_bs50_val0.1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_bs50_val0.1_v2_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_bs50_ERAI_val0.1_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_bs50_ERAI_val0.1_pred_v2_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_space_bs50_ERAI_val0.1_pred_v2_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_wsmt_scale_time_bs50_ERAI_val0.1_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_wsmt_scale_space_bs50_ERAI_val0.1_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_save_bs50_ERAI_val0.1_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_bs50_ERAI_val0.1_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_save_bs50_lambda0.1_ERAI_val0.1_atos/"
#basedir1 = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_bs50_lambda0.1_ERAI_val0.1_pred_atos/"


#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tsm_wsmt_scale_time_save_bs60_val0.1_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tsm_wsmt_scale_time_presaved_bs60_val0.1_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tsm_wsmt_scale_time_presaved_bs60_lambda0.1_ERAI_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_lambda0.1_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_save_bs50_dr0.1_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_dr0.1_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_save_bs50_lambda0.1_dr0.1_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_lambda0.1_dr0.1_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_save_bs50_lambda0.1_dr0.2_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_lambda0.1_dr0.2_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_lr1e-3_ERAI_pred_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmt_scale_time_presaved_bs50_lr1e-5_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_stdscaler_gpufix_dlr1e-5_bs100_2003_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_stdscaler_gpufix_dlr1e-5_bs50_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH50_tas_wsm_scale_time_stdscaler_gpufix_bs50_2003_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_stdscaler_gpufix_bs50_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_mnxscaler_gpufix_bs50_ERAI_pred_atos/"

#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_stdscaler_gpufix_nstd0.03_bs50_2003_ERAI_atos/"
#basedir = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_presaved_stdscaler_gpufix_nstd0.03_bs50_ERAI_pred_atos/"



if 'HCLIM' in experiment:
    x_file = f"predictant_ytest_1.nc"
elif 'SRGAN'in experiment:
    x_file = f"predictant_ypred_1.nc"
y_file = f"predictor_1.nc"

# CNN
if 'CNN' in experiment:
    basedir = f"/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_withSM_whus/"
    if 'JJA' in test_date:
        x_file = f"{basedir}simple_cnn_prediction_normalized_normal2009.nc"
        y_file = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsm_scale_time_stdscaler_gpufix_nstd0.03_bs50_2003_ERAI_atos/predictor_1.nc"
    else:
        x_file = f"{basedir}simple_cnn_prediction_normalized_20030815T1200.nc"
        y_file = f"{basedir}training_singleday/JJA2003_20030815T1200_mrsol_whus_time.nc"

#dir_fuxing_org = f'/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/ICHEC-EC-EARTH/3km/6hr/mrsol/'
#y_file = f"mrsol_3km_6hr_199501010000-200512311800.nc"
#y_file = f'training_data_fuxing_ecearth_withSM/combined_12km_6hr_199501010000-200512311800_withmrsol.nc'


if 'SRGAN' in experiment or 'HCLIM' in experiment:
    var_names = {'var1':'mrsol', 'var2':'tas'}
elif 'CNN' in experiment:
    var_names = {'var1':'mrsol', 'var2':'test'}
if 'SRGAN' in experiment:
    number_def = '(b)'
elif 'HCLIM' in experiment:
    number_def = '(a)'
elif 'CNN' in experiment:
    number_def = '(c)'
outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/scatter/"
label_def = {'xlabel': "Soil Moisture at Top 1cm (m3/m3)", 'ylabel': "2-m Air Temperature (K)"}
#title_def = f"{number_def} {experiment} {var_names['var2']} vs. {var_names['var1']} ({test_date})"
title_def = f"{number_def} {experiment}"

# --- Define the bounding box for Northern Italy ---
lat_min = 44.0
lat_max = 45.5
lon_min = 7.0
lon_max = 12.0

# --------------------
if 'CNN' in experiment:
    var2_ds = xr.open_dataset(x_file) 
    var1_ds = xr.open_dataset(y_file)
else:
    var2_ds = xr.open_dataset(basedir + x_file) 
    var1_ds = xr.open_dataset(basedir + y_file)
print("-1 var2_ds=", var2_ds)
print("-1 var1_ds=", var1_ds)

# --- Replace 0 with NaN everywhere in the Dataset ---
var1_ds = var1_ds.where(var1_ds != 0 and var1_ds < 1e20 )

# --- regrid variable from coarse to fine resolution ---
var1_np = var1_ds[var_names['var1']].values 
var1_np_refined = stats_tools.upsample_2d_array(var1_np, upscale_factor = 4)

var1_ds = xr.Dataset(
    data_vars={
        # 1. The main refined data variable
        'mrsol': (('time', 'y', 'x'), var1_np_refined),
        # 2. Add high-resolution 'lon' as a Data Variable
        'lon': (('y', 'x'), var2_ds['lon'].values), 
        # 3. Add high-resolution 'lat' as a Data Variable
        'lat': (('y', 'x'), var2_ds['lat'].values)
    },
    coords={
        # Only 1D grid coordinates remain in the 'coords' dictionary
        'time': var1_ds['time'], 
        'x': var2_ds['x'],           # High-resolution 1D 'x' coordinate
        'y': var2_ds['y'],          # High-resolution 1D 'y' coordinate
    }
) 

print("0 var2_ds=", var2_ds)
print("0 var1_ds=", var1_ds)


# --- Mask var2_ds using NaNs from refined var1_ds ---
# Extract the raw boolean data array from mrsol
mask_data = var1_ds[var_names['var1']].notnull().values

# Extract only the 1D coordinates (time, x, y) from the target dataset (tas_ds)
dimension_coords = {
    dim: var2_ds.coords[dim] 
    for dim in var2_ds[var_names['var2']].dims # 'time', 'y', 'x'
}

aligned_mask = xr.DataArray(
    mask_data,
    coords=dimension_coords, # Use ONLY the 1D coordinates for alignment
    dims=var2_ds[var_names['var2']].dims   # Ensure dimensions are named correctly
)

# Extract the data variable(s) you want to mask from tas_ds
var2_only = var2_ds[var_names['var2']].to_dataset() # Create a temporary Dataset with only 'tas'

# Apply the mask to the temporary dataset (tas_only)
# This results in a masked data variable 'tas' with 1D coordinates.
var2_mask_ds = var2_only.where(aligned_mask)

# Re-attach the original 2D lat/lon Data Variables from tas_ds
# We use the original tas_ds['lat'] and tas_ds['lon'] DataArrays,
# which preserve the (y, x) dimensions.
var2_mask_ds = var2_mask_ds.assign({
    'lat': var2_ds['lat'],
    'lon': var2_ds['lon']
})

print("1 var1_ds =", type(var1_ds), np.max(var1_ds[var_names['var1']]), np.min(var1_ds[var_names['var1']]),) # var1_ds[var_names['var1']])
print("1 var2_ds =", type(var2_ds), np.max(var2_ds[var_names['var2']]), np.min(var2_ds[var_names['var2']]),) # var2_ds[var_names['var2']])


# --- Crop over a predefined domain ---
var2_crop_ds = crop_domain.crop_latlon_box(var2_mask_ds, lat_min, lat_max, lon_min, lon_max)
var1_crop_ds = crop_domain.crop_latlon_box(var1_ds, lat_min, lat_max, lon_min, lon_max)
print("2 var1_crop_ds =", type(var1_crop_ds), np.max(var1_crop_ds[var_names['var1']]), np.min(var1_crop_ds[var_names['var1']]), ) #var1_crop_ds[var_names['var1']])
print("2 var2_crop_ds =", type(var2_crop_ds), np.max(var2_crop_ds[var_names['var2']]), np.min(var2_crop_ds[var_names['var2']]), ) #var2_crop_ds[var_names['var2']])

# Extract tas and mrsol (replace variable names if they are different)
var2 = var2_crop_ds[var_names['var2']]  # shape: (time, lat, lon) or (lat, lon)
var1 = var1_crop_ds[var_names['var1']]  # shape: (time, lat, lon) or (lat, lon)

# Ensure spatial dimensions match

print('var1 all ', var1.shape, np.max(var1), np.min(var1))
print('var2 all ', var2.shape, np.max(var2), np.min(var2))
if 'JJA' in test_date:
    var2 = var2.where(var2['time'].dt.season == "JJA", drop=True)
    var1 = var1.where(var1['time'].dt.season == "JJA", drop=True)
    var2, var1 = xr.align(var2, var1, join='inner')

print('var1 3d', var1.shape, np.max(var1), np.min(var1))
print('var2 3d', var2.shape, np.max(var2), np.min(var2))


# --- Average over space ---
spatial_dims = [d for d in var2.dims if d not in ["time"]]
var2 = var2.mean(dim=spatial_dims) #, skipna=True)
var1 = var1.mean(dim=spatial_dims) #, skipna=True)
print('spatial_dims', spatial_dims)
print('var1 ave', var1.shape, np.max(var1), np.min(var1))
print('var2 ave', var2.shape, np.max(var2), np.min(var2))


# Flatten the data
var2_flat = var2.values.flatten()
var1_flat = var1.values.flatten()
print('var2_flat:', np.shape(var2_flat), np.max(var2_flat), np.min(var2_flat))
print('var1_flat:', np.shape(var1_flat), np.max(var1_flat), np.min(var1_flat))


# --- Fit line (1st degree polynomial) ---
slope, intercept = np.polyfit(var1_flat, var2_flat, 1)
y_fit = slope * var1_flat + intercept
print(f'slope:{slope}')


# --- Create scatter plot ---
plt.figure(figsize=(6, 6))
plt.scatter(var1_flat, var2_flat, alpha=0.3, s=16)
plt.scatter(var1_flat[0], var2_flat[0], alpha=1, s=20,color="black")
plt.plot(var1_flat, y_fit, color="blue", label=f"Fit: y = {slope:.3f}x + {intercept:.2f}")
plt.legend()
plt.xlabel(label_def['xlabel'])
plt.ylabel(label_def['ylabel'])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(title_def)
plt.grid(True)
plt.tight_layout()
#plt.show()

combined_test_date = '_'.join(test_date.split()) if " " in test_date else test_date
plt.savefig(f"{outdir_fig}/Scatter_{experiment}_{var_names['var1']}_{var_names['var2']}_{combined_test_date}.png", dpi=300, bbox_inches='tight')

