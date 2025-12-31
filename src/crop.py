import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import stats_tools
from scipy.interpolate import griddata
matplotlib.use('Agg')  # use non-interactive backend


def crop_latlon_box(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Crop an xarray Dataset to a lat-lon bounding box.
    Supports both 1D and 2D lat/lon coordinates.

    Parameters:
        ds       : xarray.Dataset
        lat_min  : float
        lat_max  : float
        lon_min  : float
        lon_max  : float

    Returns:
        Cropped xarray.Dataset
    """

    # Handle 1D lat/lon
    if "lat" in ds.coords and "lon" in ds.coords:
        if ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
            return ds.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )

    # Handle 2D lat/lon
    if "lat" in ds and "lon" in ds and ds["lat"].ndim == 2 and ds["lon"].ndim == 2:
        mask = (
            (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max) &
            (ds["lon"] >= lon_min) & (ds["lon"] <= lon_max)
        )
        return ds.where(mask, drop=True)



# Load the NetCDF files
var_names = {'tas':'tas', 'mrsol':'mrsol'}
outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs"

test_date = "20050601T1200"
#test_date = "20050801T1200"
#test_date = "20050601T0000"
#test_date = "20050601"
basedir = f"/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/01-rampal2021-unet/Emulator_ECEARTH_T_withSM_whus/"
#tas_ds = xr.open_dataset(basedir+f"03-inference_comp/simple_cnn_prediction_normalized_{test_date}.nc")
tas_ds = xr.open_dataset(f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_val0.1_pred_atos/predictant_ypred_1.nc") 
#tas_ds = xr.open_dataset(f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_with_sm_val0.1_atos/predictant_ypred_1.nc") 

#dir_fuxing_org = f'/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/cropped/ICHEC-EC-EARTH/3km/6hr/mrsol/'
#y_file = f"mrsol_3km_6hr_199501010000-200512311800.nc"
#y_file = f'training_data_fuxing_ecearth_withSM/combined_12km_6hr_199501010000-200512311800_withmrsol.nc'
#mrsol_ds = xr.open_dataset(basedir+y_file)
y_file = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_mrsol_val0.1_pred_atos/predictor_1.nc"
#y_file = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_with_sm_val0.1_atos/predictor_1.nc"
mrsol_ds = xr.open_dataset(y_file)

# Extract tas and mrsol (replace variable names if they are different)
#tas = tas_ds['test']  # shape: (time, lat, lon) or (lat, lon)
#mrsol = mrsol_ds['mrsol']  # shape: (time, lat, lon) or (lat, lon)

# --- Define the bounding box for Northern Italy ---
lat_min = 44.0
lat_max = 45.5
lon_min = 7.0
lon_max = 12.0

print("-1 tas_ds=", tas_ds)
print("-1 mrsol_ds=", mrsol_ds)
#mrsol_ds, tas_ds = xr.align(mrsol_ds, tas_ds, join="inner")


msrol_np = mrsol_ds[var_names['mrsol']].values 
mrsol_np_refined = stats_tools.upsample_2d_array(msrol_np, upscale_factor = 4)

mrsol_ds = xr.Dataset(
    data_vars={
        # 1. The main refined data variable
        'mrsol': (('time', 'y', 'x'), mrsol_np_refined),
        # 2. Add high-resolution 'lon' as a Data Variable
        'lon': (('y', 'x'), tas_ds['lon'].values), 
        # 3. Add high-resolution 'lat' as a Data Variable
        'lat': (('y', 'x'), tas_ds['lat'].values)
    },
    coords={
        # Only 1D grid coordinates remain in the 'coords' dictionary
        'time': mrsol_ds['time'], 
        'x': tas_ds['x'],           # High-resolution 1D 'x' coordinate
        'y': tas_ds['y'],          # High-resolution 1D 'y' coordinate
    }
)

print("0 mrsol_ds lat =", mrsol_ds["lat"])
print("0 mrsol_ds lon =", mrsol_ds["lon"])
print("0 tas_ds=", tas_ds)
print("0 mrsol_ds=", mrsol_ds)


tas_ds = crop_latlon_box(tas_ds, lat_min, lat_max, lon_min, lon_max)
mrsol_ds = crop_latlon_box(mrsol_ds, lat_min, lat_max, lon_min, lon_max)
print("1 mrsol_ds lat =", mrsol_ds["lat"])
print("1 mrsol_ds lon =", mrsol_ds["lon"])

# Extract tas and mrsol (replace variable names if they are different)
tas = tas_ds[var_names['tas']]  # shape: (time, lat, lon) or (lat, lon)
mrsol = mrsol_ds[var_names['mrsol']]  # shape: (time, lat, lon) or (lat, lon)
print("mrsol =", mrsol)
print("tas =", tas)

# Ensure spatial dimensions match
#tas = tas.where(tas['time'].dt.season == "JJA", drop=True)
#mrsol = mrsol.where(mrsol['time'].dt.season == "JJA", drop=True)
#tas, mrsol = xr.align(tas, mrsol, join='inner')

print('mrsol shape', mrsol.shape)
print('tas shape', tas.shape)
#tas_ds = stats_tools.upsample_2d_array(tas_ds, upscale_factor = 4)
#mrsol = mrsol.interp_like(tas)

# Mask out NaNs
#mask = np.isfinite(tas) & np.isfinite(mrsol)
#tas = tas[mask]
#mrsol = mrsol[mask]
#tas = tas.where(mrsol.notnull())

# average over space
spatial_dims = [d for d in tas.dims if d not in ["time"]]
tas = tas.mean(dim=spatial_dims) #, skipna=True)
mrsol = mrsol.mean(dim=spatial_dims) #, skipna=True)

print(mrsol.shape)
print(tas.shape)

# Flatten the data
tas_flat = tas.values.flatten()
mrsol_flat = mrsol.values.flatten()
print('tas_flat:', np.shape(tas_flat), tas_flat)
print('mrsol_flat:', np.shape(mrsol_flat), mrsol_flat)

# Mask out NaNs
#mask = np.isfinite(tas_flat) & np.isfinite(mrsol_flat)
#tas_flat = tas_flat[mask]
#mrsol_flat = mrsol_flat[mask]

# Create scatter plot
#plt.figure(figsize=(6, 6))
#plt.scatter(mrsol_flat, tas_flat, alpha=0.3, s=10)
#plt.xlabel("Soil Moisture (mrsol)")
#plt.ylabel("Surface Air Temperature (tas)")
#plt.title("Scatter Plot of tas vs. mrsol (JJA 2005)")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

#plt.savefig("my_figure.png", dpi=300, bbox_inches='tight')


# --- Fit line (1st degree polynomial) ---
slope, intercept = np.polyfit(mrsol_flat, tas_flat, 1)
y_fit = slope * mrsol_flat + intercept
print(f'slope:{slope}')

# Create scatter plot
plt.figure(figsize=(6, 6))

plt.scatter(mrsol_flat, tas_flat, alpha=0.3, s=16)
plt.scatter(mrsol_flat[0], tas_flat[0], alpha=1, s=20,color="black")
plt.plot(mrsol_flat, y_fit, color="blue", label=f"Fit: y = {slope:.3f}x + {intercept:.2f}")
plt.xlabel("Soil Moisture (mrsol)")
plt.ylabel("Surface Air Temperature (tas)")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title(f"Scatter Plot of tas ({test_date}) vs. mrsol")
plt.grid(True)
plt.tight_layout()
#plt.show()

plt.savefig(f"{outdir_fig}/mean_{test_date}.png", dpi=300, bbox_inches='tight')

