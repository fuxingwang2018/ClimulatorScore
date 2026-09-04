import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plot_scatter, os

# -------------------------------------------------------------
# 1. Configuration & File Paths
# -------------------------------------------------------------
domain = {'lat_min': 44.0, 'lat_max': 45.5, 'lon_min': 7.0, 'lon_max': 12.0}

# Update these dictionary paths to match your actual files
models = {
    'HCLIM 12km': {
        #'predictor': 'predictor_HCLIM12km.nc',
        #'predictant': 'predictant_ypred_HCLIM12km.nc',
        'shortname': 'HCLIM12',
        'color': 'darkorange',
        'ls': '-.'
    },
    'HCLIM 3km': {
        #'predictor': 'predictor_HCLIM3km.nc',
        #'predictant': 'predictant_ypred_HCLIM3km.nc',
        'shortname': 'HCLIM3',
        'color': 'purple',
        'ls': ':'
    },
    'CNN 3km': {
        #'predictor': 'predictor.nc',
        #'predictant': 'predictant_ypred_CNN.nc',
        'shortname': 'CNN',
        'color': 'teal',
        'ls': '--'
    },
    'SRGAN 3km': {
        #'predictor': 'predictor.nc',
        #'predictant': 'predictant_ypred_SRGAN.nc',
        'shortname': 'SRGAN',
        'color': 'crimson',
        'ls': '-'
    },
}

experiment = 'ERAI 2003 Day 20030815T1200'

outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/pdp/"
os.makedirs(outdir_fig, exist_ok=True)

# Helper function to clean values
def clean_data(da):
    da = da.where(da != -9999.9)
    return da.where(da < 1e19)

# -------------------------------------------------------------
# 2. Process Datasets & Build 1D Line Plot
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Dictionary to hold 2D response arrays for spatial plotting later
delta_tas_all = {}
coords_all = {}

for name, cfg in models.items():
    print(f"Processing model: {name}...")
    basedir, x_file, y_file = \
        plot_scatter.get_file(cfg['shortname'], experiment)
    print('basedir, x_file, y_file', basedir, x_file, y_file)
    #ds_pred = xr.open_dataset(cfg['predictor'])
    #ds_ypred = xr.open_dataset(cfg['predictant'])
    if 'HCLIM 3km' in name:
        ds_pred = xr.open_dataset(f"{x_file}")
    else:
        ds_pred = xr.open_dataset(f"{basedir}/{x_file}")
    ds_ypred = xr.open_dataset(f"{basedir}/{y_file}")

    # Clean variables
    mrsol = clean_data(ds_pred['mrsol'])
    if 'CNN 3km' in name:
        ds_ypred = ds_ypred.isel(time=slice(0, 19))
        tas = clean_data(ds_ypred['test'])
    else:
        tas = clean_data(ds_ypred['tas'])

    #mrsol_raw = ds_pred['mrsol'].where(ds_pred['mrsol'] != -9999.9)
    #mrsol_raw = mrsol_raw.where(mrsol_raw < 1e19) # Handle 1.e+20f missing values

    #tas_raw = ds_ypred['tas'].where(ds_ypred['tas'] != -9999.9)
    #tas_raw = tas_raw.where(tas_raw < 1e19)

    # Create 2D sub-domain masks for Emilia-Romagna
    mask_pred = (
        (ds_pred['lat'] >= domain['lat_min']) & (ds_pred['lat'] <= domain['lat_max']) &
        (ds_pred['lon'] >= domain['lon_min']) & (ds_pred['lon'] <= domain['lon_max'])
    )
    mask_ypred = (
        (ds_ypred['lat'] >= domain['lat_min']) & (ds_ypred['lat'] <= domain['lat_max']) &
        (ds_ypred['lon'] >= domain['lon_min']) & (ds_ypred['lon'] <= domain['lon_max'])
    )

    # Apply masks
    mrsol_sub = mrsol.where(mask_pred)
    tas_sub = tas.where(mask_ypred)

    # Calculate 1D PDP curve (spatial means over domain per perturbation level)
    sm_levels = mrsol_sub.mean(dim=['y', 'x'], skipna=True).values
    pdp_1d = tas_sub.mean(dim=['y', 'x'], skipna=True).values

    # Plot line on 1D figure
    plt.plot(
        sm_levels, pdp_1d, 
        label=name, color=cfg['color'], linestyle=cfg['ls'], linewidth=2, marker='o', markersize=4
    )

    # Save data needed for 2D spatial sensitivity plot
    idx_min = np.argmin(sm_levels)
    idx_max = np.argmax(sm_levels)
    
    delta_tas_all[name] = tas.isel(time=idx_max) - tas.isel(time=idx_min)
    coords_all[name] = {'lon': ds_ypred['lon'].values, 'lat': ds_ypred['lat'].values}

    ds_pred.close()
    ds_ypred.close()

# Format and save 1D PDP plot
plt.title('1D Partial dependence plot: air temperature vs soil moisture\n(Emilia-Romagna sub-domain)', fontsize=12, fontweight='bold')
plt.xlabel('Soil moisture perturbation [m$^{3}$/m$^{3}$]', fontsize=11)
plt.ylabel('Regional average air temperature ($T_{2m}$) [K]', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(f'{outdir_fig}/PDP_1D_MultiModel_EmiliaRomagna.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 3. Build 2D Spatial Sensitivity Comparison (2x2 Panel)
# -------------------------------------------------------------
#fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True) #, constrained_layout=True)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharex=False, sharey=False, constrained_layout=True)
axes = axes.flatten()

# Determine global min and max for a unified colorbar
all_deltas = [d.values for d in delta_tas_all.values()]
vmin = np.nanmin([np.nanpercentile(d, 2) for d in all_deltas])
vmax = np.nanmax([np.nanpercentile(d, 98) for d in all_deltas])

count = 0
for i, (name, delta_tas) in enumerate(delta_tas_all.items()):
    letter = chr(ord('a') + count)
    ax = axes[i]
    ax.set_aspect('equal', adjustable='box')
    lon = coords_all[name]['lon']
    lat = coords_all[name]['lat']

    pcm = ax.pcolormesh(
        lon, lat, delta_tas.values,
        cmap='coolwarm', vmin=vmin, vmax=vmax, shading='auto'
    )
   
    ax.set_title(f'({letter}) {name}', fontsize=12, fontweight='bold')
    #ax.set_xlim(domain['lon_min'] - 0.5, domain['lon_max'] + 0.5)
    #ax.set_ylim(domain['lat_min'] - 0.5, domain['lat_max'] + 0.5)
    ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
    ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if i in [0, 1, 2, 3]:
        ax.set_xlabel('Longitude [°E]', fontsize=10)
    if i in [0]:
        ax.set_ylabel('Latitude [°N]', fontsize=10)
    count += 1

# Add shared colorbar
#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#fig.colorbar(pcm, cax=cbar_ax, label='$\Delta T_{2m}$ [K] ($\Theta_{max} - \Theta_{min}$)')
cbar = fig.colorbar(pcm, ax=axes, location='bottom', pad=0.04, shrink=0.6, aspect=30)
cbar.set_label(r'$\Delta T_{as}$ [K] ($\theta_{max} - \theta_{min}$)', fontsize=11)

#plt.suptitle('Spatial sensitivity ($\Delta T_{2m}$ to soil moisture perturbation)', fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(right=0.9, hspace=0.1, wspace=0.05)
plt.savefig(f'{outdir_fig}/PDP_2D_MultiModel_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Processing complete! Figures 'PDP_1D_MultiModel_EmiliaRomagna.png' and 'PDP_2D_MultiModel_Comparison.png' saved.")

