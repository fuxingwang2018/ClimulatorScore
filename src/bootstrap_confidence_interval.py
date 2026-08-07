import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import xarray as xr
import os
import get_time_index

# https://medium.com/data-science/calculating-confidence-interval-with-bootstrapping-872c657c058d

# ── Metric functions (unchanged) ─────────────────────────────────────────────

def metric_annual_mean_bias(reference, model_output):
    return np.mean(model_output - reference, axis=0) 
    #return np.mean(model_output, axis=0) - np.mean(reference, axis=0)

def metric_rmse(reference, model_output):
    return np.sqrt(np.mean((model_output - reference)**2, axis=0))

def metric_p99_bias(reference, model_output):
    return np.percentile(model_output, 99, axis=0) - np.percentile(reference, 99, axis=0)


# ── Bootstrap function (unchanged) ───────────────────────────────────────────

def bootstrap_metric(reference, model_output, metric_fn,
                     n_bootstrap=1000, ci=95, axis=0, seed=42):
    rng = np.random.default_rng(seed)
    n_time = reference.shape[axis]
    metric_obs = metric_fn(reference, model_output)
    boot_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_time, size=n_time)
        ref_boot = np.take(reference,    idx, axis=axis)
        mod_boot = np.take(model_output, idx, axis=axis)
        boot_metrics.append(metric_fn(ref_boot, mod_boot))
    boot_metrics = np.array(boot_metrics)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(boot_metrics, alpha,       axis=0)
    ci_upper = np.percentile(boot_metrics, 100 - alpha, axis=0)
    return metric_obs, ci_lower, ci_upper, boot_metrics


# ── 2D plotting helpers ───────────────────────────────────────────────────────

def plot_2d_metric_comparison(srgan_obs, cnn_obs,
                               srgan_lo, srgan_hi,
                               cnn_lo,   cnn_hi,
                               srgan_sig, cnn_sig, diff_sig,
                               metric_name, lon=None, lat=None,
                               save_path=None):
    """
    6-panel figure per metric:
      Row 1: observed metric map for SRGAN | CNN | difference (CNN - SRGAN)
      Row 2: CI width map for SRGAN        | CNN | significance of difference
    Stippling marks where CIs do NOT include zero (i.e. significant).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Bootstrap Analysis — {metric_name}  (95% CI, 1000 resamples)',
                 fontsize=14, fontweight='bold')

    diff_obs = cnn_obs - srgan_obs

    # Shared colormap limits (symmetric for bias/difference)
    abs_max_obs  = max(np.nanmax(np.abs(srgan_obs)), np.nanmax(np.abs(cnn_obs)))
    abs_max_diff = np.nanmax(np.abs(diff_obs))

    srgan_ci_width = srgan_hi - srgan_lo
    cnn_ci_width   = cnn_hi   - cnn_lo
    ci_max = max(np.nanmax(srgan_ci_width), np.nanmax(cnn_ci_width))

    def _pcolor(ax, data, cmap, vmin, vmax, title, lon, lat):
        if lon is not None and lat is not None:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=11)
        return im

    def _add_stippling_old(ax, mask, lon, lat, color='black', label='Significant'):
        """Add stippling dots where mask is True."""
        nx_, ny_ = mask.shape
        print('mask.shape', mask.shape)
        if lon is not None and lat is not None:
            yy, xx = lat[mask], lon[mask]
        else:
            yy_g, xx_g = np.meshgrid(np.arange(ny_), np.arange(nx_))
            yy, xx = yy_g[mask], xx_g[mask]
        step = max(1, len(xx) // 600)
        ax.scatter(yy[::step], xx[::step], color=color, s=3,
                   marker='.', alpha=0.7, label=label)

    def _add_stippling(ax, mask, lon, lat, color='black', label='Significant'):
        """Add stippling dots where mask is True."""
        nx_, ny_ = mask.shape  # mask shape matches data shape (nx, ny)

        if lon is not None and lat is not None:
            lon_np = np.array(lon)
            lat_np = np.array(lat)

            # <<< CHANGED: build 2D coordinate grids matching data shape (nx, ny) >>>
            # pcolormesh(lon, lat, data) expects lon/lat to match data dimensions
            # so we broadcast lon/lat to (nx, ny) the same way pcolormesh sees them
            if lon_np.ndim == 1 and lat_np.ndim == 1:
                # 1D lon/lat → build 2D meshgrid matching (nx, ny) data layout
                # pcolormesh(lon1d, lat1d, data) plots lon on x-axis, lat on y-axis
                # meshgrid: lon varies along columns (axis=1), lat along rows (axis=0)
                lon_2d, lat_2d = np.meshgrid(lon_np, lat_np)  # both (ny, nx) ... wait
                # but data is (nx, ny), so we need to transpose to match
                # Actually pcolormesh(lon1d, lat1d, data_nxny) will fail unless
                # lon1d.size==ny and lat1d.size==nx — check which axis is which:
                if lon_np.size == ny_ and lat_np.size == nx_:
                    # lon varies along axis=1 (columns), lat along axis=0 (rows)
                    lon_2d, lat_2d = np.meshgrid(lon_np, lat_np)  # (nx, ny)
                else:
                    lon_2d, lat_2d = np.meshgrid(lat_np, lon_np)  # swap if needed
            elif lon_np.ndim == 2 and lat_np.ndim == 2:
                # Already 2D — but may need transpose to match data shape (nx, ny)
                if lon_np.shape == (nx_, ny_):
                    lon_2d, lat_2d = lon_np, lat_np
                else:
                    lon_2d, lat_2d = lon_np.T, lat_np.T  # <<< CHANGED: transpose to match >>>
            else:
                lon_2d, lat_2d = lon_np, lat_np

            # Now index with mask — both lon_2d, lat_2d, mask are (nx, ny)
            xx = lon_2d[mask]  # x-axis coordinates of significant points
            yy = lat_2d[mask]  # y-axis coordinates of significant points

        else:
            # No lon/lat: use array indices directly
            # meshgrid in (nx, ny) space: columns=y-index, rows=x-index
            ii, jj = np.meshgrid(np.arange(ny_), np.arange(nx_))  # both (nx, ny)
            xx = jj[mask]  # x-axis = column index = ny direction
            yy = ii[mask]  # y-axis = row index    = nx direction ... 
            # <<< CHANGED: swap xx/yy to match pcolormesh index convention >>>
            xx, yy = ii[mask], jj[mask]

        step = max(1, len(xx) // 600)
        ax.scatter(xx[::step], yy[::step], color=color, s=3,
                   marker='.', alpha=0.7, label=label)


    # ── Row 1: observed metric ─────────────────────────────────────────────────

    # Panel (0,0): SRGAN observed metric
    _pcolor(axes[0, 0], srgan_obs, 'RdBu_r', -abs_max_obs, abs_max_obs,
            f'SRGAN — {metric_name}', lon, lat)
    _add_stippling(axes[0, 0], srgan_sig, lon, lat,
                   label='CI excludes 0 (significant)')
    axes[0, 0].legend(fontsize=7, loc='lower right', markerscale=4)

    # Panel (0,1): CNN observed metric
    _pcolor(axes[0, 1], cnn_obs, 'RdBu_r', -abs_max_obs, abs_max_obs,
            f'CNN — {metric_name}', lon, lat)
    _add_stippling(axes[0, 1], cnn_sig, lon, lat,
                   label='CI excludes 0 (significant)')
    axes[0, 1].legend(fontsize=7, loc='lower right', markerscale=4)

    # Panel (0,2): difference CNN - SRGAN
    _pcolor(axes[0, 2], diff_obs, 'RdBu_r', -abs_max_diff, abs_max_diff,
            f'Difference (CNN − SRGAN)\n{metric_name}', lon, lat)
    _add_stippling(axes[0, 2], diff_sig, lon, lat, color='black',
                   label='Significant difference')
    axes[0, 2].legend(fontsize=7, loc='lower right', markerscale=4)

    # ── Row 2: CI width ────────────────────────────────────────────────────────

    # Panel (1,0): SRGAN CI width
    _pcolor(axes[1, 0], srgan_ci_width, 'YlOrRd', 0, ci_max,
            f'SRGAN — 95% CI width\n(narrower = more certain)', lon, lat)

    # Panel (1,1): CNN CI width
    _pcolor(axes[1, 1], cnn_ci_width, 'YlOrRd', 0, ci_max,
            f'CNN — 95% CI width\n(narrower = more certain)', lon, lat)

    # Panel (1,2): significance of SRGAN vs CNN difference (4-level)
    # Encode: 0=not sig, 1=CNN better, 2=SRGAN better
    sig_map = np.zeros_like(diff_obs)
    sig_map[diff_sig & (diff_obs > 0)] = 1   # CNN worse (positive = CNN has larger error)
    sig_map[diff_sig & (diff_obs < 0)] = 2   # SRGAN worse
    cmap_sig = mcolors.ListedColormap(['lightgrey', '#d7191c', '#2c7bb6'])
    norm_sig = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    if lon is not None and lat is not None:
        im_sig = axes[1, 2].pcolormesh(lon, lat, sig_map, cmap=cmap_sig, norm=norm_sig)
    else:
        im_sig = axes[1, 2].pcolormesh(sig_map, cmap=cmap_sig, norm=norm_sig)
    legend_elements = [
        Patch(facecolor='lightgrey',  label='No significant difference'),
        Patch(facecolor='#d7191c',    label='SRGAN significantly better'),
        Patch(facecolor='#2c7bb6',    label='CNN significantly better'),
    ]
    axes[1, 2].legend(handles=legend_elements, fontsize=7, loc='lower right')
    axes[1, 2].set_title('Which model is significantly better?', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def main():

    #np.random.seed(42)
    #time, nx, ny = 365, 50, 40

    #reference = np.random.normal(10, 2, (time, nx, ny))
    #srgan_out = reference + np.random.normal(0.2, 0.5, (time, nx, ny))
    #cnn_out   = reference + np.random.normal(0.4, 0.7, (time, nx, ny))

    outdir_fig = f"/nobackup/proj/disk/hclimai/shared/Emilia_Romagna/statistic_figs/bootstrap/"

    reference, srgan_out, cnn_out, lon_ref, lat_ref = get_data()
    time, nx, ny = reference.shape
    print('time, nx, ny:', time, nx, ny)
    print('reference.shape:', reference.shape)
    print('srgan_out.shape:', srgan_out.shape)
    print('lon_ref.shape:', lon_ref.shape)

    # Optional: provide lon/lat for axis labels; set to None to use array indices
    lon = lon_ref #None  # e.g. np.linspace(-10, 30, ny)
    lat = lat_ref #None  # e.g. np.linspace(40, 70, nx)

    n_bootstrap = 10 #00

    for metric_fn, metric_name, save_path in [
        (metric_annual_mean_bias, 'Annual Mean Bias',       f'{outdir_fig}/srgan_cnn_bootstrap_mean_bias.png'),
        (metric_rmse,             'RMSE',                   f'{outdir_fig}/srgan_cnn_bootstrap_rmse.png'),
        (metric_p99_bias,         '99th Percentile Bias',   f'{outdir_fig}/srgan_cnn_bootstrap_p99_bias.png'),
    ]:
        print(f'\n=== {metric_name} ===')

        # Bootstrap for SRGAN
        srgan_obs, srgan_lo, srgan_hi, boot_srgan = bootstrap_metric(
            reference, srgan_out, metric_fn, n_bootstrap=n_bootstrap, seed=42)
        print('srgan_lo.shape:', srgan_lo.shape)

        # Bootstrap for CNN
        cnn_obs, cnn_lo, cnn_hi, boot_cnn = bootstrap_metric(
            reference, cnn_out, metric_fn, n_bootstrap=n_bootstrap, seed=42)

        # Significance: CI excludes zero (i.e. lo and hi same sign)
        srgan_sig = (srgan_lo > 0) | (srgan_hi < 0)   # SRGAN metric sig different from 0
        cnn_sig   = (cnn_lo   > 0) | (cnn_hi   < 0)   # CNN metric sig different from 0

        # Significance of SRGAN vs CNN difference
        boot_diff = boot_cnn - boot_srgan               # shape: (n_bootstrap, nx, ny)
        diff_lo   = np.percentile(boot_diff, 2.5,  axis=0)
        diff_hi   = np.percentile(boot_diff, 97.5, axis=0)
        diff_sig  = (diff_lo > 0) | (diff_hi < 0)      # CI of difference excludes zero

        n_sig_srgan = srgan_sig.sum()
        n_sig_cnn   = cnn_sig.sum()
        n_sig_diff  = diff_sig.sum()
        n_total     = nx * ny
        print(f'  SRGAN sig grid points:      {n_sig_srgan}/{n_total} ({100*n_sig_srgan/n_total:.1f}%)')
        print(f'  CNN   sig grid points:      {n_sig_cnn}/{n_total}   ({100*n_sig_cnn/n_total:.1f}%)')
        print(f'  SRGAN vs CNN sig diff pts:  {n_sig_diff}/{n_total}  ({100*n_sig_diff/n_total:.1f}%)')

        plot_2d_metric_comparison(
            srgan_obs, cnn_obs,
            srgan_lo, srgan_hi,
            cnn_lo,   cnn_hi,
            srgan_sig, cnn_sig, diff_sig,
            metric_name=metric_name,
            lon=lon, lat=lat,
            save_path=save_path
        )


def get_data():
    # --- Configuration ---
    # Update these names to match your actual experiment folders or identifiers
    #base_path = "/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/"  # Path where your NetCDF files are stored
    #outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/boxplot/"
    base_path = "/nobackup/proj/disk/hclimai/shared/Emilia_Romagna/"  # Path where your NetCDF files are stored
    var_names = {'var1':'tas'}
    var_names_to_read = {'var1':'tas'}

    GCM = 'ECMWF-ERAINT'
    #GCM = "ICHEC-EC-EARTH_HIST"
    #GCM = "ICHEC-EC-EARTH_RCP85_MC"
    #GCM = "ICHEC-EC-EARTH_RCP85_LC"
    if GCM == 'ECMWF-ERAINT':
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2000, 2009, 2000, 2009
        EXP_SRGAN = 'EPOCH100_tas_wsmto_ERAI_2009_arrhenius'
        title_def = '(k) ERAI-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_HIST":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 1995, 2005, 1995, 2005
        EXP_SRGAN = 'ECE/EPOCH100_tas_mrsol_wsmt_lnoise0.1_ECEHis_atos'
        title_def = '(l) ECE-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_RCP85_MC":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2040, 2050, 2040, 2050
        EXP_SRGAN = 'ECE/EPOCH100_tas_mrsol_wsmt_lnoise0.1_ECEFutMC_atos'
        title_def = '(m) ECE-MC2MC'
    elif GCM == "ICHEC-EC-EARTH_RCP85_LC":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2090, 2099, 2089, 2099

    file_dict = {
        'HCLIM3':  {'tas': f'{base_path}/cropped/{GCM}/3km/6hr/tas/tas_3km_6hr_{FIRST_YEAR_3km}01010000-{LAST_YEAR_3km}12311800.nc' },
        #'HCLIM12': {'tas': f'{base_path}/cropped/{GCM}/12km/6hr/tas/tas_12km_6hr_{FIRST_YEAR_12km}01010000-{LAST_YEAR_12km}12311800.nc' },
        'SRGAN':   {'tas': f'{base_path}SG/SRGAN_OUT/{EXP_SRGAN}/predictant_ypred_1.nc' }, 
        'CNN':   {'tas': f'{base_path}SG/SRGAN_OUT/EPOCH100_tas_wsmto_tile_ERAI_2009_arrhenius/predictant_ypred_1.nc' }, 
        #'CNN':     {'tas': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_tas_2009.nc' }, 
        # ERAI
        #'ERA5':  {'mrsol': f'/nobackup/rossby27/users/sm_fuxwa/ERA5/2009/tas_mrsol_ERA5_regrid_3km_2009_2009_timestd_dim.nc',
        #          'tas': f'/nobackup/rossby27/users/sm_fuxwa/ERA5/2009/tas_mrsol_ERA5_regrid_3km_2009_2009_timestd_dim.nc' },

        # ECE Hist
        #'CNN':  {'mrsol': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_mrsol_2009.nc',
        #          'tas': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_tas_2009.nc' }, 

    }

    reference_experiment, model1_experiment, model2_experiment  = 'HCLIM3', 'SRGAN', 'CNN'
    time_range = def_time_range(GCM)
    time_idx_range = get_time_index.get_time_index(time_range)
    experiment_names = file_dict.keys()
    fontsize_def = 20


    all_correlations = []
    var_dict = {}
    # --- Data Processing ---
    for exp in experiment_names:
        #if 'CNN' in exp:
        #    var_names_to_read = {'var1':'test', 'var2':'test'}
        print(f"Processing {exp}...")
        print (file_dict[exp][var_names['var1']]) 
        # 1. Open the files 
        # Assuming file naming like: Exp1_tas.nc and Exp1_mrsol.nc
        #ds_tas = xr.open_dataset(os.path.join(base_path, f"{exp}_tas.nc"))
        #ds_mrsol = xr.open_dataset(os.path.join(base_path, f"{exp}_mrsol.nc"))
        ds_var = xr.open_dataset(file_dict[exp][var_names['var1']])
        
        # 2. Extract DataArrays
        var = ds_var[var_names_to_read['var1']][time_idx_range[str(exp)][var_names['var1']]['start_idx'][0]:time_idx_range[str(exp)][var_names['var1']]['end_idx'][0]]
        var_dict[exp] = var.to_numpy()
        if exp == reference_experiment:
            lon_ref, lat_ref = ds_var['lon'].to_numpy(), ds_var['lat'].to_numpy()

        # 3. Calculate Pearson Correlation per grid point along the 'time' dimension
        # xr.corr automatically aligns coordinates and computes correlation over the specified dim
        #cor_map = xr.corr(tas, mrsol, dim='time')
        
        # 4. Flatten the map into a 1D array of values and remove NaNs (e.g., over oceans or missing data)
        #cor_values = cor_map.values.flatten()
        #cor_values = cor_values[~np.isnan(cor_values)]
        
        #all_correlations.append(cor_values)

    reference = var_dict[reference_experiment]
    srgan_out = var_dict[model1_experiment]
    cnn_out   = var_dict[model2_experiment]

    return reference, srgan_out, cnn_out, lon_ref, lat_ref

def def_time_range(GCM):

    time_range_erai = {'HCLIM12': 
        {'mrsol': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'ERA5': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}

    time_range_ecehi2hi = {'HCLIM12': 
        {'mrsol': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}

    time_range_ecemc2mc = {'HCLIM12': 
        {'mrsol': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}

    if GCM == 'ECMWF-ERAINT':
        time_range = time_range_erai
    elif GCM == "ICHEC-EC-EARTH_HIST":
        time_range = time_range_ecehi2hi
    elif GCM == "ICHEC-EC-EARTH_RCP85_MC":
        time_range = time_range_ecemc2mc
    elif GCM == "ICHEC-EC-EARTH_RCP85_LC":
        print ('LC time range not defined yet')

    return time_range


if __name__ == "__main__":
    main()
