import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import xarray as xr
import os, hpc, sys
import get_time_index
import cartopy.io.shapereader as shpreader
from shapely.geometry import MultiLineString
from statsmodels.tsa.stattools import acf

# https://medium.com/data-science/calculating-confidence-interval-with-bootstrapping-872c657c058d

# ── Metric functions (unchanged) ─────────────────────────────────────────────

def metric_annual_mean_bias(reference, model_output):
    #return np.nanmean(model_output - reference, axis=0) 
    return np.abs(np.nanmean(model_output - reference, axis=0) )
    #return np.mean(np.abs(model_output - reference), axis=0) 
    #return np.mean(model_output, axis=0) - np.mean(reference, axis=0)

def metric_rmse(reference, model_output):
    return np.sqrt(np.mean((model_output - reference)**2, axis=0))

def metric_p99_bias(reference, model_output):
    #return np.nanpercentile(model_output, 99, axis=0) - np.nanpercentile(reference, 99, axis=0)
    return np.abs(np.nanpercentile(model_output, 99, axis=0) - np.nanpercentile(reference, 99, axis=0))


# ── Bootstrap function (unchanged) ───────────────────────────────────────────

def bootstrap_metric_origi_version(reference, model_output, metric_fn,
                     n_bootstrap=1000, ci=95, axis=0, seed=42):
    rng = np.random.default_rng(seed)
    n_time = reference.shape[axis]
    metric_obs = metric_fn(reference, model_output)
    boot_metrics = []

    for _ in range(n_bootstrap):
        #block_size = 10
        #n_blocks = int(np.ceil(n_time / block_size))
        #block_starts = rng.integers(0, n_time - block_size + 1, size=n_blocks)
        #idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])[:n_time]
        idx = rng.integers(0, n_time, size=n_time)
        ref_boot = np.take(reference,    idx, axis=axis)
        mod_boot = np.take(model_output, idx, axis=axis)
        boot_metrics.append(metric_fn(ref_boot, mod_boot))
    boot_metrics = np.array(boot_metrics)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(boot_metrics, alpha,       axis=0)
    ci_upper = np.percentile(boot_metrics, 100 - alpha, axis=0)
    return metric_obs, ci_lower, ci_upper, boot_metrics

def bootstrap_metric(reference, model_output, metric_fn,
                      n_bootstrap=1000, ci=95, axis=0, seed=42,
                      method='iid', block_size=10, m=None):
    """
    Bootstrap a metric computed by metric_fn(reference, model_output).

    Parameters
    ----------
    method : str
        'iid'        - standard i.i.d. resampling (default, original behavior)
        'block'      - moving block bootstrap (overlapping blocks)
        'circular'   - circular block bootstrap (wraps around series end)
        'stationary' - stationary bootstrap (random block lengths, geometric dist.)
    block_size : int
        Block length in time steps. Used by 'block' and 'circular'.
        For 'stationary', used as the *mean* block length.
    m : int or None
        If set, draws m time steps per bootstrap replicate instead of n_time
        (m-out-of-n bootstrap). Can be combined with any method above.
        If None, m = n_time (standard case).
    """
    rng = np.random.default_rng(seed)
    n_time = reference.shape[axis]
    m_eff = n_time if m is None else m

    metric_obs = metric_fn(reference, model_output)

    def _iid_indices():
        return rng.integers(0, n_time, size=m_eff)

    def _block_indices():
        # moving/overlapping block bootstrap: block can start anywhere
        # in [0, n_time - block_size], no wraparound
        n_blocks = int(np.ceil(m_eff / block_size))
        starts = rng.integers(0, max(1, n_time - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        return idx[:m_eff]

    def _circular_indices():
        # circular block bootstrap: block can start anywhere in [0, n_time),
        # wraps around the end back to the start
        n_blocks = int(np.ceil(m_eff / block_size))
        starts = rng.integers(0, n_time, size=n_blocks)
        idx = np.concatenate([
            (np.arange(s, s + block_size)) % n_time for s in starts
        ])
        return idx[:m_eff]

    def _stationary_indices():
        # stationary bootstrap (Politis & Romano): geometric block lengths
        # with mean = block_size; each new block starts at a random point
        p = 1.0 / block_size
        idx = np.empty(m_eff, dtype=int)
        pos = 0
        while pos < m_eff:
            start = rng.integers(0, n_time)
            length = rng.geometric(p)
            block = (start + np.arange(length)) % n_time
            take = min(length, m_eff - pos)
            idx[pos:pos + take] = block[:take]
            pos += take
        return idx

    index_fn = {
        'iid':        _iid_indices,
        'block':      _block_indices,
        'circular':   _circular_indices,
        'stationary': _stationary_indices,
    }[method]

    boot_metrics = []
    for _ in range(n_bootstrap):
        idx = index_fn()
        ref_boot = np.take(reference,    idx, axis=axis)
        mod_boot = np.take(model_output, idx, axis=axis)
        boot_metrics.append(metric_fn(ref_boot, mod_boot))

    boot_metrics = np.array(boot_metrics)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(boot_metrics, alpha,       axis=0)
    ci_upper = np.percentile(boot_metrics, 100 - alpha, axis=0)
    return metric_obs, ci_lower, ci_upper, boot_metrics

def estimate_block_size(reference, model_output, axis=0,
                         min_block=2, max_block=60,
                         safety_factor=1.5, use_fft=True,
                         verbose=True):
    """
    Estimate an appropriate block length for block bootstrap by finding the
    decorrelation length of the (model - reference) residual series.

    Strategy:
      1. Spatially average the residual to get one representative 1D time series.
      2. Compute its autocorrelation function (ACF).
      3. Find the first lag where |ACF| drops below the white-noise
         significance bound (~1.96/sqrt(n)).
      4. Scale that lag by `safety_factor` to get the final block_size,
         clipped to [min_block, max_block].

    Parameters
    ----------
    reference, model_output : ndarray, shape (time, nx, ny) [or similar]
        Full fields; residual = model_output - reference is analyzed.
    axis : int
        Time axis (must match what bootstrap_metric uses).
    min_block, max_block : int
        Clip the estimated block size to a sane range.
    safety_factor : float
        Multiply the raw decorrelation lag by this factor before clipping,
        since decorrelation-lag detection tends to underestimate true
        dependence length in short/noisy series. 1.5 is a reasonable default.
    use_fft : bool
        Passed to statsmodels' acf() for speed on longer series.
    verbose : bool
        Print diagnostic info (decorrelation lag found, final block size).

    Returns
    -------
    block_size : int
    diagnostics : dict
        {'decorr_lag': int, 'acf_values': ndarray, 'n_time': int}
    """

    # Move time axis to front if needed, then average over all non-time axes
    resid = np.moveaxis(model_output - reference, axis, 0)
    resid_1d = np.nanmean(resid, axis=tuple(range(1, resid.ndim)))  # shape (n_time,)

    n_time = resid_1d.shape[0]
    max_lag = min(max_block * 4, n_time // 2 - 1)  # don't ask for absurd lags
    max_lag = max(max_lag, min_block + 1)

    acf_vals = acf(resid_1d, nlags=max_lag, fft=use_fft, missing='drop')

    # White-noise significance bound (95%)
    bound = 1.96 / np.sqrt(n_time)

    # Find first lag (>=1) where |ACF| drops below the bound
    below = np.abs(acf_vals[1:]) < bound
    if below.any():
        decorr_lag = int(np.argmax(below)) + 1   # +1 because we sliced off lag 0
    else:
        # ACF never decays within max_lag -> long-memory / strong seasonality;
        # fall back to max_lag as a conservative estimate
        decorr_lag = max_lag
        if verbose:
            print(f'  [warn] ACF did not decay below significance within '
                  f'{max_lag} lags; using max_lag as decorrelation estimate.')

    block_size = int(np.round(decorr_lag * safety_factor))
    block_size = int(np.clip(block_size, min_block, max_block))

    if verbose:
        print(f'  ACF-based block size: decorr_lag={decorr_lag}, '
              f'safety_factor={safety_factor} -> block_size={block_size} '
              f'(n_time={n_time})')

    return block_size, {'decorr_lag': decorr_lag,
                         'acf_values': acf_vals,
                         'n_time': n_time}

def variance_inflation_factor(x, max_lag=60):
    """
    Compute the variance inflation factor for the sample mean of an
    autocorrelated series x, using the standard 'long-run variance'
    correction (Bartlett-type formula).

    inflation_factor ≈ 1  -> autocorrelation negligible, i.i.d. bootstrap fine
    inflation_factor >> 1 -> autocorrelation matters, use block bootstrap

    Also returns the effective sample size n_eff = n / inflation_factor,
    i.e. how many "independent" observations your n correlated points
    are worth for estimating the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    #acf_vals = _acf_numpy(x, nlags=max_lag)  # from earlier; or use statsmodels
    acf_vals = acf(x, nlags=max_lag, fft=True, missing='drop')

    k = np.arange(1, max_lag + 1)
    inflation_factor = 1 + 2 * np.sum((1 - k / n) * acf_vals[1:max_lag + 1])
    inflation_factor = max(inflation_factor, 1.0)  # can't be less than 1

    n_eff = n / inflation_factor
    return inflation_factor, n_eff


# ── 2D plotting helpers ───────────────────────────────────────────────────────

def plot_2d_metric_comparison(srgan_obs, cnn_obs,
                               srgan_lo, srgan_hi,
                               cnn_lo,   cnn_hi,
                               srgan_sig, cnn_sig, diff_sig,
                               n_bootstrap,
                               metric_name, lon=None, lat=None,
                               save_path=None, higher_is_better=False):
    """
    6-panel figure per metric:
      Row 1: observed metric map for SRGAN | CNN | difference (CNN - SRGAN)
      Row 2: CI width map for SRGAN        | CNN | significance of difference
    Stippling marks where CIs do NOT include zero (i.e. significant).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Bootstrap Analysis — {metric_name}  (95% CI, {n_bootstrap} resamples)',
                 fontsize=14, fontweight='bold')

    #diff_obs = cnn_obs - srgan_obs
    diff_obs = srgan_obs - cnn_obs

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

    # Panel (0,0): CNN observed metric
    _pcolor(axes[0, 0], cnn_obs, 'RdBu_r', -abs_max_obs, abs_max_obs,
            f'(a) CNN', lon, lat)
            #f'(a) CNN — {metric_name}', lon, lat)
    _add_coastline(axes[0, 0], lon, lat)
    _add_stippling(axes[0, 0], cnn_sig, lon, lat,
                   label='CI excludes 0 (significant)')
    axes[0, 0].legend(fontsize=7, loc='lower right', markerscale=4)

    # Panel (0,1): SRGAN observed metric
    _pcolor(axes[0, 1], srgan_obs, 'RdBu_r', -abs_max_obs, abs_max_obs,
            f'(b) SRGAN', lon, lat)
            #f'(b) SRGAN — {metric_name}', lon, lat)
    _add_coastline(axes[0, 1], lon, lat)
    _add_stippling(axes[0, 1], srgan_sig, lon, lat,
                   label='CI excludes 0 (significant)')
    axes[0, 1].legend(fontsize=7, loc='lower right', markerscale=4)

    # Panel (0,2): difference CNN - SRGAN
    _pcolor(axes[0, 2], diff_obs, 'RdBu_r', -abs_max_diff, abs_max_diff,
            f'(c) Difference (SRGAN - CNN)', lon, lat)
            #f'(c) Difference (SRGAN - CNN)\n{metric_name}', lon, lat)
    _add_coastline(axes[0, 2], lon, lat)
    _add_stippling(axes[0, 2], diff_sig, lon, lat, color='black',
                   label='Significant difference')
    axes[0, 2].legend(fontsize=7, loc='lower right', markerscale=4)

    # ── Row 2: CI width ────────────────────────────────────────────────────────

    # Panel (1,0): CNN CI width
    _pcolor(axes[1, 0], cnn_ci_width, 'YlOrRd', 0, ci_max,
            f'(d) CNN — 95% CI width\n(narrower = more certain)', lon, lat)
    _add_coastline(axes[1, 0], lon, lat)

    # Panel (1,1): SRGAN CI width
    _pcolor(axes[1, 1], srgan_ci_width, 'YlOrRd', 0, ci_max,
            f'(e) SRGAN — 95% CI width\n(narrower = more certain)', lon, lat)
    _add_coastline(axes[1, 1], lon, lat)

    # Panel (1,2): significance of SRGAN vs CNN difference (4-level)
    # Encode: 0=not sig, 1=CNN better, 2=SRGAN better
    sig_map = np.zeros_like(diff_obs)
    if higher_is_better:
        sig_map[diff_sig & (diff_obs > 0)] = 2   # SRGAN better (higher value)
        sig_map[diff_sig & (diff_obs < 0)] = 1   # CNN better
    else:
        sig_map[diff_sig & (diff_obs > 0)] = 1   # SRGAN worse (positive = SRGAN has larger error)
        sig_map[diff_sig & (diff_obs < 0)] = 2   # CNN worse

    cmap_sig = mcolors.ListedColormap(['lightgrey', '#d7191c', '#2c7bb6'])
    norm_sig = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    if lon is not None and lat is not None:
        im_sig = axes[1, 2].pcolormesh(lon, lat, sig_map, cmap=cmap_sig, norm=norm_sig)
    else:
        im_sig = axes[1, 2].pcolormesh(sig_map, cmap=cmap_sig, norm=norm_sig)
    legend_elements = [
        Patch(facecolor='lightgrey',  label='Not significant'), # difference'),
        Patch(facecolor='#d7191c',    label='CNN'), #significantly better'),
        Patch(facecolor='#2c7bb6',    label='SRGAN') # significantly better'),
    ]
    axes[1, 2].legend(handles=legend_elements, fontsize=7, loc='lower right')
    #axes[1, 2].set_title('Which model is significantly better?', fontsize=11)
    axes[1, 2].set_title(f'(f) Model with significantly lower {metric_name}', fontsize=11)
    _add_coastline(axes[1, 2], lon, lat)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def _add_coastline(ax, lon, lat, color='black', linewidth=0.6, resolution='50m'):
    """
    Draw coastlines on a plain (non-GeoAxes) matplotlib axis, using lon/lat
    as plain x/y coordinates — matches how pcolormesh(lon, lat, data) is
    already being plotted in this figure (equirectangular / plate-carree-like).
    """
    if lon is None or lat is None:
        return  # no georeferencing available, skip silently

    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    shp = shpreader.natural_earth(resolution=resolution,
                                   category='physical', name='coastline')
    reader = shpreader.Reader(shp)

    for geom in reader.geometries():
        lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for line in lines:
            x, y = line.xy
            x, y = np.array(x), np.array(y)
            # only draw segments that fall (at least partly) within plot extent
            mask = (x >= lon_min) & (x <= lon_max) & (y >= lat_min) & (y <= lat_max)
            if mask.any():
                ax.plot(x, y, color=color, linewidth=linewidth, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def main():

    #np.random.seed(42)
    #time, nx, ny = 365, 50, 40

    #reference = np.random.normal(10, 2, (time, nx, ny))
    #srgan_out = reference + np.random.normal(0.2, 0.5, (time, nx, ny))
    #cnn_out   = reference + np.random.normal(0.4, 0.7, (time, nx, ny))

    #method_bootstrap = 'block'
    n_bootstrap = 1000  #1000
    var_name = 'pr' #'tas', 'pr'
    hpc_name = hpc.get_hpc_name()
    reference, srgan_out, cnn_out, lon_ref, lat_ref, outdir_fig = get_data(hpc_name, var_name)
    assert reference.shape == srgan_out.shape == cnn_out.shape
    time, nx, ny = reference.shape
    print('time, nx, ny:', time, nx, ny)
    #print('reference.shape:', reference.shape)
    #print('srgan_out.shape:', srgan_out.shape)
    #print('lon_ref.shape:', lon_ref.shape)

    # Optional: provide lon/lat for axis labels; set to None to use array indices
    lon = lon_ref #None  # e.g. np.linspace(-10, 30, ny)
    lat = lat_ref #None  # e.g. np.linspace(40, 70, nx)

    print(f'\n=== Checking autocorrelation impact on annual mean for {var_name} ===')

    resid_srgan_1d = np.nanmean(srgan_out - reference, axis=(1, 2))  # shape (time,)
    resid_cnn_1d   = np.nanmean(cnn_out   - reference, axis=(1, 2))  # shape (time,)

    vif_srgan, neff_srgan = variance_inflation_factor(resid_srgan_1d)
    vif_cnn,   neff_cnn   = variance_inflation_factor(resid_cnn_1d)

    print(f'  SRGAN: variance inflation factor = {vif_srgan:.2f}, '
          f'n_eff = {neff_srgan:.0f} (raw n = {time})')
    print(f'  CNN:   variance inflation factor = {vif_cnn:.2f}, '
          f'n_eff = {neff_cnn:.0f} (raw n = {time})')

    # decide bootstrap method for the mean-bias metric based on this diagnostic
    mean_bias_method = 'iid' if max(vif_srgan, vif_cnn) < 1.3 else 'circular'
    print(f'  -> using method="{mean_bias_method}" for Annual Mean Bias bootstrap\n')

   
    # ── Empirically estimate block size from the data ──────────────────────
    print(f'\n=== Estimating block size for {var_name} ===')
    block_size_srgan, diag_srgan = estimate_block_size(reference, srgan_out)
    block_size_cnn,   diag_cnn   = estimate_block_size(reference, cnn_out)
    # Use the larger of the two so the block scheme is conservative enough
    # for both models' residual structure
    block_size = max(block_size_srgan, block_size_cnn)
    print(f'  block_size for SRGAN is {block_size_srgan} and for CNN is {block_size_cnn}\n')
    print(f'  -> using block_size={block_size} for {var_name}\n')

    m_reduced = int(0.75 * time)   # m-out-of-n sample size for extreme-quantile metrics
    #m_reduced = int(2. / 3. * time)   # m-out-of-n sample size for extreme-quantile metrics
    print(f'  -> using m={m_reduced} (of n={time}) for order-statistic metrics\n')

    # ── Per-metric bootstrap configuration ──────────────────────────────────
    metric_configs = [
        dict(fn=metric_annual_mean_bias, name='Annual Mean Bias',
             method='circular', block_size=block_size, m=None,
             save='mean_bias'),
        dict(fn=metric_rmse, name='RMSE',
             method='circular', block_size=block_size, m=None,
             save='rmse'),
        dict(fn=metric_p99_bias, name='99th Percentile Bias',
             method='circular', block_size=block_size, m=m_reduced,
             save='p99_bias'),
    ]

    #for metric_fn, metric_name, save_path in [
    #    (metric_annual_mean_bias, 'Annual Mean Bias',       f'{outdir_fig}/srgan_cnn_bootstrap_{method_bootstrap}_n{n_bootstrap}_{var_name}_mean_bias.png'),
    #    (metric_rmse,             'RMSE',                   f'{outdir_fig}/srgan_cnn_bootstrap_{method_bootstrap}_n{n_bootstrap}_{var_name}_rmse.png'),
    #    (metric_p99_bias,         '99th Percentile Bias',   f'{outdir_fig}/srgan_cnn_bootstrap_{method_bootstrap}_n{n_bootstrap}_{var_name}_p99_bias.png'),
    #]:
    #    print(f'\n=== {metric_name} ===')
    for cfg in metric_configs:
        metric_fn, metric_name = cfg['fn'], cfg['name']
        save_path = f"{outdir_fig}/srgan_npnorog_cnn_bootstrap_{cfg['method']}_n{n_bootstrap}_{var_name}_{cfg['save']}_v2.png"
        print(f'\n=== {metric_name} '
              f'(method={cfg["method"]}, block_size={cfg["block_size"]}, m={cfg["m"]}) ===')

        srgan_obs, srgan_lo, srgan_hi, boot_srgan = bootstrap_metric(
            reference, srgan_out, metric_fn,
            n_bootstrap=n_bootstrap, seed=42,
            method=cfg['method'], block_size=cfg['block_size'], m=cfg['m'])

        cnn_obs, cnn_lo, cnn_hi, boot_cnn = bootstrap_metric(
            reference, cnn_out, metric_fn,
            n_bootstrap=n_bootstrap, seed=42,
            method=cfg['method'], block_size=cfg['block_size'], m=cfg['m'])

        # Bootstrap for SRGAN
        #srgan_obs, srgan_lo, srgan_hi, boot_srgan = bootstrap_metric(
        #    reference, srgan_out, metric_fn, n_bootstrap=n_bootstrap, seed=42)
        #print('srgan_lo.shape:', srgan_lo.shape)

        # Bootstrap for CNN
        #cnn_obs, cnn_lo, cnn_hi, boot_cnn = bootstrap_metric(
        #    reference, cnn_out, metric_fn, n_bootstrap=n_bootstrap, seed=42)

        #srgan_obs, srgan_lo, srgan_hi, boot_srgan = bootstrap_metric(
        #    reference, srgan_out, metric_fn,
        #    n_bootstrap=n_bootstrap, seed=42,
        #    method=method_bootstrap, block_size=block_size)

        #cnn_obs, cnn_lo, cnn_hi, boot_cnn = bootstrap_metric(
        #    reference, cnn_out, metric_fn,
        #    n_bootstrap=n_bootstrap, seed=42,
        #    method=method_bootstrap, block_size=block_size)

        # Significance: CI excludes zero (i.e. lo and hi same sign)
        srgan_sig = (srgan_lo > 0) | (srgan_hi < 0)   # SRGAN metric sig different from 0
        cnn_sig   = (cnn_lo   > 0) | (cnn_hi   < 0)   # CNN metric sig different from 0

        # Significance of SRGAN vs CNN difference
        boot_diff = boot_srgan - boot_cnn #- boot_srgan               # shape: (n_bootstrap, nx, ny)
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
            n_bootstrap,
            metric_name=metric_name,
            lon=lon, lat=lat,
            save_path=save_path,
            higher_is_better=False,
        )


def get_data(hpc_name, var_name):

    # --- Configuration ---
    if hpc_name == 'freja':
        base_path = "/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/"  # Path where your NetCDF files are stored
        outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/bootstrap_metrics/TEST_2003_2009/"
    elif hpc_name == 'arrhenius':
        base_path = "/nobackup/proj/disk/hclimai/shared/Emilia_Romagna/"  # Path where your NetCDF files are stored
        outdir_fig = f"/nobackup/proj/disk/hclimai/shared/Emilia_Romagna/statistic_figs/bootstrap_metrics/"
    else:
        sys.exit(f'HPC configuration not defined for {hpc_name}')

    os.makedirs(outdir_fig, exist_ok=True)

    var_names = {'var1': var_name}
    var_names_to_read = {'var1': var_name}
    unit_convert = { 'pr': {'SRGAN': 86400.0, 'CNN': 1.0, 'HCLIM3': 86400.0, 'HCLIM12': 86400.0 }, 
        'tas': {'SRGAN': 1.0, 'CNN': 1.0, 'HCLIM3': 1.0, 'HCLIM12': 1.0 }}

    GCM = 'ECMWF-ERAINT'
    #GCM = "ICHEC-EC-EARTH_HIST"
    #GCM = "ICHEC-EC-EARTH_RCP85_MC"
    #GCM = "ICHEC-EC-EARTH_RCP85_LC"
    if GCM == 'ECMWF-ERAINT':
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2000, 2009, 2000, 2009
        #EXP_SRGAN = 'EPOCH100_tas_wsmto_ERAI_2009_arrhenius'
        if  var_names['var1'] == 'tas':
            #EXP_SRGAN = 'EPOCH100_tas_scale_time_stdscaler_wt_worog_gpufix_bs50_ERAI_atos'
            #FILENAME='predictant_ypred_1.nc'
            #EXP_SRGAN = '/ARRHENIUS/EPOCH100_tas_wsmto_ERAI_2003_2009_arrhenius'
            EXP_SRGAN = '/ARRHENIUS/EPOCH100_tas_wsmt_ERAI_2003_2009_arrhenius'
            FILENAME='predictant_ypred_2.nc'
        elif var_names['var1'] == 'pr':
            #EXP_SRGAN = 'EPOCH100_pr_scale_time_stdscaler_wp_worog_gpufix_bs50_ERAI_atos'
            #FILENAME='predictant_ypred_1.nc'
            #EXP_SRGAN = '/ARRHENIUS/EPOCH100_pr_wsmpo_v2_ERAI_2003_2009_arrhenius'
            #FILENAME='predictant_ypred_2.nc'
            ###EXP_SRGAN = '/ARRHENIUS/EPOCH100_pr_wsm_corrv2_ERAI_2003_2009_arrhenius'
            EXP_SRGAN = '/ARRHENIUS/EPOCH100_pr_wsm_corr_ERAI_2003_2009_arrhenius'
            FILENAME='predictant_ypred_2.nc'
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
        'HCLIM3':  { 'pr': f'{base_path}/cropped/{GCM}/3km/6hr/pr/pr_3km_6hr_{FIRST_YEAR_3km}01010300-{LAST_YEAR_3km}12312100.nc', \
                    'tas': f'{base_path}/cropped/{GCM}/3km/6hr/tas/tas_3km_6hr_{FIRST_YEAR_3km}01010000-{LAST_YEAR_3km}12311800.nc' },
        #'HCLIM12': {'tas': f'{base_path}/cropped/{GCM}/12km/6hr/tas/tas_12km_6hr_{FIRST_YEAR_12km}01010000-{LAST_YEAR_12km}12311800.nc' },
        'SRGAN':   {'pr': f'{base_path}SG/SRGAN_OUT/{EXP_SRGAN}/{FILENAME}', \
                   'tas': f'{base_path}SG/SRGAN_OUT/{EXP_SRGAN}/{FILENAME}' }, 
        #'CNN':   {'tas': f'{base_path}SG/SRGAN_OUT/EPOCH100_tas_wsmto_tile_ERAI_2009_arrhenius/predictant_ypred_1.nc' }, 
        'CNN':     {'tas': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_tas_2009.nc', \
                     'pr':  f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_pr_2009.nc'}, 
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
    print('time_idx_range:', time_idx_range)
    experiment_names = file_dict.keys()
    fontsize_def = 20


    all_correlations = []
    var_dict = {}
    # --- Data Processing ---
    for exp in experiment_names:
        if 'CNN' in exp:
            var_names_to_read = {'var1':'test'}
        print(f"Processing {exp}...")
        #print (file_dict[exp][var_names['var1']]) 
        # 1. Open the files 
        # Assuming file naming like: Exp1_tas.nc and Exp1_mrsol.nc
        #ds_tas = xr.open_dataset(os.path.join(base_path, f"{exp}_tas.nc"))
        ds_var = xr.open_dataset(file_dict[exp][var_names['var1']])
        
        # 2. Extract DataArrays
        var = ds_var[var_names_to_read['var1']][time_idx_range[str(exp)][var_names['var1']]['start_idx'][0]:time_idx_range[str(exp)][var_names['var1']]['end_idx'][0]]
        var_dict[exp] = var.to_numpy()
        if exp == reference_experiment:
            lon_ref, lat_ref = ds_var['lon'].to_numpy(), ds_var['lat'].to_numpy()

    reference = var_dict[reference_experiment] * unit_convert[var_name][reference_experiment]
    srgan_out = var_dict[model1_experiment] * unit_convert[var_name][model1_experiment]
    cnn_out   = var_dict[model2_experiment] * unit_convert[var_name][model2_experiment]

    return reference, srgan_out, cnn_out, lon_ref, lat_ref, outdir_fig


def def_time_range(GCM):

    time_range_erai = {'HCLIM12': 
        {'pr': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'pr': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'ERA5': 
        {'pr': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'pr': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'pr': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}

    time_range_ecehi2hi = {'HCLIM12': 
        {'pr': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'pr': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'pr': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'pr': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}

    time_range_ecemc2mc = {'HCLIM12': 
        {'pr': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'pr': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'pr': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'pr': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
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
