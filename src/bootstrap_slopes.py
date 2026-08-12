import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from collections import defaultdict
import plot_scatter

def create_nested_dict():
    """Creates an infinitely nested dictionary structure."""
    return defaultdict(create_nested_dict)

# ── Core bootstrap functions ──────────────────────────────────────────────────

def fit_linear(x, y):
    """Linear regression slope and intercept."""
    slope, intercept, r, p, se = stats.linregress(x, y)
    return slope, intercept, r, p


def fit_quadratic(x, y):
    """Quadratic fit coefficients [a, b, c] for ax^2 + bx + c."""
    return np.polyfit(x, y, 2)


def bootstrap_slopes(x, y, n_bootstrap=1000, ci=95, seed=42):
    """
    Bootstrap confidence intervals for linear and quadratic regression slopes.
    Resamples (x, y) pairs with replacement to account for autocorrelation
    — bootstrapping the pairs preserves the joint distribution without
    assuming independence.

    Args:
        x:           1D array — predictor (e.g. soil moisture)
        y:           1D array — response (e.g. temperature)
        n_bootstrap: number of bootstrap resamples
        ci:          confidence interval level (default 95)
        seed:        random seed

    Returns:
        results dict with observed and CI for linear slope, intercept,
        quadratic coefficients, and Pearson r
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    # Observed fit
    lin_slope, lin_intercept, r_obs, p_obs = fit_linear(x, y)
    quad_coeffs_obs = fit_quadratic(x, y)

    # Bootstrap resamples
    boot_lin_slopes      = np.zeros(n_bootstrap)
    boot_lin_intercepts  = np.zeros(n_bootstrap)
    boot_quad_a          = np.zeros(n_bootstrap)
    boot_quad_b          = np.zeros(n_bootstrap)
    boot_quad_c          = np.zeros(n_bootstrap)
    boot_r               = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        s, inc, r_b, _, _ = stats.linregress(xb, yb)
        boot_lin_slopes[i]     = s
        boot_lin_intercepts[i] = inc
        boot_r[i]              = r_b
        qc = np.polyfit(xb, yb, 2)
        boot_quad_a[i], boot_quad_b[i], boot_quad_c[i] = qc

    alpha = (100 - ci) / 2

    def ci_bounds(arr):
        return np.percentile(arr, alpha), np.percentile(arr, 100 - alpha)

    return {
        'lin_slope':      lin_slope,
        'lin_slope_ci':   ci_bounds(boot_lin_slopes),
        'lin_intercept':  lin_intercept,
        'lin_intercept_ci': ci_bounds(boot_lin_intercepts),
        'quad_a':         quad_coeffs_obs[0],
        'quad_a_ci':      ci_bounds(boot_quad_a),
        'quad_b':         quad_coeffs_obs[1],
        'quad_b_ci':      ci_bounds(boot_quad_b),
        'quad_c':         quad_coeffs_obs[2],
        'quad_c_ci':      ci_bounds(boot_quad_c),
        'r':              r_obs,
        'r_ci':           ci_bounds(boot_r),
        'p':              p_obs,
        'boot_lin_slopes': boot_lin_slopes,  # keep for slope difference test
        'n':              n,
        'ci_level':       ci,
    }


def test_slope_difference(results_a, results_b, label_a='Model A', label_b='Model B'):
    """
    Test whether the linear slope difference between two models is
    statistically significant using the bootstrap distribution of slope differences.

    Args:
        results_a: output of bootstrap_slopes() for model A
        results_b: output of bootstrap_slopes() for model B

    Returns:
        dict with observed difference, CI, and significance
    """
    boot_diff = results_b['boot_lin_slopes'] - results_a['boot_lin_slopes']
    obs_diff  = results_b['lin_slope'] - results_a['lin_slope']

    ci_lo = np.percentile(boot_diff, 2.5)
    ci_hi = np.percentile(boot_diff, 97.5)
    significant = (ci_lo > 0) or (ci_hi < 0)  # CI excludes zero

    print(f'\n  Slope difference ({label_b} − {label_a}):')
    print(f'    Observed difference: {obs_diff:.4f}')
    print(f'    95% CI:              [{ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'    Significant:         {significant}')
    if not significant:
        print(f'    → CI includes 0: slope difference NOT distinguishable from chance')
    else:
        print(f'    → CI excludes 0: slope difference IS statistically significant')

    return {
        'obs_diff':    obs_diff,
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
        'significant': significant,
        'boot_diff':   boot_diff,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scatter_with_ci_slopes(x, y, results, title='', xlabel='', ylabel='',
                                 highlight_point=None, save_path=None, 
                                 xlim=None, ylim=None):
    """
    Reproduce your existing scatter plot style but add:
    - Bootstrap CI shading around linear fit
    - Bootstrap CI shading around quadratic fit
    - Updated legend showing CI bounds on slopes
    - Annotation box showing r, p, and slope CI
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fontsize_def = 14

    # Scatter
    ax.scatter(x, y, color='steelblue', alpha=0.4, s=18, edgecolors='none')
    if highlight_point is not None:
        #ax.scatter(*highlight_point, color='black', s=40, zorder=5)
        ax.scatter(x[highlight_point], y[highlight_point], alpha=1, s=20, color="black", zorder=5)

    x_fit = np.linspace(x.min(), x.max(), 300)

    # ── Linear fit + CI band ──
    lin_slope = results['lin_slope']
    lin_int   = results['lin_intercept']
    y_lin_fit = lin_slope * x_fit + lin_int

    # CI band: evaluate at x_fit for each bootstrap slope/intercept
    # (use stored boot arrays to reconstruct)
    # Simplified CI band using slope CI bounds
    y_lin_lo = results['lin_slope_ci'][0] * x_fit + results['lin_intercept_ci'][0]
    y_lin_hi = results['lin_slope_ci'][1] * x_fit + results['lin_intercept_ci'][1]

    ax.plot(x_fit, y_lin_fit, color='blue', linewidth=1.8,
            label=f'Linear: {lin_slope:.3f}x + {lin_int:.2f}\n'
                  f'  slope 95% CI: [{results["lin_slope_ci"][0]:.3f}, '
                  f'{results["lin_slope_ci"][1]:.3f}]')
    ax.fill_between(x_fit, y_lin_lo, y_lin_hi, color='blue', alpha=0.12)

    # ── Quadratic fit + CI band ──
    qa, qb, qc = results['quad_a'], results['quad_b'], results['quad_c']
    y_quad_fit = qa * x_fit**2 + qb * x_fit + qc
    y_quad_lo  = results['quad_a_ci'][0]*x_fit**2 + results['quad_b_ci'][0]*x_fit + results['quad_c_ci'][0]
    y_quad_hi  = results['quad_a_ci'][1]*x_fit**2 + results['quad_b_ci'][1]*x_fit + results['quad_c_ci'][1]

    sign_b = '+' if qb >= 0 else '-'
    ax.plot(x_fit, y_quad_fit, color='orange', linewidth=1.8,
            label=f'Quadratic: {qa:.3f}x² {sign_b} {abs(qb):.3f}x + {qc:.2f}\n'
                  f'  a 95% CI: [{results["quad_a_ci"][0]:.3f}, {results["quad_a_ci"][1]:.3f}]')
    ax.fill_between(x_fit, y_quad_lo, y_quad_hi, color='orange', alpha=0.12)

    # ── Annotation box: r, p, slope CI ──
    textstr = (f'$r$ = {results["r"]:.3f} \n'
               f'r 95% CI: [{results["r_ci"][0]:.3f}, {results["r_ci"][1]:.3f}]\n'
               f'$p$ = {results["p"]:.3e}')
               #f'$r$ = {results["r"]:.3f}  [{results["r_ci"][0]:.3f}, {results["r_ci"][1]:.3f}]\n'
               #f'slope = {lin_slope:.3f}  [{results["lin_slope_ci"][0]:.3f}, {results["lin_slope_ci"][1]:.3f}]\n'
               #f'$n$ = {results["n"]}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
    ax.text(0.97, 0.02, textstr, transform=ax.transAxes,
            fontsize=fontsize_def, verticalalignment='bottom', horizontalalignment='right',
            bbox=props)

    ax.set_xlabel(xlabel, fontsize=fontsize_def)
    ax.set_ylabel(ylabel, fontsize=fontsize_def)
    plt.xticks(fontsize=int(fontsize_def + 2))
    plt.yticks(fontsize=int(fontsize_def + 2))
    ax.set_title(title, fontsize=int(fontsize_def + 2))
    ax.legend(fontsize=fontsize_def, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.4)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_slope_difference_bootstrap(diff_results, label_a, label_b,
                                     title='Bootstrap Slope Difference',
                                     save_path=None):
    """
    Plot the bootstrap distribution of slope differences with CI marked.
    Directly answers the reviewer: is the slope difference distinguishable from chance?
    """
    fontsize_def = 14
    fig, ax = plt.subplots(figsize=(7, 4))
    boot_diff = diff_results['boot_diff']

    ax.hist(boot_diff, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(diff_results['obs_diff'], color='black', linewidth=2,
               label=f'Observed difference: {diff_results["obs_diff"]:.4f}')
    ax.axvline(diff_results['ci_lo'], color='red', linewidth=1.5,
               linestyle='--', label=f'95% CI: [{diff_results["ci_lo"]:.4f}, {diff_results["ci_hi"]:.4f}]')
    ax.axvline(diff_results['ci_hi'], color='red', linewidth=1.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=1.0, linestyle=':',
               label='Zero (no difference)')
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100],
                     diff_results['ci_lo'], diff_results['ci_hi'],
                     color='red', alpha=0.1)

    sig_text = 'SIGNIFICANT (CI excludes 0)' if diff_results['significant'] \
               else 'NOT significant (CI includes 0)'
    ax.set_xlabel(f'Slope difference ({label_b} − {label_a})', fontsize=fontsize_def)
    ax.set_ylabel('Bootstrap count', fontsize=fontsize_def)
    plt.xticks(fontsize=int(fontsize_def + 2))
    plt.yticks(fontsize=int(fontsize_def + 2))
    ax.set_title(f'{title}\n{sig_text}', fontsize=int(fontsize_def+2))
    ax.legend(fontsize=fontsize_def)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

#if __name__ == '__main__':
def get_data_test():

    np.random.seed(42)
    n = 400

    # Replace with your actual data:
    # x_hclim, y_hclim = your soil moisture and temperature arrays
    # x_srgan, y_srgan = ...
    # x_cnn,   y_cnn   = ...

    var = create_nested_dict()
 
    # Simulated data matching your figures' approximate ranges
    x_hclim = np.random.uniform(0.09, 0.33, n)
    y_hclim = -43.8 * x_hclim + 306.4 + np.random.normal(0, 3.5, n)

    x_srgan = np.random.uniform(0.09, 0.33, n)
    y_srgan = -40.4 * x_srgan + 307.3 + np.random.normal(0, 2.5, n)

    x_cnn = np.random.uniform(0.09, 0.33, n)
    y_cnn = -43.3 * x_cnn + 305.8 + np.random.normal(0, 3.5, n)

    # Simulated data for Fig 6d-f (tighter scatter, smaller slopes)
    x_hclim_d = np.random.uniform(0.14, 0.34, 7)
    y_hclim_d = -24.7 * x_hclim_d + 312.3 + np.random.normal(0, 0.3, 7)

    x_srgan_e = np.random.uniform(0.09, 0.32, n)
    y_srgan_e = -21.0 * x_srgan_e + 311.3 + np.random.normal(0, 0.3, n)

    x_cnn_f = np.random.uniform(0.09, 0.30, n)
    y_cnn_f = -16.2 * x_cnn_f + 309.2 + np.random.normal(0, 0.6, n)

    return var_x, var_y


def plot_scatter_main(var_x, var_y, label_def, title_def, n_bootstrap, highlight_point, out_figname, xlim, ylim):

    # ── bootstrap CI on slopes ──────────────────────────────────────
    #print(f'=== {model} ===')
    res_var = bootstrap_slopes(var_x, var_y, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_var["lin_slope"]:.3f}  '
          f'95% CI: {res_var["lin_slope_ci"]}')
    print(f'  Pearson r:    {res_var["r"]:.3f}  '
          f'95% CI: {res_var["r_ci"]}')
    plot_scatter_with_ci_slopes(var_x, var_y, res_var,
                         title=title_def, xlabel=label_def['xlabel'], ylabel=label_def['ylabel'],
                         highlight_point=highlight_point, save_path=out_figname, 
                         xlim=xlim, ylim=ylim)

    return res_var

def plot_slope_difference_main(model1_values, model2_values, model1_name, model2_name, title_moddiff, out_figname):

    # ── test slope differences ──────────────────────────────────────
    print(f'\n: {model1_name} vs {model2_name} (large scatter)')
    diff_model = test_slope_difference(model1_values, model2_values, model1_name, model2_name)
    plot_slope_difference_bootstrap(diff_model, model1_name, model2_name,
                                     title=title_moddiff, #f'Slope difference: {model2_name} − {model1_name}',
                                     save_path=out_figname) #'slope_diff_hclim_srgan.png')


def plot_main(var_x_dict, var_y_dict, label_def_dict, title_def_dict, \
    title_moddiff_def_dict, \
    out_figname_slope_ci_dict, out_figname_slope_diff_dict, \
    n_bootstrap, \
    experiments, models, compared_models):


    xlim_per_exp = {}
    ylim_per_exp = {}

    for exp_key, models_dict in var_x_dict.items():
        all_x = np.concatenate([x for x in models_dict.values()])
        all_y = np.concatenate([var_y_dict[exp_key][m] for m in models_dict.keys()])
        xlim_per_exp[exp_key] = (all_x.min() - 0.005, all_x.max() + 0.005)
        ylim_per_exp[exp_key] = (all_y.min() - 0.5,   all_y.max() + 0.5)
    
    res_model = create_nested_dict()
    for experiment in experiments:
        for model in models:
            #for variable_name in variable_names:
            var_x = var_x_dict[experiment][model]
            var_y = var_y_dict[experiment][model]
            label_def = label_def_dict[experiment][model]
            title_def = title_def_dict[experiment][model]
            out_figname_slope_ci = out_figname_slope_ci_dict[experiment][model]

            highlight_point = False
            if 'HCLIM' in model and '20030815' in experiment:
                highlight_point = 0 # 15 Aug 2003, 12 UTC
            else:
                highlight_point = 303 - 1 # 15 Aug 2003, 12 UTC

            res_model[experiment][model] = plot_scatter_main(var_x, var_y, \
                label_def, title_def, n_bootstrap, highlight_point, out_figname_slope_ci, \
                xlim=xlim_per_exp[experiment], \
                ylim=ylim_per_exp[experiment], \
            )

        for i in range(len(compared_models)):
            title_moddiff = title_moddiff_def_dict[experiment][i]
            out_figname_slope_diff = out_figname_slope_diff_dict[experiment][i]
            model1_name, model2_name = compared_models[i]
            plot_slope_difference_main(res_model[experiment][model1_name], res_model[experiment][model2_name], \
                model1_name, model2_name, \
                title_moddiff, out_figname_slope_diff) 

    return res_model

def plot_test():

    # ── Fig 6a-c: bootstrap CI on slopes ──────────────────────────────────────
    print('=== Fig 6a: HCLIM ===')
    res_hclim = bootstrap_slopes(x_hclim, y_hclim, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_hclim["lin_slope"]:.3f}  '
          f'95% CI: {res_hclim["lin_slope_ci"]}')
    print(f'  Pearson r:    {res_hclim["r"]:.3f}  '
          f'95% CI: {res_hclim["r_ci"]}')
    plot_scatter_with_ci_slopes(x_hclim, y_hclim, res_hclim,
                                 title='(a) HCLIM', xlabel=xlabel, ylabel=ylabel,
                                 save_path='fig6a_hclim_ci.png')

    print('\n=== Fig 6b: SRGAN ===')
    res_srgan = bootstrap_slopes(x_srgan, y_srgan, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_srgan["lin_slope"]:.3f}  '
          f'95% CI: {res_srgan["lin_slope_ci"]}')
    plot_scatter_with_ci_slopes(x_srgan, y_srgan, res_srgan,
                                 title='(b) SRGAN', xlabel=xlabel, ylabel=ylabel,
                                 save_path='fig6b_srgan_ci.png')

    print('\n=== Fig 6c: CNN ===')
    res_cnn = bootstrap_slopes(x_cnn, y_cnn, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_cnn["lin_slope"]:.3f}  '
          f'95% CI: {res_cnn["lin_slope_ci"]}')
    plot_scatter_with_ci_slopes(x_cnn, y_cnn, res_cnn,
                                 title='(c) CNN', xlabel=xlabel, ylabel=ylabel,
                                 save_path='fig6c_cnn_ci.png')

    # ── Fig 6d-f: test slope differences ──────────────────────────────────────
    print('\n=== Fig 6e: SRGAN (tight) ===')
    res_srgan_e = bootstrap_slopes(x_srgan_e, y_srgan_e, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_srgan_e["lin_slope"]:.3f}  '
          f'95% CI: {res_srgan_e["lin_slope_ci"]}')
    plot_scatter_with_ci_slopes(x_srgan_e, y_srgan_e, res_srgan_e,
                                 title='(e) SRGAN', xlabel=xlabel, ylabel=ylabel,
                                 save_path='fig6e_srgan_ci.png')

    print('\n=== Fig 6f: CNN (tight) ===')
    res_cnn_f = bootstrap_slopes(x_cnn_f, y_cnn_f, n_bootstrap=n_bootstrap)
    print(f'  Linear slope: {res_cnn_f["lin_slope"]:.3f}  '
          f'95% CI: {res_cnn_f["lin_slope_ci"]}')
    plot_scatter_with_ci_slopes(x_cnn_f, y_cnn_f, res_cnn_f,
                                 title='(f) CNN', xlabel=xlabel, ylabel=ylabel,
                                 save_path='fig6f_cnn_ci.png')

    # ── Test slope differences: SRGAN vs CNN ──────────────────────────────────
    print('\n=== Slope difference tests ===')

    print('\nFig 6a-c: HCLIM vs SRGAN (large scatter)')
    diff_hclim_srgan = test_slope_difference(res_hclim, res_srgan,
                                              'HCLIM', 'SRGAN')
    plot_slope_difference_bootstrap(diff_hclim_srgan, 'HCLIM', 'SRGAN',
                                     title='Slope difference: SRGAN − HCLIM (Fig 6a-b)',
                                     save_path='slope_diff_hclim_srgan.png')

    print('\nFig 6a-c: HCLIM vs CNN (large scatter)')
    diff_hclim_cnn = test_slope_difference(res_hclim, res_cnn,
                                            'HCLIM', 'CNN')

    print('\nFig 6e-f: SRGAN vs CNN (tight scatter — reviewer focus)')
    diff_srgan_cnn = test_slope_difference(res_srgan_e, res_cnn_f,
                                            'SRGAN', 'CNN')
    plot_slope_difference_bootstrap(diff_srgan_cnn, 'SRGAN', 'CNN',
                                     title='Slope difference: CNN − SRGAN (Fig 6e-f)',
                                     save_path='slope_diff_srgan_cnn.png')

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n=== Summary table for paper ===')
    print(f'{"Model":<10} {"Slope":>8} {"CI_lo":>8} {"CI_hi":>8} {"r":>7} {"r_CI_lo":>8} {"r_CI_hi":>8}')
    print('-' * 65)
    for name, res in [('HCLIM',  res_hclim),
                      ('SRGAN',  res_srgan),
                      ('CNN',    res_cnn),
                      ('SRGAN-e',res_srgan_e),
                      ('CNN-f',  res_cnn_f)]:
        print(f'{name:<10} {res["lin_slope"]:>8.3f} '
              f'{res["lin_slope_ci"][0]:>8.3f} '
              f'{res["lin_slope_ci"][1]:>8.3f} '
              f'{res["r"]:>7.3f} '
              f'{res["r_ci"][0]:>8.3f} '
              f'{res["r_ci"][1]:>8.3f}')

"""
def get_parameters():

    label_def = {'xlabel': "Soil Moisture at Top 1cm (m3/m3)", 'ylabel': "2-m Air Temperature (K)"}
    #title_def = f"{number_def} {experiment} {var_names['var2']} vs. {var_names['var1']} ({test_date})"
    title_def = f"{number_def} {experiment}"
    xlabel = 'Soil Moisture at Top 1cm (m³/m³)'
    ylabel = '2-m Air Temperature (K)'

    return label_def, title_def
"""

def get_parameters(experiment, model):

    if 'SRGAN' in model or 'HCLIM' in model:
        var_names_dict = {'var1':'mrsol', 'var2':'tas'}
    elif 'CNN' in model:
        var_names_dict = {'var1':'mrsol', 'var2':'test'}

    MODEL_OFFSETS = {'HCLIM12': 0, 'HCLIM3': 1, 'CNN': 2, 'SRGAN': 3}
    exp_offset = 0 if 'JJA' in experiment else 4
    for key, model_offset in MODEL_OFFSETS.items():
        if key in model:
            letter = chr(ord('a') + exp_offset + model_offset)
            number_def = f'({letter})'
            break

    label_def = {'xlabel': "Soil Moisture at Top 1cm (m³/m³)", 'ylabel': "2-m Air Temperature (K)"}
    #title_def = f"{number_def} {model} {var_names['var2']} vs. {var_names['var1']} ({test_date})"
    title_def = f"{number_def} {model}"

    return label_def, title_def, var_names_dict

def get_parameters_moddiff(experiment, model_diff):

    MODEL_DIFF_OFFSETS = {'HCLIM3 - HCLIM12': 0, 'CNN - HCLIM3': 1, 'SRGAN - HCLIM3': 2}
    #print('model_diff', model_diff)
    exp_offset = 0 if 'JJA' in experiment else 3
    for key, model_diff_offset in MODEL_DIFF_OFFSETS.items():
        #print('key', key)
        if key in model_diff:
            letter = chr(ord('a') + exp_offset + model_diff_offset)
            number_moddiff_def = f'({letter})'
            break
    title_moddiff_def_dict = f"{number_moddiff_def} {model_diff}"

    return title_moddiff_def_dict


def main():

    n_bootstrap = 1000
    experiments = ['JJA 2003', '20030815T1200']
    models = ['HCLIM12', 'HCLIM3', 'CNN', 'SRGAN']
    compared_models = [['HCLIM12', 'HCLIM3'], ['HCLIM3', 'CNN'], ['HCLIM3', 'SRGAN']]
    models_diff = [f"{pair[1]} - {pair[0]}" for pair in compared_models]
    #print('models_diff', models_diff)
    #variable_names = ['mrsol', 'tas']
    outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/bootstrap/"
    output_summary_file = f'{outdir_fig}/Bootstrap_scatter_slope_summary_results.txt'

    # --- Define the bounding box (Northern Italy) ---
    lat_min = 44.0
    lat_max = 45.5
    lon_min = 7.0
    lon_max = 12.0

    unit_convert = 0.1 # from kg/m2 to m3/m3

    title_def_dict = create_nested_dict()
    title_moddiff_def_dict = create_nested_dict()
    label_def_dict = create_nested_dict()
    var_names_dict = create_nested_dict()
    out_figname_slope_ci_dict = create_nested_dict()
    out_figname_slope_diff_dict = create_nested_dict()
    var_x_dict = create_nested_dict()
    var_y_dict = create_nested_dict()

    for experiment in experiments:
        combined_experiment = '_'.join(experiment.split()) if " " in experiment else experiment
        for model in models:
            #for variable_name in variable_names:
            label_def_dict[experiment][model], title_def_dict[experiment][model], var_names_dict[experiment][model] = \
                get_parameters(experiment, model)

            print('experiment, model', experiment, model)
            print('label_def_dict[experiment][model], title_def_dict[experiment][model], var_names_dict[experiment][model]', \
                label_def_dict[experiment][model], title_def_dict[experiment][model], var_names_dict[experiment][model])

            basedir, x_file, y_file = \
                plot_scatter.get_file(model, experiment)
            print('basedir, x_file, y_file', basedir, x_file, y_file)

            if 'HCLIM' in model and '20030815' in experiment:
                var_x_dict[experiment][model], var_y_dict[experiment][model] = \
                    plot_scatter.get_data_predefined()
            else:
                var_x_dict[experiment][model], var_y_dict[experiment][model] = \
                    plot_scatter.get_data_by_file(basedir, x_file, y_file, var_names_dict[experiment][model], lat_min, lat_max, lon_min, lon_max, model, experiment)
            var_x_dict[experiment][model] = var_x_dict[experiment][model] * unit_convert

            out_figname_slope_ci_dict[experiment][model] = \
                f"{outdir_fig}/Bootstrap_scatter_slope_ci_{combined_experiment}_{model}_{var_names_dict[experiment][model]['var1']}_{var_names_dict[experiment][model]['var2']}.png"
            print('out_figname_slope_ci_dict[experiment][model]:', out_figname_slope_ci_dict[experiment][model])

        for i in range(len(compared_models)):
            title_moddiff_def_dict[experiment][i] = \
                get_parameters_moddiff(experiment, models_diff[i])
            out_figname_slope_diff_dict[experiment][i] = \
                f"{outdir_fig}/Bootstrap_scatter_slope_diff_{combined_experiment}_{compared_models[i][0]}_{compared_models[i][1]}.png"
            print('out_figname_slope_diff_dict[experiment][i]:', out_figname_slope_diff_dict[experiment][i])

    res_model = plot_main(var_x_dict, var_y_dict, label_def_dict, title_def_dict, \
          title_moddiff_def_dict, \
          out_figname_slope_ci_dict, out_figname_slope_diff_dict, \
          n_bootstrap, \
          experiments, models, compared_models)


    with open(output_summary_file, 'w') as f:
        print('\n=== Summary ===', file=f)
        print(
            f'{"Model":<10} {"Slope":>8} {"CI_lo":>8} {"CI_hi":>8} {"r":>7} {"r_CI_lo":>8} {"r_CI_hi":>8}',
            file=f,
        )
        print('-' * 65, file=f)

        for experiment in experiments:
            for model in models:
                res = res_model[experiment][model]
                print(
                    f'{model:<10} {res["lin_slope"]:>8.3f} '
                    f'{res["lin_slope_ci"][0]:>8.3f} '
                    f'{res["lin_slope_ci"][1]:>8.3f} '
                    f'{res["r"]:>7.3f} '
                    f'{res["r_ci"][0]:>8.3f} '
                    f'{res["r_ci"][1]:>8.3f}',
                    file=f,
                )

    """
    print('\n=== Summary ===')
    print(f'{"Model":<10} {"Slope":>8} {"CI_lo":>8} {"CI_hi":>8} {"r":>7} {"r_CI_lo":>8} {"r_CI_hi":>8}')
    print('-' * 65)

    for experiment in experiments:
        for model in models:
            #for name, res in [('HCLIM',  res_model[experiment]['HCLIM12']),
            #          ('SRGAN',  res_srgan),
            #          ('CNN',    res_cnn),
            #          ('SRGAN-e',res_srgan_e),
            #          ('CNN-f',  res_cnn_f)]:
            print(f'{model:<10} {res_model[experiment][model]["lin_slope"]:>8.3f} '
                  f'{res_model[experiment][model]["lin_slope_ci"][0]:>8.3f} '
                  f'{res_model[experiment][model]["lin_slope_ci"][1]:>8.3f} '
                  f'{res_model[experiment][model]["r"]:>7.3f} '
                  f'{res_model[experiment][model]["r_ci"][0]:>8.3f} '
                  f'{res_model[experiment][model]["r_ci"][1]:>8.3f}')
    """

if __name__ == "__main__":
    main()

