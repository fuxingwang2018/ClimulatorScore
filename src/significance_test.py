import numpy as np
from scipy import stats


def calculate_ttest_significance(reference, comparison, alpha=0.05):
    """
    Paired t-test at each grid point over time dimension.
    Appropriate when differences are normally distributed.

    Args:
        reference:   3D array (time, x, y)
        comparison:  3D array (time, x, y)
        alpha:       significance level (default 0.05)

    Returns:
        stat:        2D array (x, y) - t-statistic
        p_value:     2D array (x, y) - p-value
        significant: 2D boolean array (x, y) - True where difference is significant
    """
    stat, p_value = stats.ttest_rel(comparison, reference, axis=0)
    significant = p_value < alpha
    return stat, p_value, significant


def calculate_wilcoxon_significance(reference, comparison, alpha=0.05):
    """
    Wilcoxon signed-rank test at each grid point over time dimension.
    Non-parametric alternative to paired t-test — does not assume normality.

    Args:
        reference:   3D array (time, x, y)
        comparison:  3D array (time, x, y)
        alpha:       significance level (default 0.05)

    Returns:
        stat:        2D array (x, y) - Wilcoxon statistic
        p_value:     2D array (x, y) - p-value
        significant: 2D boolean array (x, y) - True where difference is significant
    """
    nx, ny = reference.shape[1], reference.shape[2]
    stat    = np.full((nx, ny), np.nan)
    p_value = np.full((nx, ny), np.nan)

    for i in range(nx):
        for j in range(ny):
            diff = comparison[:, i, j] - reference[:, i, j]
            if np.any(diff != 0):
                stat[i, j], p_value[i, j] = stats.wilcoxon(diff)

    significant = p_value < alpha
    return stat, p_value, significant


def test_normality(reference, comparison, alpha_normality=0.05):
    """
    Test normality of differences at each grid point using Shapiro-Wilk test.
    Shapiro-Wilk is most appropriate for n < 5000; falls back to D'Agostino-Pearson for larger samples.

    Args:
        reference:        3D array (time, x, y)
        comparison:       3D array (time, x, y)
        alpha_normality:  significance level for normality test (default 0.05)

    Returns:
        is_normal:        2D boolean array (x, y) - True where differences are normally distributed
        normality_pvalue: 2D array (x, y) - p-value of normality test
        test_used:        str - name of normality test used ('shapiro' or 'dagostino')
    """
    nx, ny = reference.shape[1], reference.shape[2]
    n_time = reference.shape[0]
    normality_pvalue = np.full((nx, ny), np.nan)

    # Choose normality test based on sample size
    # Shapiro-Wilk: accurate but limited to n < 5000
    # D'Agostino-Pearson: robust for larger samples
    if n_time < 5000:
        test_used = 'shapiro-wilk'
        test_fn = lambda diff: stats.shapiro(diff)
    else:
        test_used = 'dagostino-pearson'
        test_fn = lambda diff: stats.normaltest(diff)

    print(f'Normality test: {test_used} (n_time={n_time})')

    for i in range(nx):
        for j in range(ny):
            diff = comparison[:, i, j] - reference[:, i, j]
            _, normality_pvalue[i, j] = test_fn(diff)

    is_normal = normality_pvalue >= alpha_normality  # fail to reject H0 → normal
    return is_normal, normality_pvalue, test_used


def calculate_significance_auto(reference, comparison,
                                 alpha=0.05, alpha_normality=0.05):
    """
    Automatically test normality of differences at each grid point,
    then apply the appropriate significance test:
        - Normally distributed differences     → paired t-test
        - Non-normally distributed differences → Wilcoxon signed-rank test

    Args:
        reference:        3D array (time, x, y)
        comparison:       3D array (time, x, y)
        alpha:            significance level for the main test (default 0.05)
        alpha_normality:  significance level for normality test (default 0.05)

    Returns:
        stat:             2D array (x, y) - test statistic (t or W depending on grid point)
        p_value:          2D array (x, y) - p-value of chosen test
        significant:      2D boolean array (x, y) - True where difference is significant
        is_normal:        2D boolean array (x, y) - True where differences are normal
        normality_pvalue: 2D array (x, y) - p-value of normality test
        method_map:       2D array (x, y) - 0=ttest, 1=wilcoxon (which test was used per grid point)
        normality_test:   str - name of normality test used
    """
    nx, ny = reference.shape[1], reference.shape[2]

    # Step 1 — test normality of differences at every grid point
    print('Step 1: Testing normality of differences...')
    is_normal, normality_pvalue, normality_test = test_normality(
        reference, comparison, alpha_normality)

    n_normal     = is_normal.sum()
    n_nonnormal  = (~is_normal).sum()
    n_total      = nx * ny
    print(f'  Normal grid points:     {n_normal} / {n_total} ({100*n_normal/n_total:.1f}%)')
    print(f'  Non-normal grid points: {n_nonnormal} / {n_total} ({100*n_nonnormal/n_total:.1f}%)')

    # Step 2 — apply appropriate test per grid point
    print('Step 2: Applying significance tests...')
    print('  Normal grid points     → paired t-test')
    print('  Non-normal grid points → Wilcoxon signed-rank test')

    stat       = np.full((nx, ny), np.nan)
    p_value    = np.full((nx, ny), np.nan)
    method_map = np.zeros((nx, ny), dtype=int)  # 0=ttest, 1=wilcoxon

    # Vectorised t-test for all grid points first (fast)
    t_stat_all, p_ttest_all = stats.ttest_rel(comparison, reference, axis=0)

    for i in range(nx):
        for j in range(ny):
            if is_normal[i, j]:
                # Paired t-test
                stat[i, j]       = t_stat_all[i, j]
                p_value[i, j]    = p_ttest_all[i, j]
                method_map[i, j] = 0
            else:
                # Wilcoxon signed-rank
                diff = comparison[:, i, j] - reference[:, i, j]
                if np.any(diff != 0):
                    stat[i, j], p_value[i, j] = stats.wilcoxon(diff)
                method_map[i, j] = 1

    significant = p_value < alpha

    n_sig = np.nansum(significant)
    print(f'Step 3: Results (alpha={alpha}):')
    print(f'  Significant grid points: {n_sig} / {n_total} ({100*n_sig/n_total:.1f}%)')
    print(f'  Grid points tested with t-test:  {(method_map==0).sum()}')
    print(f'  Grid points tested with Wilcoxon: {(method_map==1).sum()}')

    return stat, p_value, significant, is_normal, normality_pvalue, method_map, normality_test


def get_pvalue_of_significance_test(reference, comparison, alpha=0.05, alpha_normality=0.05):
    
    stat, p_value, significant, is_normal, normality_pvalue, method_map, normality_test = \
            calculate_significance_auto(reference, comparison, alpha=0.05, alpha_normality=0.05)

    return p_value

# How it works   
# For each grid point (i, j):
#    differences = comparison[:, i, j] - reference[:, i, j]
#         ↓
#    Normality test (Shapiro-Wilk if n<5000, D'Agostino-Pearson if n≥5000)
#         ↓
#    p_normality >= 0.05          p_normality < 0.05
#    (normal differences)         (non-normal differences)
#         ↓                              ↓
#    paired t-test              Wilcoxon signed-rank

# Example usage
""" 
if __name__ == '__main__':
    np.random.seed(42)
    time, nx, ny = 100, 20, 15
    reference  = np.random.normal(0, 1, (time, nx, ny))
    comparison = np.random.normal(0.3, 1, (time, nx, ny))  # slight positive bias

    stat, p_value, significant, is_normal, norm_pval, method_map, norm_test = \
        calculate_significance_auto(reference, comparison, alpha=0.05, alpha_normality=0.05)

    print(f'\nNormality test used: {norm_test}')
    print(f'Significance map shape: {significant.shape}')
    print(f'Significant grid points: {significant.sum()} / {nx*ny}')

    # Optional: save results
    # np.save('significant.npy', significant)
    # np.save('p_value.npy', p_value)
    # np.save('method_map.npy', method_map)  # 0=ttest, 1=wilcoxon

"""
