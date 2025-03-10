import numpy as np
import stats_tools

def get_statistics(experiment_val, abs_value_max_scale, var_name, selected_statistics=None):
    """
    Compute selected statistics between reference (HCLIM 3km) and other experiments.
    
    Parameters:
    - experiment_val (dict): Experimental data with experiment names as keys.
    - abs_value_max_scale (dict): Scaling factors for absolute values.
    - var_name (str): Variable name for scaling.
    - selected_statistics (list, optional): List of statistics to compute.

    Returns:
    - List of computed statistics with metadata.
    """

    # Default to all available statistics
    all_stats = ['rmse', 'mean_bias', 'variance_ratio', 'correlation', 'wasserstein', 'percentile_99', 'mean_value', 'abs_value']
    selected_statistics = selected_statistics or all_stats

    experiment_name_with_ref = list(experiment_val.keys())
    experiment_name_without_ref = [name for name in experiment_val if name != 'HCLIM 3km']

    reference = experiment_val['HCLIM 3km']
    comparisons = {k: v for k, v in experiment_val.items() if k != 'HCLIM 3km'}
    
    # Function mapping for statistics calculations
    stat_functions = {
        'rmse': stats_tools.calculate_rmse,
        'mean_bias': stats_tools.calculate_mean_bias,
        'variance_ratio': stats_tools.calculate_ratio_of_variance,
        'correlation': stats_tools.calculate_correlation,
        'wasserstein': stats_tools.calculate_wasserstein_distance_rel,
        'percentile_99': stats_tools.calculate_99th_percentile,
        'mean_value': stats_tools.calculate_mean_value,
    }

    # Compute statistics dynamically
    statistics = {stat: [stat_functions[stat](reference, val) for val in comparisons.values()]
                  for stat in selected_statistics if stat in stat_functions}

    # Include reference values where necessary
    for stat in ['percentile_99', 'mean_value']:
        if stat in selected_statistics:
            statistics[stat].insert(0, stat_functions[stat](reference))

    # Compute absolute values
    if 'abs_value' in selected_statistics:
        time_step = 108  # Fixed time step
        statistics['abs_value'] = [arr[time_step, :, :] for arr in [reference] + list(comparisons.values())]

    # Compute global color scale ranges
    vmin_vmax = {stat: (np.min(vals), np.max(vals)) for stat, vals in statistics.items()}
    if 'abs_value' in vmin_vmax:
        vmin_vmax['abs_value'] = (vmin_vmax['abs_value'][0], vmin_vmax['abs_value'][1] * abs_value_max_scale[var_name])

    # Metadata for visualization
    stat_meta = {
        'rmse': ('RMSE Maps', 'rmse_maps', 'viridis'),
        'correlation': ('Correlation Maps', 'correlation_maps', 'coolwarm'),
        'mean_bias': ('Mean Bias Maps', 'mean_bias_maps', 'seismic'),
        'variance_ratio': ('Ratio of Variance Maps', 'variance_ratio_maps', 'RdBu'),
        'wasserstein': ('Wasserstein Distance Maps', 'wasserstein_maps', 'plasma'),
        'percentile_99': ('99th Percentile Maps', 'percentile_99_maps', 'inferno'),
        'mean_value': ('Mean Value Maps', 'mean_value_maps', 'cividis'),
        'abs_value': ('Abs Value Maps', 'abs_value_maps', 'Blues'),
    }

    #all_statistics = [(statistics[stat], *stat_meta[stat], vmin_vmax[stat][0], vmin_vmax[stat][1])
    #        for stat in selected_statistics if stat in statistics]

    all_statistics = [
        (statistics[stat], title, filename, *vmin_vmax[stat], cmap, 
         experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value'} else experiment_name_without_ref)
        for stat, (title, filename, cmap) in stat_meta.items() if stat in statistics
    ]

    return all_statistics

