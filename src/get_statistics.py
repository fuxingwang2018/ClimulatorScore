import numpy as np
import stats_tools
import sys

def get_statistics(experiment_val, min_max_scale, abs_value_max_scale, variables, selected_statistics=None):
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
    all_stats = ['rmse', 'mean_bias', 'variance_ratio', 'correlation', 'wasserstein', 'percentile_99', 'mean_value', 'abs_value', 'std']
    selected_statistics = selected_statistics or all_stats

    experiment_name_with_ref = list(experiment_val.keys())
    experiment_name_without_ref = [name for name in experiment_val if name != 'HCLIM 3km']

    
    # Function mapping for statistics calculations
    stat_functions = {
        'rmse': stats_tools.calculate_rmse,
        'mean_bias': stats_tools.calculate_mean_bias,
        'variance_ratio': stats_tools.calculate_ratio_of_variance,
        'correlation': stats_tools.calculate_correlation,
        'cpl_corr': stats_tools.calculate_correlation,
        'wasserstein': stats_tools.calculate_wasserstein_distance_rel,
        #'percentile_99': stats_tools.calculate_99th_percentile,
        #'mean_value': stats_tools.calculate_mean_value,
        #'abs_value': stats_tools.calculate_abs_value,
    }

    if len(variables) > 1:

        variables_data = {
            exp: (vals[variables[0]], vals[variables[1]])
            for exp, vals in experiment_val.items()
        }

        #print('experiment_val:', np.shape(experiment_val['HCLIM 3km']['tas']))
        #for exp, (v1, v2) in variables_data.items():
        #    print ('v1', exp, v1, type(v1), np.shape(v1), len(v1))

        # Compute statistics dynamically
        statistics = {
            stat: [
                stat_functions[stat](v1, v2)
                for exp, (v1, v2) in variables_data.items()
        ]
        for stat in selected_statistics
        if stat in stat_functions
        }

        ## Compute global color scale ranges
        #vmin_vmax = {stat: (np.min(vals), np.max(vals)) for stat, vals in statistics.items()}
        ##print('vmin_vmax 1', type(vmin_vmax), vmin_vmax)
        #vmin_vmax = {k: tuple(v) if v else vmin_vmax[k] for k, v in min_max_scale.items()}
        ##print('vmin_vmax 2', type(vmin_vmax), vmin_vmax)

    elif len(variables) == 1:
  
        for var_name in variables:

            reference = experiment_val['HCLIM 3km'][var_name]
            #comparisons = {k: v for k, v in experiment_val.items() if k != 'HCLIM 3km'}
            comparisons = {exp: vals[var_name] for exp, vals in experiment_val.items() if exp != "HCLIM 3km"}


            # Compute statistics dynamically
            statistics = {stat: [stat_functions[stat](reference, val) for val in comparisons.values()]
                  for stat in selected_statistics if stat in stat_functions}

            # Compute single-input statistics
            for stat, func in {'percentile_99': stats_tools.calculate_99th_percentile,
                       'mean_value': stats_tools.calculate_mean_value,
                       'abs_value': stats_tools.calculate_abs_value,
                       'std': stats_tools.calculate_std}.items():
                if stat in selected_statistics:
                    statistics[stat] = [func(data) for data in [reference] + list(comparisons.values())]

    # Compute global color scale ranges
    #vmin_vmax = {stat: (np.nanmin(vals), np.nanmax(vals)) for stat, vals in statistics.items()}
    vmin_vmax = {stat: \
        (np.nanmin(np.where((np.array(vals) > 1e10) | (np.array(vals) < -1e10), np.nan, np.array(vals))), \
        np.nanmax(np.where((np.array(vals) > 1e10) | (np.array(vals) < -1e10), np.nan, np.array(vals)))) \
        for stat, vals in statistics.items()}
    #print('vmin_vmax 1', type(vmin_vmax), vmin_vmax)
    vmin_vmax = {k: tuple(v) if v else vmin_vmax[k] for k, v in min_max_scale.items()}
    #print('vmin_vmax 2', type(vmin_vmax), vmin_vmax)
    #if 'abs_value' in vmin_vmax:
    #    vmin_vmax['abs_value'] = (vmin_vmax['abs_value'][0], vmin_vmax['abs_value'][1] * abs_value_max_scale[var_name])

    # Metadata for visualization
    stat_meta = {
        'rmse': ('RMSE Maps', 'rmse_maps', 'viridis'),
        'correlation': ('Correlation Maps', 'correlation_maps', 'Reds'),
        'cpl_corr': ('Coupling Correlation Maps', 'coupling_correlation_maps', 'seismic'),
        'mean_bias': ('Mean Bias Maps', 'mean_bias_maps', 'seismic'),
        'variance_ratio': ('Ratio of Variance Maps', 'variance_ratio_maps', 'Blues'),
        'wasserstein': ('Wasserstein Distance Maps', 'wasserstein_maps', 'plasma'),
        'percentile_99': ('99th Percentile Maps', 'percentile_99_maps', 'inferno'),
        'mean_value': ('Mean Value Maps', 'mean_value_maps', 'cividis'),
        'abs_value': ('Abs Value Maps', 'abs_value_maps', 'Reds'),
        'std': ('Standard Deviation Maps', 'standard_deviation_maps', 'Reds'),
    }

    #all_statistics = [(statistics[stat], *stat_meta[stat], vmin_vmax[stat][0], vmin_vmax[stat][1])
    #        for stat in selected_statistics if stat in statistics]

    all_statistics = [
        (statistics[stat], title, filename, *vmin_vmax[stat], cmap, 
         experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 else experiment_name_without_ref)
        for stat, (title, filename, cmap) in stat_meta.items() if stat in statistics
    ]
    #print('all_statistics', all_statistics)

    return all_statistics

