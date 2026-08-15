import numpy as np
import stats_tools
import sys
import significance_test
from matplotlib.colors import LinearSegmentedColormap




def get_statistics(experiment_val, min_max_scale, abs_value_max_scale, variables, \
    reference_experiment, selected_statistics=None):

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

    # Define hex colors: Dry Brown -> Tan -> Light Green -> Cyan -> Deep Blue
    soil_colors = ['#543005', '#8c510a', '#dfc27d', '#80cdc1', '#01665e', '#003c30']
    custom_soil_cmap = LinearSegmentedColormap.from_list('SoilMoisture', soil_colors)

    # Default to all available statistics
    all_stats = ['rmse', 'mean_bias', 'variance_ratio', 'correlation', 'wasserstein', \
            'percentile_99', 'mean_value', 'abs_value', 'std', 'detection_metrics', \
            'percentile_99_bias', 'significance_pvalue']
    selected_statistics = selected_statistics or all_stats

    #if 'significance_pvalue' in selected_statistics and len(selected_statistics) == 1:
    #    reference_experiment = 'HCLIM 12km'
    #else:
    #    reference_experiment = 'HCLIM 3km'
    print('reference_experiment is:', reference_experiment)

    experiment_name_with_ref = list(experiment_val.keys())
    #experiment_name_without_ref = [name for name in experiment_val if name != ref_experiment]
    experiment_name_without_ref = [name for name in experiment_val if reference_experiment.get(name) is not None]
    
    # Function mapping for statistics calculations
    stat_functions = {
        'rmse': stats_tools.calculate_rmse,
        'mean_bias': stats_tools.calculate_mean_bias,
        'percentile_99_bias': stats_tools.calculate_99th_percentile_bias,
        'variance_ratio': stats_tools.calculate_ratio_of_variance,
        'correlation': stats_tools.calculate_correlation,
        'cpl_corr': stats_tools.calculate_correlation,
        'wasserstein': stats_tools.calculate_wasserstein_distance_rel,
        'significance_pvalue': significance_test.get_pvalue_of_significance_test,
        'detection_metrics': stats_tools.compute_metrics,
    }


    cmap_dict = {'tas': {'mean_value': 'cividis', 'percentile_99': 'inferno', 'mean_bias': 'seismic'}, \
                 'mrsol': {'mean_value': 'BrBG', 'percentile_99': 'Blues', 'mean_bias': 'seismic'}, \
                 'snc': {'mean_value': 'PuBu', 'percentile_99': 'YlGnBu', 'mean_bias': 'seismic'}, \
                 'snd': {'mean_value': 'PuBu', 'percentile_99': 'YlGnBu', 'mean_bias': 'seismic'}, \
                 'snw': {'mean_value': 'PuBu', 'percentile_99': 'YlGnBu', 'mean_bias': 'seismic'}, \
                 'pr': {'mean_value': 'Blues',    'percentile_99': 'BuPu', 'mean_bias': 'seismic'} }

    title_dict = {'snc': {'mean_bias': 'Difference'}, \
        'snw': {'mean_bias': 'Difference'}, \
        'snd': {'mean_bias': 'Difference'}, \
        'mrsol': {'mean_bias': 'Difference'}, \
        'tas': {'mean_bias': 'Mean Bias'}, \
        'pr': {'mean_bias': 'Mean Bias'}, }

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
     

    elif len(variables) == 1:
  
        for var_name in variables:

            """
            reference = experiment_val[ref_experiment][var_name]
            comparisons = {exp: vals[var_name] for exp, vals in experiment_val.items() if exp != ref_experiment}

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
            """

            # Experiments that have an assigned reference model
            comparisons = {exp: vals[var_name] for exp, vals in experiment_val.items() if reference_experiment.get(exp) is not None}

            # Compute statistics dynamically using each experiment's specific reference
            statistics = {
                stat: [
                    stat_functions[stat](experiment_val[reference_experiment[exp]][var_name], val)
                    for exp, val in comparisons.items()
                ]
                for stat in selected_statistics if stat in stat_functions
            }

            # Compute single-input statistics across all experiments
            all_exp_data = [vals[var_name] for vals in experiment_val.values()]
            
            for stat, func in {'percentile_99': stats_tools.calculate_99th_percentile,
                               'mean_value': stats_tools.calculate_mean_value,
                               'abs_value': stats_tools.calculate_abs_value,
                               'std': stats_tools.calculate_std}.items():
                if stat in selected_statistics:
                    statistics[stat] = [func(data) for data in all_exp_data]

    # Compute global color scale ranges
    #vmin_vmax = {stat: (np.nanmin(vals), np.nanmax(vals)) for stat, vals in statistics.items()}
    #vmin_vmax = {stat: \
    #    (np.nanmin(np.where((np.array(vals) > 1e10) | (np.array(vals) < -1e10), np.nan, np.array(vals))), \
    #    np.nanmax(np.where((np.array(vals) > 1e10) | (np.array(vals) < -1e10), np.nan, np.array(vals)))) \
    #    for stat, vals in statistics.items()}
    vmin_vmax = {}
    for stat, vals in statistics.items():
        arr = np.array(vals)
        arr_clean = np.where((arr > 1e10) | (arr < -1e10), np.nan, arr)
        vmin = np.nanmin(arr_clean)
        vmax = np.nanmax(arr_clean)
        vmin_vmax[stat] = (vmin, vmax) 
        if 'bias' in stat: 
            abs_max = min(abs(vmin), abs(vmax))   
            vmin_vmax[stat] = (-abs_max, abs_max) 

    print('vmin_vmax 1', type(vmin_vmax), vmin_vmax)
    vmin_vmax = {k: tuple(v) if v else vmin_vmax[k] for k, v in min_max_scale.items()}
    print('vmin_vmax 2', type(vmin_vmax), vmin_vmax)
    #if 'abs_value' in vmin_vmax:
    #    vmin_vmax['abs_value'] = (vmin_vmax['abs_value'][0], vmin_vmax['abs_value'][1] * abs_value_max_scale[var_name])

    
    # Metadata for visualization
    stat_meta = {
        'rmse': ('RMSE', 'rmse_maps', 'viridis'),
        'correlation': ('Correlation', 'correlation_maps', 'Reds'),
        'cpl_corr': ('Correlation', 'coupling_correlation_maps', 'seismic'),
        'mean_bias': (title_dict[variables[0]]['mean_bias'], 'mean_bias_maps', cmap_dict[variables[0]]['mean_bias']),
        'percentile_99_bias': ('99th Percentile Bias', 'percentile_99_bias_maps', 'seismic'),
        'variance_ratio': ('Ratio of Variance', 'variance_ratio_maps', 'Blues'),
        'wasserstein': ('Wasserstein Distance', 'wasserstein_maps', 'plasma'),
        'significance_pvalue': ('Significance P-value', 'significance_pvalue_maps', 'plasma'),
        'detection_metrics': ('Detection Metrics', 'detection_maps', 'plasma'),
        'percentile_99': ('99th Percentile', 'percentile_99_maps', cmap_dict[variables[0]]['percentile_99']), #tas
        'mean_value': ('Mean Value', 'mean_value_maps', cmap_dict[variables[0]]['mean_value']), #tas
        'abs_value': ('Abs Value', 'abs_value_maps', cmap_dict[variables[0]]['mean_value']),
        'std': ('Standard Deviation', 'standard_deviation_maps', 'Reds'),
    }

    #all_statistics = [(statistics[stat], *stat_meta[stat], vmin_vmax[stat][0], vmin_vmax[stat][1])
    #        for stat in selected_statistics if stat in statistics]

    all_statistics = [
        (statistics[stat], title, filename, *vmin_vmax[stat], cmap, 
         experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 else experiment_name_without_ref)
        for stat, (title, filename, cmap) in stat_meta.items() if stat in statistics
    ]

    """
    plot_counter = 0 

    active_stats = [(stat, meta) for stat, meta in stat_meta.items() if stat in statistics]

    all_statistics = []
    for stat, (title, filename, cmap) in active_stats:
        # 2. Use the counter to get the letter
        letter = chr(97 + plot_counter) 
        
        stat_entry = (
            statistics[stat], 
            f"({letter}) {title}", 
            filename, 
            *vmin_vmax[stat], 
            cmap,
            experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 
            else experiment_name_without_ref
        )
        all_statistics.append(stat_entry)
        
        # 3. Increment the counter so the NEXT plot gets the next letter
        plot_counter += 1
    """


    """
    plot_counter = 0 
    all_statistics = []
    for stat, (title, filename, cmap) in stat_meta.items():
        print ('plot_counter, title', plot_counter, title)
        if stat in statistics:
            # Use the counter to get the letter, then increment it
            letter = chr(97 + plot_counter) 
            new_title = f"({letter}) {title}"
            
            stat_tuple = (
                statistics[stat], 
                new_title, 
                filename, 
                *vmin_vmax[stat], 
                cmap,
                experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 
                else experiment_name_without_ref
            )
            all_statistics.append(stat_tuple)
            
            # Increment ONLY when a plot is successfully added
            plot_counter += 1 
    """


    """
    active_stats = [(stat, meta) for stat, meta in stat_meta.items() if stat in statistics]

    all_statistics = [
    (
        statistics[stat], 
        f"({chr(97 + i)}) {title}",  # i will now go 0, 1, 2... for active plots
        filename, 
        *vmin_vmax[stat], 
        cmap,
        experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 
        else experiment_name_without_ref
    )
    for i, (stat, (title, filename, cmap)) in enumerate(active_stats)
    ]
    """
    #all_statistics = [
    #    (
    #    statistics[stat], 
    #    f"({chr(97 + i)}) {title}",  # Adds (a), (b), (c)... to the title
    #    filename, 
    #    *vmin_vmax[stat], 
    #    cmap,
    #    experiment_name_with_ref if stat in {'percentile_99', 'mean_value', 'abs_value', 'std'} or len(variables) > 1 
    #    else experiment_name_without_ref
    #    )
    #for i, (stat, (title, filename, cmap)) in enumerate(stat_meta.items()) 
    #if stat in statistics
    #]
    #print('all_statistics', all_statistics)

    return all_statistics

