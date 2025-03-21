import os
import sys
import get_experiments
import get_data
import get_statistics
import plot_tools

def main():
    experiment = 'test_algorithm'
    var_name = 'tas' #'mrsol' #'tas' #pr
    abs_value_max_scale = {'pr':0.5, 'tas':1.0, 'mrsol':1.0}
    unit_convert = {'pr': 86400.0, 'tas': 1.0, 'mrsol': 1.0,}
    selected_statistics = [ \
            'rmse', 'correlation', 'mean_bias', \
            'variance_ratio', 'wasserstein', 'percentile_99', \
            'mean_value', 'abs_value']

    # Load reference and comparison data
    base_dir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/'
    #base_dir = '/perm/smf/HCLIMAI/Test_Domain/'
    #base_dir = '/perm/smf/HCLIMAI/Emilia_Romagna/'
    output_dir = str(base_dir) + "/statistic_figs"  # Update to desired output folder
    os.makedirs(output_dir, exist_ok=True)

    experiment_files = \
        get_experiments.get_experiments(base_dir, var_name)

    experiment_val = \
        get_data.get_data(experiment_files, var_name, unit_convert)

    all_statistics = get_statistics.get_statistics(experiment_val, \
        abs_value_max_scale, var_name, selected_statistics)

    # Plot and save each statistics set
    for stats, title, filename, vmin, vmax, cmap, exp_name in all_statistics:
        output_path = os.path.join(output_dir, f'{filename}_{experiment}_{var_name}.png')
        plot_tools.plot_and_save_maps(stats, \
            [f'{title} {exp_name[i]}' for i in range(len(stats))], \
            output_path, vmin=vmin, vmax=vmax, cmap=cmap)

if __name__ == "__main__":
    main()
