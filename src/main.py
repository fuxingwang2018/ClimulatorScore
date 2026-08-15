import os
import sys
import get_experiments
import get_data
import get_statistics
import plot_tools
import get_config
import argparse
import get_time_index

def main():

    # Set up argument parser to accept config file as a command-line argument
    parser = argparse.ArgumentParser(description="Run statistical analysis with a given config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # get config
    config = get_config.get_config(args.config)
    experiment = config['experiment']
    reference_experiment = config['reference_experiment']
    variables = config['variables']
    abs_value_max_scale = config['abs_value_max_scale']
    min_max_scale = config['min_max_scale']
    unit_convert = config['unit_convert']
    selected_statistics = config['statistics']
    output_dir = config['output_dir']
    experiment_files = config['data_path']
    time_range = config['time_range']
    fig_parameters = config['fig_parameters']
    GCM = config['GCM']
    print('fig_parameters', fig_parameters)
    print('time_range', time_range)

    time_idx_range = get_time_index.get_time_index_v2(time_range)
    print('time_idx_range:', time_idx_range)
    os.makedirs(output_dir, exist_ok=True)

    [experiment_val, lat, lon] = \
        get_data.get_data(experiment_files, variables, unit_convert, time_idx_range)

    all_statistics = get_statistics.get_statistics(experiment_val, \
        min_max_scale, abs_value_max_scale, variables, \
        reference_experiment, selected_statistics)


    # Plot and save each statistics set
    for stats, title, filename, vmin, vmax, cmap, exp_name in all_statistics:
        output_path = os.path.join(output_dir, f"{filename}_{experiment}_{'_'.join(variables)}.png")
        #plot_tools.plot_and_save_maps(stats, \
        #    [f'{title} {exp_name[i]}' for i in range(len(stats))], \
        #    output_path, vmin=vmin, vmax=vmax, cmap=cmap)
        #print('stats', stats)
        #print('exp_name', exp_name)
        if 'bias' in filename:
            full_title = [f'{title} {exp_name[i]} - {reference_experiment[exp_name[i]]}' for i in range(len(stats))]
        else:
            full_title = [f'{title} {exp_name[i]}' for i in range(len(stats))]

        plot_tools.plot_and_save_maps_latlon(stats, lat, lon,\
            full_title,\
            output_path, vmin=vmin, vmax=vmax, cmap=cmap, \
            fig_parameters=fig_parameters)
        if 'Correlation' in title:
            output_path = os.path.join(str(output_dir), f"Boxplot_{filename}_{GCM}_{'_'.join(variables)}.png")
            plot_tools.plot_and_save_boxplot(stats, \
                [f'{exp_name[i]}' for i in range(len(stats))],\
                GCM, output_path, \
                fig_parameters=fig_parameters)

if __name__ == "__main__":
    main()
