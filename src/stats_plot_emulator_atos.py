import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
import stats_tools

experiment = 'test_algorithm'
var_name = 'tas' #'mrsol' #'tas' #pr
unit_convert = {'pr': 86400.0, 'tas': 1.0, 'mrsol': 1.0,}

# Load reference and comparison data
#base_dir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/'
#base_dir = '/perm/smf/HCLIMAI/Test_Domain/'
base_dir = '/perm/smf/HCLIMAI/Emilia_Romagna/'

if var_name == 'pr':
    #comp_experiment_list1 = [ 'SG_nopr', 'SG_withprn', 'ViT', 'Swin', 'Swinn', 'EffNetV2', 'EffNetV2rev', 'EffNetV2rev_withpr', 'HCLIM 12km']
    #comp_experiment_list2 = [ 'HCLIM 3km', 'SG_nopr',  'SG_withprn', 'ViT', 'Swin', 'Swinn', 'EffNetV2', 'EffNetV2rev', 'EffNetV2rev_withpr', 'HCLIM 12km']
    comp_experiment_list1 = [ 'SG_1gpu', 'SG_2gpu', 'SG_npr_50', 'SG_npr_50_drop', 'SG_npr_50_drop_lr1e-5', 'SG_npr_50_drop_lr1e-5_earlystop', 'SG_pr_50_drop0.3_lr1e-5_earlystop_discdrop_genalldrop', 'HCLIM 12km']
    comp_experiment_list2 = [ 'HCLIM 3km', 'SG_1gpu',  'SG_2gpu', 'SG_npr_50', 'SG_npr_50_drop', 'SG_npr_50_drop_lr1e-5', 'SG_npr_50_drop_lr1e-5_earlystop', 'SG_pr_50_drop0.3_lr1e-5_earlystop_discdrop_genalldrop', 'HCLIM 12km']

elif var_name == 'tas':
    comp_experiment_list1 = [ 'SG', 'SG_tas', 'HCLIM 12km']
    comp_experiment_list2 = [ 'HCLIM 3km', 'SG', 'SG_tas', 'HCLIM 12km']

elif var_name == 'mrsol':
    comp_experiment_list1 = [ 'SG', 'HCLIM 12km']
    comp_experiment_list2 = [ 'HCLIM 3km', 'SG', 'HCLIM 12km']

comp_experiment_list = { \
    'RMSE Maps': comp_experiment_list1, \
    'Correlation Maps': comp_experiment_list1, \
    'Mean Bias Maps': comp_experiment_list1, \
    'Ratio of Variance Maps': comp_experiment_list1, \
    'Wasserstein Distance Maps': comp_experiment_list1, \
    '99th Percentile Maps': comp_experiment_list2, \
    'Mean Value Maps': comp_experiment_list2, \
    'Abs Value Maps': comp_experiment_list2, \
    }

abs_value_max_scale = {'pr':0.5, 'tas':1.0, 'mrsol':1.0}

if var_name == 'pr':

    reference_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu'
    reference_file = "predictant_ytest.nc"

    reference_lowres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu'
    reference_lowres_file = "predictor.nc"

    comparison_files = [\
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_2gpu/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_lossv1/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_wBCE0.5/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_l21e-4/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.5_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.1_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_disc64/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_discdrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop_discdrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop_discdrop_gennodrop/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH5_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/Swin/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/Swin/SRGAN_OUT/EPOCH100n_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_rev_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_rev_{var_name}_with_pr/predictant_ypred.nc', \
                   ]
#                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_pr_tas_with_pr/predictant_ypred.nc', \

elif var_name == 'tas':

    reference_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_earlystop'
    reference_file = "predictant_ytest.nc"

    reference_lowres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH1_{var_name}_with_tas_earlystop'
    reference_lowres_file = "predictor.nc"

    comparison_files = [\
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_earlystop/predictant_ypred.nc", \
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_with_{var_name}_earlystop/predictant_ypred.nc", \
                   ]

elif var_name == 'mrsol':

    reference_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH1_{var_name}_with_mrsol_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/'
    reference_file = "predictant_ytest.nc"
    reference_lowres_folder = str(base_dir) + f"/SG/SRGAN_OUT/EPOCH1_{var_name}_with_mrsol_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/"
    reference_lowres_file = "predictor.nc"

    comparison_files = [\
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/predictant_ypred.nc", \
                   ]


output_dir = str(base_dir) + "/statistic_figs"  # Update to desired output folder
os.makedirs(output_dir, exist_ok=True)

reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_val = reference_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr

reference_lowres_ds = xr.open_dataset(reference_lowres_folder + '/' + reference_lowres_file, decode_cf=False)
reference_lowres_var = reference_lowres_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
reference_low2highres_val = stats_tools.upsample_2d_array(reference_lowres_var, upscale_factor = 4)

comparison_val = []
for file in comparison_files:
    ds = xr.open_dataset(file, decode_cf=False)
    comparison_val.append(ds[var_name].values * unit_convert[var_name])  # Convert to mm/day for pr

comparison_val.append(reference_low2highres_val)

abs_val_all = [reference_val] + comparison_val

comparison_val = np.array(comparison_val)

# Calculate statistics
rmse_map = [stats_tools.calculate_rmse(reference_val, comparison) for comparison in comparison_val]
mean_bias_map = [stats_tools.calculate_mean_bias(reference_val, comparison) for comparison in comparison_val]
ratio_of_variance_map = [stats_tools.calculate_ratio_of_variance(reference_val, comparison) for comparison in comparison_val]
correlation_map = [stats_tools.calculate_correlation(reference_val, comparison) for comparison in comparison_val]
wasserstein_map = [stats_tools.calculate_wasserstein_distance_rel(reference_val, comparison) for comparison in comparison_val]
percentile_99_ref = stats_tools.calculate_99th_percentile(reference_val)
percentile_99_comparisons = [stats_tools.calculate_99th_percentile(comparison) for comparison in comparison_val]
mean_value_ref = stats_tools.calculate_mean_value(reference_val)
mean_value_comparisons = [stats_tools.calculate_mean_value(comparison) for comparison in comparison_val]
time_step = 108
abs_val_map = [arr[time_step, :, :] for arr in abs_val_all]
print('ref:', len([reference_val]), type([reference_val]), np.shape(reference_val))
print('comparison:', len(comparison_val), type(comparison_val))
print('abs_val_all:', len(abs_val_all), type(abs_val_all))
#sys.exit()

# Find global color scale ranges for all statistics
rmse_all = rmse_map
mean_bias_all = mean_bias_map
ratio_of_variance_all = ratio_of_variance_map
correlation_all = correlation_map
wasserstein_all = wasserstein_map

rmse_vmin, rmse_vmax = np.min(rmse_all), np.max(rmse_all)
mean_bias_vmin, mean_bias_vmax = np.min(mean_bias_all), np.max(mean_bias_all)
ratio_of_variance_vmin, ratio_of_variance_vmax = 50, 150 #np.min(ratio_of_variance_all), np.max(ratio_of_variance_all)
correlation_vmin, correlation_vmax = np.min(correlation_all), np.max(correlation_all)
wasserstein_vmin, wasserstein_vmax = np.min(wasserstein_all), np.max(wasserstein_all)
percentile_99_all = [percentile_99_ref] + percentile_99_comparisons
mean_value_all = [mean_value_ref] + mean_value_comparisons
percentile_99_vmin, percentile_99_vmax = np.min(percentile_99_all), np.max(percentile_99_all)
mean_value_vmin, mean_value_vmax = np.min(mean_value_all), np.max(mean_value_all)
abs_value_vmin, abs_value_vmax = np.min(abs_val_map), np.max(abs_val_map) * abs_value_max_scale[var_name]


# Prepare statistics for plotting
all_statistics = [
    (rmse_map, 'RMSE Maps', 'rmse_maps', rmse_vmin, rmse_vmax, 'viridis'),
    (correlation_map, 'Correlation Maps', 'correlation_maps', correlation_vmin, correlation_vmax, 'coolwarm'),
    (mean_bias_map, 'Mean Bias Maps', 'mean_bias_maps', mean_bias_vmin, mean_bias_vmax, 'seismic'),
    (ratio_of_variance_map, 'Ratio of Variance Maps', 'ratio_of_variance_maps', ratio_of_variance_vmin, ratio_of_variance_vmax, 'RdBu'),
    (wasserstein_map, 'Wasserstein Distance Maps', 'wasserstein_maps', wasserstein_vmin, wasserstein_vmax, 'plasma'),
    ([percentile_99_ref] + percentile_99_comparisons, '99th Percentile Maps', 'percentile_99_maps', percentile_99_vmin, percentile_99_vmax, 'inferno'),
    ([mean_value_ref] + mean_value_comparisons, 'Mean Value Maps', 'mean_value_maps', mean_value_vmin, mean_value_vmax, 'cividis'),
    (abs_val_map, 'Abs Value Maps', 'abs_value_maps', abs_value_vmin, abs_value_vmax, 'Blues'),
]


# Plot and save each statistics set
for stats, title, filename, vmin, vmax, cmap in all_statistics:
    output_path = os.path.join(output_dir, f'{filename}_{experiment}_{var_name}.png')
    stats_tools.plot_and_save_maps(stats, [f'{title} {comp_experiment_list[title][i]}' for i in range(len(stats))], output_path, vmin=vmin, vmax=vmax, cmap=cmap)


