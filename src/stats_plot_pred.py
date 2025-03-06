import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import stats_tools
#from multiprocessing import Pool

# Define function to plot and save maps
def plot_and_save_maps(statistics, titles, output_file, vmin=None, vmax=None, cmap='coolwarm'):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        print(titles)
        print('stat', stat, len(stat))
        im = axes[i].imshow(stat, cmap=cmap, vmin=vmin[i], vmax=vmax[i])
        axes[i].set_title(title, fontsize=10)
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # Dynamically adjust colorbar size to match the axis height
        #cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        #cbar_height = axes[i].get_position().height  # Get the height of the axis
        #cbar.ax.set_aspect(cbar_height / cbar.ax.get_position().height)

    # Hide unused subplots if there are any
    for i in range(len(statistics), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

#with Pool() as pool:
#    metrics_list = pool.starmap(compute_single_metrics, [(pred, ground_truth) for pred in predictions])


extreme_threshold = 0.2
experiment = 'extreme_detection'
var_name = 'pr' #'tas' #pr
var_name_mask = 'prdmax_mask' #'tas' #pr
unit_convert = {'pr': 1.0, 'tas': 1.0}
unit_convert_ref = {'pr': 86400.0, 'tas': 1.0}

# Load reference and comparison data
base_dir = '/nobackup/rossby27/users/sm_fuxwa/Extreme_Detection/'
reference_folder = str(base_dir) + '/SRGAN_OUT/' + 'EPOCH50_6var' + '/'
reference_file = "predictant_ytest.nc"

comp_experiment = ['EPOCH100_6var',  'EPOCH100_6var_BS10', 'EPOCH100_5var', 'EPOCH100_1var', 'EPOCH50_6var', 'HCLIM12']
comp_experiment_list = { \
    'Accuracy Maps': comp_experiment, \
    'Precision Maps': comp_experiment, \
    'Recall Maps': comp_experiment, \
    'F1 Maps': comp_experiment, \
    'roc_auc Maps': comp_experiment, \
    }


reference_lowres_folder = str(base_dir) + '/input_data/CORDEXFPS_HCLIM12/12km/day/prdmax_mask/' 
reference_lowres_file = "prdmax_mask_alp-12_regrid_3km_hclim_ec-earth_his_moments_D_max_native_grid_1996-2005_ANN.nc"

comparison_files = [\
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_6var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_5var_BS10' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_5var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH100_1var' + '/predictant_ypred.nc', \
                    str(base_dir) + '/SRGAN_OUT/' + 'EPOCH50_6var' + '/predictant_ypred.nc', \
                   ]

output_dir = str(base_dir) + "/Figure/" + 'thres_' + str(extreme_threshold)  # Update to desired output folder
os.makedirs(output_dir, exist_ok=True)

reference_ds = xr.open_dataset(reference_folder + '/' + reference_file)
reference_var = reference_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr

reference_lowres_ds = xr.open_dataset(reference_lowres_folder + '/' + reference_lowres_file, decode_cf=False)
reference_lowres_var = reference_lowres_ds[var_name_mask].values * unit_convert[var_name]  # Convert to mm/day for pr
reference_low2highres = upsample_2d_array(reference_lowres_var, upscale_factor = 4)

comparison_var = []
for file in comparison_files:
    ds = xr.open_dataset(file)
    comparison_var.append(ds[var_name].values * unit_convert[var_name])  # Convert to mm/day for pr

comparison_var.append(reference_low2highres[-753:-53,:,:])
print('comparison_var.shape:', comparison_var[0].shape, comparison_var[1].shape, comparison_var[2].shape)
print('comparison_var type:', type(comparison_var[0]))

comparison_var = np.array(comparison_var)

# Compute metrics
#results = compute_metrics(reference_var, comparison_var, threshold=extreme_threshold)
results = stats_tools.compute_metrics_vectorized(reference_var, comparison_var, threshold=extreme_threshold)
print('results', results)


# Access results for each metric
accuracy = results["accuracy"]  # List of 2D arrays (latitude, longitude) for accuracy
precision = results["precision"]  # List of 2D arrays (latitude, longitude) for precision
recall = results["recall"]       # List of 2D arrays for recall
f1 = results["f1"]               # List of 2D arrays for F1 score
roc_auc = results["roc_auc"]     # List of 2D arrays for ROC-AUC

accuracy_vmin, accuracy_vmax = [np.min(accuracy)] * len(accuracy), [np.max(accuracy)] * len(accuracy)
precision_vmin, precision_vmax = [np.min(precision)] * len(precision), [np.max(precision)] * len(precision)
recall_vmin, recall_vmax = [np.min(recall)] * len(recall), [np.max(recall)] * len(recall)
f1_vmin, f1_vmax = [np.min(f1)] * len(f1), [np.max(f1)] * len(f1)
roc_auc_vmin, roc_auc_vmax = [np.min(roc_auc)] * len(roc_auc), [np.max(roc_auc)] * len(roc_auc)

print('accuracy', accuracy, type(accuracy))


all_statistics = [
    (accuracy, 'Accuracy Maps', 'accuracy_maps', accuracy_vmin, accuracy_vmax, 'viridis'),
    (precision, 'Precision Maps', 'precision_maps', precision_vmin, precision_vmax, 'coolwarm'),
    (recall, 'Recall Maps', 'recall_maps', recall_vmin, recall_vmax, 'seismic'),
    (f1, 'F1 Maps', 'f1_maps', f1_vmin, f1_vmax, 'YlGnBu'),
    (roc_auc, 'roc_auc Maps', 'roc_auc_maps', roc_auc_vmin, roc_auc_vmax, 'plasma'),
]

# Plot and save each statistics set
for stats, title, filename, vmin, vmax, cmap in all_statistics:
    output_path = os.path.join(output_dir, f'{filename}_{experiment}_thres_{extreme_threshold}_{var_name}.png')
    stats_tools.plot_and_save_maps(stats, [f'{title} {comp_experiment_list[title][i]}' for i in range(len(stats))], output_path, vmin=vmin, vmax=vmax, cmap=cmap)


