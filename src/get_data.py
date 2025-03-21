import xarray as xr
import stats_tools
import numpy as np

def get_data(experiment_dict, var_name, unit_convert):

    """
    reference_highres_ds = xr.open_dataset(experiment_dict['HCLIM 3km'])
    reference_highres_val = reference_highres_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr

    reference_lowres_ds = xr.open_dataset(experiment_dict['HCLIM 12km'], decode_cf=False)
    reference_lowres_var = reference_lowres_ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
    reference_low2highres_val = stats_tools.upsample_2d_array(reference_lowres_var, upscale_factor = 4)
    """

    experiment_val = {}
    for experiment_name, file_name in experiment_dict.items():
        ds = xr.open_dataset(file_name, decode_cf=False)
        experiment_val[experiment_name] = ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
        if experiment_name == 'HCLIM 12km':
            experiment_val['HCLIM 12km'] = stats_tools.upsample_2d_array(experiment_val['HCLIM 12km'], upscale_factor = 4)
        experiment_val[experiment_name] = np.array(experiment_val[experiment_name]) 

    return experiment_val
