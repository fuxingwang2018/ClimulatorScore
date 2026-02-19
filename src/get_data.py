import xarray as xr
import stats_tools
import numpy as np
import sys
import pandas as pd
print("xarray:", xr.__version__)
print("pandas:", pd.__version__)

def get_data(experiment_dict, variables, unit_convert, time_idx_range):

    # common period for GAN and CNN: 2009-01-01 to 2009-12-28
    """
    index_range = {\
        'CNN':   [0, 1460], \
        'SRGAN': [0, 1460], \
        'SRGANCOR01': [0, 1460], \
        'SRGANCOR05': [0, 1460], \
        'HCLIM': [13152, 14612], \
        #'HCLIM': [14612, 16072], \
        }
    """

    var_name_modify = {\
        'tas': {'CNN': 'test'}, \
        'hfls': {'CNN': 'test'}, \
        'mrsol': {'CNN': 'test'}, \
        'pr': {'CNN': 'test'}, \
        }

    experiment_val = {}
    for experiment_name_origin, file_name in experiment_dict.items():
        method = experiment_name_origin.strip().split()[0]
        #ds = xr.open_dataset(file_name, decode_cf=False)
        ds = xr.open_dataset(file_name, decode_times=True, use_cftime=False)
        print('len variables', len(variables))
        if len(variables) > 1 and all(var in experiment_name_origin for var in variables):
            experiment_name = experiment_name_origin
        else:
            experiment_name = " ".join(experiment_name_origin.split()[:2])
            #experiment_val.setdefault(experiment_name, {})
        #experiment_val[experiment_name] = {}
        experiment_val.setdefault(experiment_name, {})
        
        lat= ds['lat'].values
        lon= ds['lon'].values
        print(ds.time.dtype)
        print(ds.time.attrs)

        #if experiment_name == 'CNN 3km':
        #    for var_name in variables:
        #        experiment_val[experiment_name][var_name] = ds[var_name_modify[var_name]['CNN']].values * unit_convert[var_name]  # Convert to mm/day for pr
        #        experiment_val[experiment_name][var_name] = experiment_val[experiment_name][var_name][index_range['CNN'][0]:index_range['CNN'][1]]
        #elif 'HCLIM 3km' in experiment_name_origin or 'HCLIM 12km' in experiment_name_origin:
        if len(variables) > 1 and all(var in experiment_name_origin for var in variables):
            for var_name in variables:
                #daily_ds = ds[var_name].resample(time='1D').mean()
                #experiment_val[experiment_name][var_name] = daily_ds.values * unit_convert[var_name]  # Convert to mm/day for pr
                experiment_val[experiment_name][var_name] = ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
                experiment_val[experiment_name][var_name] = experiment_val[experiment_name][var_name][time_idx_range[str(experiment_name)][var_name]['start_idx'][0]:time_idx_range[str(experiment_name)][var_name]['end_idx'][0]]
        else:
            var_name = experiment_name_origin.strip().split()[-1]
            print('var_name, experiment_name_origin, experiment_name', var_name, experiment_name_origin, experiment_name)
            if 'CNN' in experiment_name:
                #daily_ds = ds[var_name_modify[var_name][method]].resample(time='1D').mean()
                #experiment_val[experiment_name][var_name] = daily_ds.values #* unit_convert[var_name]  # Convert to mm/day for pr
                experiment_val[experiment_name][var_name] = ds[var_name_modify[var_name][method]].values #* unit_convert[var_name]  # Convert to mm/day for pr
            else:
                #print('type of ds[var_name]', type(ds[var_name]))
                #print('type of ds[var_name].values', type(ds[var_name].values))
                #daily_ds = ds[var_name].resample(time='1D').mean()
                #experiment_val[experiment_name][var_name] = daily_ds.values * unit_convert[var_name]  # Convert to mm/day for pr
                experiment_val[experiment_name][var_name] = ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
            experiment_val[experiment_name][var_name] = experiment_val[experiment_name][var_name][time_idx_range[str(experiment_name)][var_name]['start_idx'][0]:time_idx_range[str(experiment_name)][var_name]['end_idx'][0]]
            print('experiment_val:', experiment_val[experiment_name].keys())
        
            if 'HCLIM 12km' in experiment_name:
                experiment_val[experiment_name][var_name] = stats_tools.upsample_2d_array(experiment_val[experiment_name][var_name], upscale_factor = 4)


        experiment_val[experiment_name][var_name] = np.array(experiment_val[experiment_name][var_name]) 

        print('experiment_name:', experiment_name, var_name, experiment_val[experiment_name][var_name].shape, experiment_val[experiment_name][var_name])
        #print('experiment_name index:', experiment_val[experiment_name][var_name][0].shape)

    #print('experiment_val 12:', experiment_val['HCLIM 12km']) #, var_name, experiment_val[experiment_name][var_name].shape, experiment_val[experiment_name][var_name])
    #print('experiment_val 3:', experiment_val['HCLIM 3km']) #, var_name, experiment_val[experiment_name][var_name].shape, experiment_val[experiment_name][var_name])
    if 'HCLIM 12km' in experiment_name:
        low_res_shape, high_res_shape = experiment_val['HCLIM 12km'][variables[0]].shape, experiment_val['HCLIM 3km'][variables[0]].shape

    #for key_hr, values_hr in var_high_res_adjusted_dict.items():
    if 'HCLIM 12km' in experiment_name:
      for var_name in variables:
        residue_geo = []
        for i in range(len(low_res_shape)):
            #print ('high_res_shape[i] % low_res_shape[i]', var_name, i, high_res_shape[i], low_res_shape[i])
            residue_geo.append(high_res_shape[i] % low_res_shape[i])
            if residue_geo[i] != 0:
                print('Not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
                if i == 2 and residue_geo[i] == 2:
                    experiment_val['HCLIM 3km'][var_name] = experiment_val['HCLIM 3km'][var_name][:, :, 1:-1] # remove the 1st and last elements

    # Loop over nested dict
    #for experiment_name in experiment_val:
    #    for var_name in experiment_val[experiment_name]:
    #        arr = experiment_val[experiment_name][var_name]
    #        arr = np.where(np.abs(arr) > 1e10, np.nan, arr)  # replace too large/small with nan
    #        arr = np.where(np.abs(arr) < -1e10, np.nan, arr)  # replace too large/small with nan
    #        experiment_val[experiment_name][var_name] = arr

    return experiment_val, lat, lon
