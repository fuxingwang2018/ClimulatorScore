import xarray as xr
import stats_tools
import numpy as np
import sys

def get_data(experiment_dict, var_name, unit_convert):

    # common period for GAN and CNN: 2009-01-01 to 2009-12-28
    index_range = {\
        'CNN':   [0, 1448], \
        'SRGAN': [1552, 3000], \
        'HCLIM': [13152, 14600], \
        }
    
    var_name_modify = {\
        'tas': {'CNN': 'test'}, \
        'hfls': {'CNN': 'test'}, \
        'mrsol': {'CNN': 'test'}, \
        'pr': {'CNN': 'test'}, \
        }

    experiment_val = {}
    for experiment_name, file_name in experiment_dict.items():
        ds = xr.open_dataset(file_name, decode_cf=False)

        if experiment_name == 'CNN 3km':
            experiment_val[experiment_name] = ds[var_name_modify[var_name]['CNN']].values * unit_convert[var_name]  # Convert to mm/day for pr
            experiment_val[experiment_name] = experiment_val[experiment_name][index_range['CNN'][0]:index_range['CNN'][1]]
        elif 'SRGAN 3km' in experiment_name:
            experiment_val[experiment_name] = ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
            experiment_val[experiment_name] = experiment_val[experiment_name][index_range['SRGAN'][0]:index_range['SRGAN'][1]]
        elif experiment_name == 'HCLIM 3km' or experiment_name == 'HCLIM 12km':
            experiment_val[experiment_name] = ds[var_name].values * unit_convert[var_name]  # Convert to mm/day for pr
            experiment_val[experiment_name] = experiment_val[experiment_name][index_range['HCLIM'][0]:index_range['HCLIM'][1]]
        
        if experiment_name == 'HCLIM 12km':
            experiment_val['HCLIM 12km'] = stats_tools.upsample_2d_array(experiment_val['HCLIM 12km'], upscale_factor = 4)

        #print('experiment_name:', experiment_name, experiment_val[experiment_name].shape, experiment_val[experiment_name])
        #print('experiment_name index:', experiment_val[experiment_name][0].shape)

        experiment_val[experiment_name] = np.array(experiment_val[experiment_name]) 

        #if experiment_name == 'HCLIM 3km':

    low_res_shape, high_res_shape = experiment_val['HCLIM 12km'].shape, experiment_val['HCLIM 3km'].shape

    #for key_hr, values_hr in var_high_res_adjusted_dict.items():
    residue_geo = []
    for i in range(len(low_res_shape)):
        residue_geo.append(high_res_shape[i] % low_res_shape[i])
        if residue_geo[i] != 0:
            print('Not divisible:', high_res_shape[i], low_res_shape[i], i, residue_geo[i])
            if i == 2 and residue_geo[i] == 2:
                experiment_val['HCLIM 3km'] = experiment_val['HCLIM 3km'][:, :, 1:-1] # remove the 1st and last elements

    return experiment_val
