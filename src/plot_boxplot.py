import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import get_time_index
#from matplotlib import rcParams


def main():
    # --- Configuration ---
    # Update these names to match your actual experiment folders or identifiers
    base_path = "/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/"  # Path where your NetCDF files are stored
    outdir_fig = f"/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/boxplot/"
    var_names = {'var1':'mrsol', 'var2':'tas'}
    var_names_to_read = {'var1':'mrsol', 'var2':'tas'}

    #GCM = 'ECMWF-ERAINT'
    #GCM = "ICHEC-EC-EARTH_HIST"
    GCM = "ICHEC-EC-EARTH_RCP85_MC"
    #GCM = "ICHEC-EC-EARTH_RCP85_LC"
    if GCM == 'ECMWF-ERAINT':
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2000, 2009, 2000, 2009
        EXP_SRGAN = 'EPOCH100_tas_mrsol_scale_time_val0.1_batch50_lamb0.1_wtm_atos'
        title_def = '(k) ERAI-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_HIST":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 1995, 2005, 1995, 2005
        EXP_SRGAN = 'ECE/EPOCH100_tas_mrsol_wsmt_lnoise0.1_ECEHis_atos'
        title_def = '(l) ECE-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_RCP85_MC":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2040, 2050, 2040, 2050
        EXP_SRGAN = 'ECE/EPOCH100_tas_mrsol_wsmt_lnoise0.1_ECEFutMC_atos'
        title_def = '(m) ECE-MC2MC'
    elif GCM == "ICHEC-EC-EARTH_RCP85_LC":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2090, 2099, 2089, 2099

    file_dict = {
        'HCLIM3': {'mrsol': f'{base_path}/cropped/{GCM}/3km/6hr/mrsol/mrsol_3km_6hr_{FIRST_YEAR_3km}01010000-{LAST_YEAR_3km}12311800.nc',
                  'tas': f'{base_path}/cropped/{GCM}/3km/6hr/tas/tas_3km_6hr_{FIRST_YEAR_3km}01010000-{LAST_YEAR_3km}12311800.nc' },
        'HCLIM12': {'mrsol': f'{base_path}/cropped/{GCM}/12km/6hr/mrsol/mrsol_12km_6hr_{FIRST_YEAR_12km}01010000-{LAST_YEAR_12km}12311800.nc',
                  'tas': f'{base_path}/cropped/{GCM}/12km/6hr/tas/tas_12km_6hr_{FIRST_YEAR_12km}01010000-{LAST_YEAR_12km}12311800.nc' },
        'SRGAN':  {'mrsol': f'{base_path}SG/SRGAN_OUT/{EXP_SRGAN}/predictant_ypred_1.nc', 
                  'tas': f'{base_path}SG/SRGAN_OUT/{EXP_SRGAN}/predictant_ypred_1.nc' }, 
        #'CNN':  {'mrsol': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_mrsol_2009.nc', 
        #          'tas': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_tas_2009.nc' }, 
        # ERAI
        #'ERA5':  {'mrsol': f'/nobackup/rossby27/users/sm_fuxwa/ERA5/2009/tas_mrsol_ERA5_regrid_3km_2009_2009_timestd_dim.nc',
        #          'tas': f'/nobackup/rossby27/users/sm_fuxwa/ERA5/2009/tas_mrsol_ERA5_regrid_3km_2009_2009_timestd_dim.nc' },

        # ECE Hist
        #'CNN':  {'mrsol': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_mrsol_2009.nc',
        #          'tas': f'/nobackup/rossby27/users/sm_yicwa/DATA_shared/Climulator/Emulator_HCLIM_CRM_T_SM/cnn_prediction_tas_2009.nc' }, 

        # ECE FutMC
    }

    time_range = def_time_range(GCM)
    time_idx_range = get_time_index.get_time_index(time_range)
    experiment_names = file_dict.keys()
    fontsize_def = 20

    all_correlations = []
    # --- Data Processing ---
    for exp in experiment_names:
        if 'CNN' in exp:
            var_names_to_read = {'var1':'test', 'var2':'test'}
        print(f"Processing {exp}...")
        print (file_dict[exp][var_names['var2']]) 
        print (file_dict[exp][var_names['var1']]) 
        # 1. Open the separate files for tas and mrsol
        # Assuming file naming like: Exp1_tas.nc and Exp1_mrsol.nc
        #ds_tas = xr.open_dataset(os.path.join(base_path, f"{exp}_tas.nc"))
        #ds_mrsol = xr.open_dataset(os.path.join(base_path, f"{exp}_mrsol.nc"))
        ds_tas = xr.open_dataset(file_dict[exp][var_names['var2']])
        ds_mrsol = xr.open_dataset(file_dict[exp][var_names['var1']])
        
        # 2. Extract DataArrays
        #tas = ds_tas[var_names_to_read['var2']]
        #mrsol = ds_mrsol[var_names_to_read['var1']]
        tas = ds_tas[var_names_to_read['var2']][time_idx_range[str(exp)][var_names['var2']]['start_idx'][0]:time_idx_range[str(exp)][var_names['var2']]['end_idx'][0]]
        mrsol = ds_mrsol[var_names_to_read['var1']][time_idx_range[str(exp)][var_names['var1']]['start_idx'][0]:time_idx_range[str(exp)][var_names['var1']]['end_idx'][0]]
        
        # 3. Calculate Pearson Correlation per grid point along the 'time' dimension
        # xr.corr automatically aligns coordinates and computes correlation over the specified dim
        cor_map = xr.corr(tas, mrsol, dim='time')
        
        # 4. Flatten the map into a 1D array of values and remove NaNs (e.g., over oceans or missing data)
        cor_values = cor_map.values.flatten()
        cor_values = cor_values[~np.isnan(cor_values)]
        
        all_correlations.append(cor_values)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    #plt.rcParams['xtick.labelsize'] = 15
    #plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams.update({'font.size': fontsize_def})

    # Create the boxplot
    # patch_artist=True allows us to fill the boxes with color
    bp = plt.boxplot(all_correlations, labels=experiment_names, patch_artist=True,
                     showmeans=True, meanline=True, 
                     medianprops={'color': 'black', 'linewidth': 2},
                     flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})

    # Customize box colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Final Polish

    plt.ylim(-1, 1)  # As requested, y-axis ranging from 0 to 1
    plt.ylabel("Correlation ($r$)", fontsize=fontsize_def)
    #plt.xlabel("Experiment", fontsize=12)
    plt.title(title_def, fontsize=fontsize_def, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{outdir_fig}/Boxplot_{GCM}_{var_names['var1']}_{var_names['var2']}.png", dpi=300, bbox_inches='tight')
    #plt.show()


def def_time_range(GCM):

    time_range_erai = {'HCLIM12': 
        {'mrsol': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2000-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'ERA5': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2009-01-01 00:00'], 'end_date': ['2009-12-31 24:00'], 'step_hours': 6}}

    time_range_ecehi2hi = {'HCLIM12': 
        {'mrsol': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['1995-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2005-01-01 00:00'], 'end_date': ['2005-12-31 24:00'], 'step_hours': 6}}

    time_range_ecemc2mc = {'HCLIM12': 
        {'mrsol': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'HCLIM3': 
        {'mrsol': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2040-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'SRGAN': 
        {'mrsol': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
         'CNN': 
        {'mrsol': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}, 
           'tas': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}, 
        'target': {'start_date': ['2050-01-01 00:00'], 'end_date': ['2050-12-31 24:00'], 'step_hours': 6}}

    if GCM == 'ECMWF-ERAINT':
        time_range = time_range_erai
    elif GCM == "ICHEC-EC-EARTH_HIST":
        time_range = time_range_ecehi2hi
    elif GCM == "ICHEC-EC-EARTH_RCP85_MC":
        time_range = time_range_ecemc2mc
    elif GCM == "ICHEC-EC-EARTH_RCP85_LC":
        print ('LC time range not defined yet')

    return time_range


if __name__ == "__main__":
    main()
