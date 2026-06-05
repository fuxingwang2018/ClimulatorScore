# ClimulatorScore  
**Machine Learning-Based Climate Emulator Evaluation Score**  

## Overview  
**ClimulatorScore** is a toolkit for evaluating machine learning-based climate emulators for HCLIM. 
It provides tools to load experimental data, calculate statistics, and generate diagnostic plots for emulator performance assessment.  

## Installation and Run
### On ALVIS/NAISS

```
 mkdir $HOME/venvs
 cd $HOME/venvs
 module load virtualenv/20.23.1-GCCcore-12.3.0
 virtualenv --system-site-packages climulatorscore
 pip install cartopy
 #pip install "numpy<2" #Solution for error: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.2 as it may crash.

```

### On ECMWF/Atos

```
 module load python3/new
 mkdir -p $PERM/venvs
 cd $PERM/venvs
 python3 -m venv --system-site-packages climulatorscore
 source $PERM/venvs/climulatorscore/bin/activate

```

## Repository Structure  

### Scripts and Main Program  
- **run_main_statsplot.sh** — Shell script to run `main.py` with SLURM.  
- **main.py** — The main program for executing evaluation workflows.  
  - `base_dir`: directory containing input data  
  - `output_dir`: directory to store output figures and results  

### Modules  
- **get_experiments.py** — Retrieves file paths for prediction results to be compared.  
- **get_data.py** — Loads and structures data using information from `get_experiments.py`.  
- **get_statistics.py** — Computes evaluation statistics for emulator performance.  
- **plot_tools.py** — Provides functions to plot and save figures for analysis.  
- **stats_tools.py** — Contains helper functions for statistical calculations.  

## How to Run  

 1. Copy run.sh and config file
 
    ```
    cd <path-to-climulatorscore>

    cp config/ATOS/config_TestDomain_tas_ERAI_Atos.yml config/ATOS/<name-of-the-new-config-file>

    cp run/ATOS/run_main_statsplot_1v_tas_Atos.sh run/ATOS/<name-of-the-new-run-file>
    ```

      For example, on ATOS, ```path-to-climulatorscore``` can be ```$HOME/Scripts/ClimulatorScore/``` where the repo is located.

     If needed, change file names of config_TestDomain_tas_ERAI_Atos.yml and run_main_statsplot_1v_tas_Atos.sh.

2. Modify configuration and run
    
    Modify configuration in config.yaml (e.g., config/ATOS/config_TestDomain_tas_ERAI_Atos.yml on Atos).

    Modify run.sh (e.g., run/ATOS/run_main_statsplot_1v_tas_Atos.sh file).

3. Run   

    ```
    cd <path-to-climulatorscore>
    sbatch run/ATOS/run_main_statsplot_1v_tas_Atos.sh
    ```
    (path-to-climulatorscore can be $HOME/Scripts/ClimulatorScore/ where the repo is located)

4. Output

    Outputs are set by base_dir and output_dir in config_*.ini. e.g., in config/ATOS/config_TestDomain_tas_ERAI_Atos.yml

    ```
    base_dir: "/ec/res4/scratch/smf/HCLIMAI/Test_Domain"
    output_dir: "${base_dir}/statistic_figs/1var"
    ```

