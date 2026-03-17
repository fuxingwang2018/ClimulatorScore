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
1. Modify paths in `main.py` for `base_dir` and `output_dir`.
2. Modify configuration file in `config/` for
   - `reference_highres_folder`, `reference_highres_file`:
   - `reference_lowres_folder`, `predictor.nc` :
   - `comparison_files`: 
4. Run using:  
```python src/main.py -c /nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/04-evaluation_fuxing/ClimulatorScore/config/config_EmiliaRomagna_Freja.yml```
or
```sbatch run/run_main_statsplot.sh```
