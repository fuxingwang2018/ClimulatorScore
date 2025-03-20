# ClimulatorScore  
**Machine Learning-Based Climate Emulator Evaluation Score**  

## Overview  
**ClimulatorScore** is a toolkit for evaluating machine learning-based climate emulators for HCLIM. 
It provides tools to load experimental data, calculate statistics, and generate diagnostic plots for emulator performance assessment.  

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
2. Run using:  
```bash
bash run_main_statsplot.sh
