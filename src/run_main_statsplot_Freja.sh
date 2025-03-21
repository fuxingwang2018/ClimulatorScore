#!/bin/bash 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
###SBATCH -n 1  ##ntasks 
#SBATCH --mem=16G
#SBATCH -J stats 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log_stats
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --ntasks=1
#SBATCH -A rossby
###SBATCH --qos=low

module load netCDF-HDF5-utils/4.9.2-1.12.2-hpc1-intel-2023a-eb
#module load Mambaforge
module load Mambaforge/23.3.1-1-hpc1

current_date_time="`date`";
echo The run starts from $current_date_time

source /nobackup/rossby24/users/sm_fuxwa/conda/miniconda3/bin/activate
#conda activate hclimai
mamba activate ml_analysis

set -exu 

cd $HOME/Script/ClimulatorScore/src
python main.py

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

