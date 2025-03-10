#!/bin/bash 
###SBATCH -N 1 
#SBATCH -t 01:00:00 
###SBATCH -n 1  ##ntasks 
#SBATCH -J stats 
#SBATCH --chdir=/ec/res4/scratch/smf/tmp/log_stats
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --gpus=1
#SBATCH --qos=nf
###SBATCH --mem-per-cpu=2G
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

current_date_time="`date`";
echo The run starts from $current_date_time

#source /nobackup/rossby24/users/sm_fuxwa/conda/miniconda3/bin/activate
#conda activate hclimai
#mamba activate ml_analysis

module load netcdf4/4.9.2
#module load python3/new cuda
module load cuda
source /perm/smf/venvs/climulator2/bin/activate

set -exu 

cd $HOME/Scripts/ClimulatorScore/src
python main.py

#cd $HOME/Script/HCLIMAI
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

