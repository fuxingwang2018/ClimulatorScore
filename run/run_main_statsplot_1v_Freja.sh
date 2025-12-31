#!/bin/bash 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
###SBATCH -n 1  ##ntasks 
###SBATCH --mem=16G
#SBATCH -J stat1v 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log_stats
###SBATCH --chdir=/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/04-evaluation_fuxing/ClimulatorScore
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --ntasks=1
#SBATCH -A rossby
###SBATCH --qos=low

module load netCDF-HDF5-utils/4.9.2-1.12.2-hpc1-intel-2023a-eb
#module load Mambaforge
module load Mambaforge/23.3.1-1-hpc1

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
#VARIABLE='tas_mrsol' #'pr' #tas
VARIABLE='tas' #'pr' #tas
#GCM='ECE'
GCM='ERAI'

current_date_time="`date`";
echo The run starts from $current_date_time

#source /nobackup/rossby24/users/sm_fuxwa/conda/miniconda3/bin/activate
#conda activate hclimai
#mamba activate ml_analysis
mamba activate climulatorscore

set -exu 

cd $HOME/Script/ClimulatorScore/src
#cd /nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/04-evaluation_fuxing/ClimulatorScore/src
#python main.py -c /nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/04-evaluation_fuxing/ClimulatorScore/config/config_${DOMAIN}_Freja.yml

#python main.py -c $HOME/Script/ClimulatorScore/config/config_${DOMAIN}_${VARIABLE}_Freja.yml
#python main.py -c $HOME/Script/ClimulatorScore/config/config_${DOMAIN}_${VARIABLE}_${GCM}_Freja.yml

#python taylor_climulator.py
python plot_scatter.py

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

