#!/bin/bash 
#SBATCH -A NAISS2025-1-11  
#SBATCH -p alvis
#SBATCH -N 1 
###SBATCH --gpus-per-node=A40:1
###SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=T4:1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00 
#SBATCH -J st1vtas 
#SBATCH --chdir=/mimer/NOBACKUP/groups/naiss2025-6-138/HCLIMAI/log/log_stats/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --ntasks=1
###SBATCH -n 1  ##ntasks 
###SBATCH --mem=16G

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
#VARIABLE='tas_mrsol' #'pr' #tas
VARIABLE='tas' #'pr' #tas
#GCM='ECE'
GCM='ERAI'
echo 'domain is' ${DOMAIN}

module --force purge
#module load virtualenv/20.26.2-GCCcore-13.3.0
#module load Python/3.12.3-GCCcore-13.3.0
#module load netcdf4-python/1.7.1.post2-foss-2024a
module load virtualenv/20.23.1-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load netcdf4-python/1.6.4-foss-2023a
module load scikit-learn/1.4.2-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load xarray/2023.9.0-gfbf-2023a
module load PyYAML/6.0-GCCcore-12.3.0
source $HOME/venvs/climulatorscore/bin/activate

current_date_time="`date`";
echo The run starts from $current_date_time

set -exu 

cd $HOME/ClimulatorScore/src

python main.py -c $HOME/ClimulatorScore/config/Alvis/config_${DOMAIN}_${VARIABLE}_${GCM}_Alvis.yml
#python main.py -c $HOME/ClimulatorScore/config/config_${DOMAIN}_${VARIABLE}_${GCM}_paper_Freja.yml
#python main.py -c $HOME/ClimulatorScore/config/config_${DOMAIN}_${VARIABLE}_${GCM}_2003_Freja.yml

#python taylor_climulator.py
#python plot_scatter.py

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

