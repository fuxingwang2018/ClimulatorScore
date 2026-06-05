#!/bin/bash 
#SBATCH -N 1
###SBATCH --nodes 1
#SBATCH -t 01:00:00 
#SBATCH -J sttas
#SBATCH --chdir=/ec/res4/scratch/smf/log/log_stats/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
##BATCH --mem-per-cpu=4G
###SBATCH --ntasks=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
###SBATCH --mem=80G

DOMAIN='TestDomain'
VARIABLE='tas' #'pr' #tas
#GCM='ECE'
GCM='ERAI'
echo 'domain is' ${DOMAIN}

module --force purge
#module load virtualenv/20.23.1-GCCcore-12.3.0
#module load Python/3.11.3-GCCcore-12.3.0
#module load netcdf4-python/1.6.4-foss-2023a
#module load scikit-learn/1.4.2-gfbf-2023a
#module load matplotlib/3.7.2-gfbf-2023a
#module load xarray/2023.9.0-gfbf-2023a
#module load PyYAML/6.0-GCCcore-12.3.0
source $PERM/venvs/climulatorscore/bin/activate

current_date_time="`date`";
echo The run starts from $current_date_time

set -exu 

cd $HOME/Scripts/ClimulatorScore/src

python main.py -c $HOME/Scripts/ClimulatorScore/config/ATOS/config_${DOMAIN}_${VARIABLE}_${GCM}_Atos.yml

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

