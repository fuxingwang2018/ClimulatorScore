#!/bin/bash 
#SBATCH -A NAISS2026-4-912-gpu
###SBATCH -N 1 
#SBATCH -t 00:30:00
#SBATCH -J stem
#SBATCH --chdir=/nobackup/proj/disk/hclimai/personal/fuxing/log/log_stats/
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --cpus-per-task=16
#SBATCH --gpus 1
#SBATCH -p gpu
###SBATCH --mem-per-cpu=10G 
#SBATCH --mem-per-gpu=400G

DOMAIN='EmiliaRomagna'
#DOMAIN='TestDomain'
#VARIABLE='tas_mrsol' #'pr' #tas
VARIABLE='tas' #'pr' #tas
#GCM='ECE'
GCM='ERAI'
YEAR='2003'
echo 'domain is' ${DOMAIN}


current_date_time="`date`";
echo The run starts from $current_date_time

#module --force purge
#module load GPU/Python/3.13.5-bare-gcc-2025b-eb
#pip install xarray #xarray-2026.4.0
#pip install scipy #scipy-1.17.1
#pip install scikit-learn #scikit-learn-1.9.0
#pip install matplotlib #matplotlib-3.10.9
#pip install cartopy #cartopy-0.25.0
#pip install pyyaml #pyyaml-6.0.3
#pip install netcdf4 #netcdf4-1.7.4
source $HOME/venvs/climulatorscore/bin/activate

set -exu 

cd $HOME/ClimulatorScore/src

python main.py -c $HOME/ClimulatorScore/config/Arrhenius/config_${DOMAIN}_${VARIABLE}_${GCM}_${YEAR}_Arrhenius.yml

#cd $HOME/Script/HCLIMAI
#python -m pytest

current_date_time="`date`";
echo The run ends at $current_date_time

exit 0 

