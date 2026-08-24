#!/bin/bash
#SBATCH -N 1 
#SBATCH -t 01:00:00 
###SBATCH -n 1  ##ntasks 
###SBATCH --mem=16G
#SBATCH -J preproc 
#SBATCH --chdir=/nobackup/rossby26/users/sm_fuxwa/AI/log_stats
#SBATCH --error=%x-%j.error 
#SBATCH --output=%x-%j.out
###SBATCH --ntasks=1
#SBATCH -A rossby
###SBATCH --qos=low

#module load CDO/2.3.0-eccodes-aec-cmor-fftw-hpc2-intel-2023a-eb
module load NCO/5.1.3-hpc1-gcc-2022a-eb

#CNN
basedir_cnn_out='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/CNN'
basedir_cnn_in='/nobackup/rossby27/users/sm_yicwa/DATA_shared/AIES_revision_aug2026/PR_ALLEXPS/'
file_cnn_hi2hi='ECE-HI2HI_prediction.nc' 
file_cnn_mc2mc='ECE-MC2MC_prediction.nc' 
file_cnn_hm2hm_hist='ECE-HM2HM_hist_2005_prediction.nc' 
file_cnn_hm2hm_mc='ECE-HM2HM_mc_2050_prediction.nc' 
file_cnn_hi2mc='ECE-HI2MC_mc_2050_prediction.nc' 
file_cnn_mc2hi='ECE-MC2HI_hist_2005_prediction.nc' 

# SRGAN
basedir_srgan_2005='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ECE/EPOCH100_pr_wpr_worog_ECEHist_atos/'
basedir_srgan_2050='/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ECE/EPOCH100_pr_wpr_worog_ECEFutMC_atos/'
file_srgan='predictant_ypred_1.nc'

#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_hi2hi} 
#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_mc2hi} 
#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_hm2hm_hist} 
#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_mc2mc} 
#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_hi2mc} 
#ncks -A -v time ${basedir_srgan_2005}/${file_srgan} ${basedir_cnn}/${file_cnn_hm2hm_mc} 

cdo settaxis,2005-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_hi2hi} ${basedir_cnn_out}/${file_cnn_hi2hi}
cdo settaxis,2005-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_mc2hi} ${basedir_cnn_out}/${file_cnn_mc2hi}
cdo settaxis,2005-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_hm2hm_hist} ${basedir_cnn_out}/${file_cnn_hm2hm_hist}
cdo settaxis,2050-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_mc2mc} ${basedir_cnn_out}/${file_cnn_mc2mc}
cdo settaxis,2050-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_hi2mc} ${basedir_cnn_out}/${file_cnn_hi2mc}
cdo settaxis,2050-01-01,03:00:00,6hour ${basedir_cnn_in}/${file_cnn_hm2hm_mc} ${basedir_cnn_out}/${file_cnn_hm2hm_mc}


