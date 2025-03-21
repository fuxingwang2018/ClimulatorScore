

def get_experiments(base_dir, var_name):

    if var_name == 'pr':

        reference_highres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu'
        reference_highres_file = "predictant_ytest.nc"

        reference_lowres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu'
        reference_lowres_file = "predictor.nc"

        comparison_files = [\
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_1gpu/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_2gpu/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_lossv1/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_with_pr_wBCE0.5/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_l21e-4/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.5_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.1_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_disc64/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_discdrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop_discdrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/SG/SRGAN_OUT/EPOCH50_{var_name}_with_pr_dropout0.3_lr1e-5_earlystop_discdrop_gennodrop/predictant_ypred.nc', \
                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH5_{var_name}_no_pr_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/predictant_ypred.nc', \
                    #str(base_dir) + f'/Swin/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/Swin/SRGAN_OUT/EPOCH100n_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_rev_{var_name}_no_pr/predictant_ypred.nc', \
                    #str(base_dir) + f'/EffNetV2/SRGAN_OUT/EPOCH100_rev_{var_name}_with_pr/predictant_ypred.nc', \
                   ]
        #                    str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_pr_tas_with_pr/predictant_ypred.nc', \
        #comp_experiment_list = [ 'SG_nopr', 'SG_withprn', 'ViT', 'Swin', 'Swinn', 'EffNetV2', 'EffNetV2rev', 'EffNetV2rev_withpr', 'HCLIM 12km']
        comp_experiment_list = [ 'SG_1gpu', 'SG_2gpu', 'SG_npr_50', 'SG_npr_50_drop', 'SG_npr_50_drop_lr1e-5', 'SG_npr_50_drop_lr1e-5_earlystop', 'SG_pr_50_drop0.3_lr1e-5_earlystop_discdrop_genalldrop', 'HCLIM 12km']

    elif var_name == 'tas':

        reference_highres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH100_{var_name}_no_pr'
        reference_highres_file = "predictant_ytest.nc"

        reference_lowres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH1_{var_name}_with_tas'
        reference_lowres_file = "predictor.nc"

        """
        comparison_files = [\
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_earlystop/predictant_ypred.nc", \
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_with_{var_name}_earlystop/predictant_ypred.nc", \
                   ]
        comp_experiment_list = [ 'SG', 'SG_tas', 'HCLIM 12km']
        """

        experiment_files = { \
                'HCLIM 3km': reference_highres_folder + '/' + reference_highres_file, \
                'SG': str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_no_pr/predictant_ypred.nc", \
                #'SG_tas': str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_with_{var_name}_earlystop/predictant_ypred.nc", \
                'HCLIM 12km': reference_lowres_folder + '/' + reference_lowres_file, \
                }

    elif var_name == 'mrsol':

        reference_highres_folder = str(base_dir) + f'/SG/SRGAN_OUT/EPOCH1_{var_name}_with_mrsol_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/'
        reference_highres_file = "predictant_ytest.nc"
        reference_lowres_folder = str(base_dir) + f"/SG/SRGAN_OUT/EPOCH1_{var_name}_with_mrsol_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/"
        reference_lowres_file = "predictor.nc"

        comparison_files = [\
                    str(base_dir) + f"/SG/SRGAN_OUT/EPOCH100_{var_name}_dropout0.3_lr1e-5_earlystop_discdrop_genalldrop/predictant_ypred.nc", \
                   ]
        comp_experiment_list = [ 'SG', 'HCLIM 12km']

    """
    ref_experiment_list = [ 'HCLIM 3km' ]

    """

    return experiment_files #, comp_experiment_dict

