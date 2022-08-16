
''' basic dependencies '''
import numpy as np
import os
import nibabel
import h5py
import random
from scipy import ndimage
import pdb


from tool_package import load_nii,export_to_h5,pad_image, augment_data,read_csv,image_normalization,label_cleanup
import itertools
'''
contrast
'''
# list_contrast_keyword = ['FUDWI','FLAIR'] #regular perfusion data
# list_contrast_keyword = ['DWI','ADC','PWITMAX','TMAXthresholded','ADCthresholded','PWICBV','PWICBF','PWIMTT'] #regular perfusion data
# list_contrast_keyword = ['PWITMAX','TMAXthresholded','PWICBF','rCBFthresholded','PWICBV','PWIMTT'] ## PWI to ADC seg, transfer for CT with major reper
# list_contrast_keyword = ['CTPTMAX','TMAXthresholded','CTPCBF','rCBFthresholded','CTPCBV','CTPMTT']
list_contrast_keyword = ['DWI','ADC','ADCthresholded'] # DWI only
# keyword_output = 'LESION'
keyword_output='Tmax+ADC_seg_cleanup'
ext_data = 'nii'
target_dim = (128,128,60) # padding to this dimension
mode='concat'
clinical_var=True
'''
get files
'''
tab_data, header = read_csv('tabular_data.csv')
# get all file list
dir_stroke = '/data/yannanyu/newstroke_masked/'#CTPbaseline_masked; PWImasked185 ;D3_masked; newstroke_masked
dir_preprocessing = '/data/yannanyu/preprocess_DWItoTMAXclean_noreshape_clinical_normalized/' #preprocess_8inputs_noreshape_clinical_normalized
if not os.path.exists(dir_preprocessing):
    os.mkdir(dir_preprocessing)
# dim_reshape = [128,128]
options_augmentation = {'flipx':1,'flipc':1}
key_augments = sorted(options_augmentation.keys())


axis_augmentation = [1,2]
num_slice_augmentation = 2

# create output folder if not available
if not os.path.exists(dir_preprocessing):
    os.mkdir(dir_preprocessing)
'''
adjust list_subfolder
'''
###use all cases
list_subfolder = sorted([os.path.join(dir_stroke,x) for x in os.listdir(dir_stroke) if
                  os.path.isdir(os.path.join(dir_stroke,x))])

# loop for subjects
num_subject = len(list_subfolder)
for subfolder in list_subfolder:
    subject_name = subfolder.split('/')[-1]
    if len(subject_name)>5:
        subject_name= subject_name[:5]
    if subject_name[:3]=='000':
        print('skipping',subject_name)
        continue
    print('now processing ', subject_name)
    # output folder
    dir_output = os.path.join(dir_preprocessing, subject_name)
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)

    # check subfolder
    list_subfiles = os.listdir(subfolder)
    list_files_with_ext = sorted([x for x in list_subfiles if (x.endswith(ext_data) and (not x.startswith('.')))])

    # get each contrast
    list_filename_sample_contrasts = []
    for index_contrast in range(len(list_contrast_keyword)):
        keyword_contrast = list_contrast_keyword[index_contrast]
        list_filename_sample_contrasts.append(
            sorted([x for x in list_files_with_ext if x.find(keyword_contrast)>=0])
            )

    # get output
    list_filename_sample3 = sorted([x for x in list_files_with_ext if x.find(keyword_output)>=0])

    # load data
    if mode=='overlap':
        for contrast in list_contrast_keyword:
            if not os.path.exists(os.path.join(subfolder, contrast + '.nii')):
                continue
            print('use contrast: {} as input image'.format(contrast))
            data_load, _ = load_nii(os.path.join(subfolder, contrast + '.nii'))
            data_load = pad_image(data_load,target_dim)
    elif mode == 'concat':
        data_load = np.empty((len(list_contrast_keyword),)+target_dim)
        for contrast_index,contrast in enumerate(list_contrast_keyword):
            if not os.path.exists(os.path.join(subfolder, contrast + '.nii')):
                print('no contrast', contrast,'for subject', subject_name,', loading zeros to the image')
                zeros = np.zeros(target_dim)
                data_load[contrast_index] = data_contrast
                continue
            data_contrast, _ = load_nii(os.path.join(subfolder, contrast + '.nii'))
            data_contrast = image_normalization(data_contrast,contrast) # normalize data by non zero mean or min.
            # pdb.set_trace()
            data_contrast = pad_image(data_contrast,target_dim)
            # load
            data_load[contrast_index] =data_contrast
    elif mode == 'separate':
        data_dict ={key:[] for key in list_contrast_keyword}
        for contrast in list_contrast_keyword:
            if not os.path.exists(os.path.join(subfolder, contrast + '.nii')):
                print('no contrast', contrast,'for subject', subject_name,', loading zeros to the image')
                zeros = np.zeros(target_dim)
                data_dict[contrast] = data_contrast
                continue
            print('use contrast: {} as input image'.format(contrast))
            data_contrast, _ = load_nii(os.path.join(subfolder, contrast + '.nii'))
            data_contrast = pad_image(data_contrast,target_dim)
            data_dict[contrast] = data_contrast
    # output segmentation
    if list_filename_sample3 == []:
        print(subfolder,'no output file, skipping')
        continue
    filename_full=list_filename_sample3[0]
    data_full,_ = load_nii(os.path.join(subfolder, filename_full))
    data_full = label_cleanup(data_full) # clean up non binary segmentation maps
    data_full = pad_image(data_full,target_dim)

    if clinical_var:
    # load tab data
        pt_list = tab_data[:,0]
        OIT = tab_data[:,np.where(header == 'onset_to_imaging')]
        pt_index = np.where(pt_list==subject_name)
        treatment = np.squeeze(tab_data[pt_index,np.where(header == 'treatment')])
        mrs = np.squeeze(tab_data[pt_index,np.where(header == 'mrs')])
        tpa = np.squeeze(tab_data[pt_index,np.where(header == 'tpa')])
        onset_to_imaging = np.squeeze(tab_data[pt_index,np.where(header == 'onset_to_imaging')])
        tici =np.squeeze(tab_data[pt_index,np.where(header == 'tci_num')])
        image_to_dsa = np.squeeze(tab_data[pt_index,np.where(header == 'image_to_dsa')])
        # image_to_eariest_treat = np.squeeze(tab_data[pt_index,np.where(header == 'image_to_eariest_treat')])
        nihss = np.squeeze(tab_data[pt_index,np.where(header == 'baselinenihssscore')])
        side = np.squeeze(tab_data[pt_index,np.where(header == 'side')])
        occlusion = np.squeeze(tab_data[pt_index,np.where(header == 'occlusion')])



        # treatment = int(treatment)
        tici = int(tici)
        occlusion = int(occlusion)
        side = int(side)
        # pdb.set_trace()
        '''
        replace empty field
        '''
        if str(np.squeeze(mrs)) == 'nan': # because did not record
            mrs = int(3) ## replace the missing value
        else:
            mrs = int(mrs)
        if str(np.squeeze(tpa)) == 'nan':
            tpa = int(-1)
        else:
            tpa = int(tpa)

        if str(np.squeeze(nihss)) == 'nan':
            nihss =float(-1)
        else:
            nihss = float(nihss)/21
        # if str(np.squeeze(onset_to_imaging)) == 'nan': # because did not record
        #     onset_to_imaging = float(-1)
        # else:
        #     onset_to_imaging = float(onset_to_imaging)
        # if str(np.squeeze(image_to_dsa)) == 'nan': # because did not treat
        #     image_to_dsa = float(-1)
        # else:
        #     image_to_dsa = float(image_to_dsa)
        # if str(np.squeeze(image_to_eariest_treat)) == 'nan':# because did not treat
        #     image_to_eariest_treat = float(-1)
        # else:
        #     image_to_eariest_treat = float(image_to_eariest_treat)

    '''
    export to h5 format
    '''
    aug_index=0
    for x in key_augments:
        augment = {x:options_augmentation[x]}
        # augment
        data_input_augment = augment_data(data_load, axis_xy=axis_augmentation, augment=augment)
        data_full_augment = augment_data(data_full, axis_xy=axis_augmentation, augment=augment)
        side_aug = side
        if aug_index==1: # if mirror augmentation, change the side variable
            if side ==1 or side ==0: # side =2 means bilateral, unchanged even if mirrored.
                side_aug = 1 - side
        # pdb.set_trace()
        #filename
        filename_inputs_export_aug = '{0}_aug{1}.hdf5'.format('inputs', aug_index)
        filename_output_export_aug = 'output_aug{0}.hdf5'.format(aug_index)
        aug_index += 1
        #export
        print(filename_inputs_export_aug,data_input_augment.shape, filename_output_export_aug,data_full_augment.shape)
        print(os.path.join(dir_output, filename_inputs_export_aug))

        img_samples = h5py.File(os.path.join(dir_output, filename_inputs_export_aug),'w')
        img_samples.create_dataset('init', data=data_input_augment)
        if clinical_var:
            img_samples.create_dataset('tpa', data=tpa)
            # img_samples.create_dataset('treatment', data=treatment)
            # img_samples.create_dataset('onset_to_imaging', data=onset_to_imaging)
            # img_samples.create_dataset('image_to_dsa', data=image_to_dsa)
            # img_samples.create_dataset('image_to_eariest_treat', data=image_to_eariest_treat)
            img_samples.create_dataset('side', data=side_aug)
            img_samples.create_dataset('occlusion', data=occlusion)
            img_samples.create_dataset('nihss', data=nihss)
            img_samples.create_dataset('tici', data=tici)
        img_samples.close()

        img_samples = h5py.File(os.path.join(dir_output, filename_output_export_aug),'w')
        img_samples.create_dataset('init', data=data_full_augment)
        if clinical_var:
            img_samples.create_dataset('mrs', data=mrs)
        img_samples.close()
