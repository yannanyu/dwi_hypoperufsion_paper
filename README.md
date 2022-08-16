# dwi_hypoperufsion_paper
code deposit for UNet model predicting hypoperfusion from DWI.

## data preparation:  
For CAFN member, check for data structure on Longo: /data/yannanyu/newstroke_masked/  
```
data_folder ----00001(patient ID) ---DWI.nii (baseline)
                                  ---LESION.nii (binary segmentation mask)  
                                  ---PWICBV.nii
                                  ---PWICBF.nii
                                  ---PWIMTT.nii
                                  ---PWITMAX.nii
                                  ---ADC.nii

            ----00002             ---DWI.nii
                                  ...
```

The above contrast names can be modified, e.g. change PWICBV to CTPCBV, you will only need to change the keywords in nii_to_h5.py.  

## Convert nifty file to h5 file: modify and run nii_to_h5.py  
1. what to process: modify line 17-27  
   - list_contrast_keyword = the image contrast used in the input , match the nii file names in the data folder, note that sequence matters.  
   - Keyword_output = the output image (segmentation), match the nii file name in the data. ext_data = default is nii.  
   - mode: so far I'm only using 'concat' right now. The rest two modes 'overlap' (basically only saves the last contrast as input) and 'separate' (saves in different array instead of stacking them), you need to add image_normalization function in those modes.  
   - target_dim = a dimension that images are padding into. e.g. in our dataset, images are 91 x 109 x 60, we padded them into 128x128x60 with 0s.  
   - Clinical_var = whether to include clinical variables in the input h5 files.  
2. prepare tabular_data.csv contains patientID in the first column, clinical variables in the rest columns.  
   - modify line 130-142 (as well as line 146-164 if specifici manipulation of data is required) to adjust to the clinical variable you want to include.  
   - if you don't plan to include any clinical information, simply modify clinical_var=False at line 27.      
3. where to look and save: modify line 33-43  
   - dir_stroke = where you read the case nifty files  
   - dir_preprocessing = where you store the output h5 files.  
   - options_augmentation and axis_augmentation: please refer to function augment_data in tool package. 'flipx' means flip left and right, 'flipc' means no augmentation. I'm usually just use the default setting. Later we can choose whether to include the augmentation in training/val/testing.  


## Train, test, calculate metrics: modify and run Unet_traintest.py  
1. mode and which GPU to use: modify line 23 31:
   - mode = 'train test metrics', as long as the string "train",'test','metrics' in mode, it will execute those steps. otherwise, modify mode='train' if you only want to train.
2. where/what to look and save: modify line 47-54  
   - model_name = the name you want to give to a new model  
   - model_name_tag = additional name tag when you want to create a different saving folder - see output_path. default is empty.  
   - read_path = path to read patient list.  
   - source_path = where contains the nifity file. referred as data_folder above.   
   - dir_brainmask = the brain mask is used when calculating the metrics, we remove pixels outside the brain mask.  
   - exclude_list = in case you want to exclude certain cases in validation or testing, list their ID here (which is also the folder name of that patient).  
   - gt_filename = the ground truth to compare to when calculating metrics. if the ground truth is final stroke lesion, then put "LESION".  
3. common adjustable parameters: modify line 59-99  
   - slice_expand = for 2.5 D model, we fed in 5 consecutive slices. Therefore, 2 represent expand 2 slice above and 2 slice below the target slice.   
   - num_channel = how many contrasts in your input. the number should be consistent with the length of list_contrast_keyword in nii_to_h5.py  
   - sample_dim = the size of input sample. if the input is not 128x128, modify the numbers.
   - numeric_dict = decide what and how many clinical variables to fed into the network. the dict key needs to fit the key of h5 files, e.g. if there is a key called 'nihss' in h5, then use 'nihss' as key in the dict. the number after the key indicates the sequence of numeric input, which needs to be continuous integers (e.g. 0,1,2,3).
     - e.g. numeric_dict= {'nihss':0,'side':1} represent we fed in as the sequence of nihss then side.
   - output_dim = the dimension of final output layer of the Unet. Usually the same x and y dimension with the input, but only 1 in the z axis.
   - predict_dim = this is the size of final output nifty. usually the same size as the nifty file in data_folder.
   - batch_size, epochs, init_lr (initial learning rate) = hyperparameters of the model  
   - cnn3d = whether to use 3d model.  
   - metrics, print_metrics, losses = determine what metrics to print during training, testing, and what losses to use.  
   - model_parameters = 'num_pooling' determine the depth of the Unet, 'num_conv_per_pooling' determine how many convolution blocks before each pooling layer. 'num_channel_first' determines the CNN filter size for the first layer. 'dropout_rate' determine the drop out rate after convolution blocks. 'pool_size' determine the size of filter in pooling layer.'activation_output' determine the activation function for Unet output.'kernel_initializer' determine the initialization method of the model.  
4. adjust patient list for training: modify line 106-109
   - file_list = a condition such as if x[:3]!='000' can be added to further select or exclude some cases.
   - file_list_D1 = additional cases excluded from file_list you want to add in training but not testing or validation.
