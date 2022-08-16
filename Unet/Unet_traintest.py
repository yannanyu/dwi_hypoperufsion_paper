import numpy as np
import pdb
from keras.models import Sequential,load_model, Model
from keras import backend as K
import keras.metrics
import keras.losses
import keras.optimizers
from glob import glob
import os
import h5py
import matplotlib.pyplot as plt
from random import shuffle, seed, choice
from skimage.io import imsave
import tensorflow as tf
# from tensorflow.random import set_seed
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session
from tool_package import *
import generator_unet as gen
from test_metrics import stroke_test_metrics, create_fig_for_model, save_dict, create_roc
from model_utils import *
from Unet_model import *
mode = 'train test metrics ' # train test metrics
save_test= True
save_all_test = False
K.clear_session()
'''
setup gpu
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
try:
    config = tf.ConfigProto() # lower version tensorflow
except:
    config = tf.compat.v1.ConfigProto() # tensorflow 2.0
config.gpu_options.allow_growth = True
try:
    set_session(tf.Session(config=config))
except:
    set_session(tf.compat.v1.Session(config=config))
# set_seed(42)# tensorflow seed fixing

'''
load h5 data
'''

model_name = '202104015_DWItoTMAXclean_dropoutU_corrected'
model_name_tag = ''
read_path = '/data/yannanyu/preprocess_DWItoTMAXclean_noreshape_clinical_normalized/'
source_path = '/data/yannanyu/newstroke_masked/' #PWImasked185
output_path = 'models/'+ model_name + model_name_tag +'/'
dir_brainmask = '/data/yannanyu/brain_mask/T1_cerebrum.nii'
exclude_list=['03046','09003','30069','032','063','069','096','116','236','356'] # poor quality ground truth, not used in val or test
gt_filename='Tmax+ADC_seg_cleanup' # LESION,Tmax+ADC_seg_cleanup
if not os.path.exists(output_path):
    os.makedirs(output_path)

# parameters
slice_expand = 2
num_channel = 3
sample_dim = (128,128,(2*slice_expand + 1)*num_channel) # image will be padded to the dimension
# numeric_dict= {'tici':0,'onset_to_imaging':1,'image_to_dsa':2,'image_to_eariest_treat':3,'tpa':4,'treatment':5} #8inputs dataset
# numeric_dict= {'nihss':0}# DWItoTMAX dataset {'tici':0,'tpa':1,'nihss':2}
# numeric_dict= {'nihss':0,'tici':1} # DWI to LESION
numeric_dict= {'nihss':0,'side':1}
numeric_dim = len(numeric_dict)
output_dim = (128,128,1)
predict_dim = (91,109,60)
batch_size = 16
epochs = 50
init_lr = 0.0005
cnn3d = False

if cnn3d:
    sample_dim = (num_channel,128,128,(2*slice_expand + 1)) # 3d if use 2 D CNN, 4D if use 3D CNN


## setup training parameter
metrics = [dice_coef,recall,precision,vol_diff]#cate_acc_plus_minus_one 'categorical_accuracy'
print_metrics = [dice_coef,recall,precision,vol_diff]
losses = weighted_ce_l1_bycase

model_parameters = {
    "num_channel_input": sample_dim[2],
    'img_rows':sample_dim[0],
    'img_cols':sample_dim[1],
    'numeric_dim':numeric_dim,
    'lr_init':init_lr,
    'loss_function':losses,
    'metrics_monitor':metrics,
    'num_poolings':3,
    'num_conv_per_pooling':2,
    'num_channel_first':32,
    'dropout_rate':0.25,
    'pool_size':(2,2),
    'activation_output':'sigmoid',
    'kernel_initializer':'he_normal',

}

# select to print model performance after training or testing.
print_dict ={'Test loss': 'loss','Test dice':'dice_coef'}

# get patient for each fold
data_path = '/data/yannanyu/preprocess_DWItoTMAXclean_noreshape_clinical_normalized/'
file_list_all = glob_filename_list(data_path)
file_list = [x for x in file_list_all if x[:3]!='000']
# file_list += ['00063','00064'] # depends on if there is D1 case in the folder.
# file_list_D1 = [x for x in file_list_all if x[:3]=='000' and not x in ['00063','00064']]
file_list_D1 = []
# pdb.set_trace()
seed(1)
shuffle(file_list)
subj_list = []
folds=5
case_per_fold = round(len(file_list)/folds)
test_result_all = [] ## save test results of all folds.


for n in range(1,folds+1):
    if n*case_per_fold >= len(file_list):
        end = len(file_list)
    elif n == folds:
        end = len(file_list)
    else:
        end = n * case_per_fold
    list_n = file_list[(n-1)*case_per_fold:end]
    subj_list.append(list_n)
# pdb.set_trace()
if 'train' in mode:
    array_subj = np.array(subj_list)
    np.savetxt(output_path + 'subject_per_fold.csv', array_subj,fmt='%s',delimiter=',')
for i in range(1,6): ## 1~6
    test_subj = subj_list[i-1].copy()
    test_subj = remove_from_list(test_subj,exclude_list) ## remove cases that is unwanted in test
    val_subj = subj_list[i%folds].copy()
    val_subj = remove_from_list(val_subj,exclude_list) ## remove cases that is unwanted in val
    print('test subj length',len(test_subj),len(subj_list[i-1]))
    print('val subj length',len(val_subj),len(subj_list[i%folds]))
    train_subj = [subj_list[(i+x)%folds] for x in range(1,folds-1)]

    flat_train_subj = []
    for sublist in train_subj:
        for item in sublist:
            flat_train_subj.append(item)
    flat_train_subj += file_list_D1


    if 'train' in mode or 'test' in mode:
        test_path = gen.get_h5_path(data_path,test_subj,include_mirror=False)
        val_path = gen.get_h5_path(data_path,val_subj,include_mirror=False)
        train_path = gen.get_h5_path(data_path,flat_train_subj,include_mirror=True)

        print('fold', i)
        print('test',test_subj,'with {} samples'.format(len(test_path)))
        print('val', val_subj,'with {} samples'.format(len(val_path)))
        print('train',flat_train_subj,'with {} samples'.format(len(train_path)))
        # pdb.set_trace()


        filepath_checkpoint = 'models/'+ model_name + model_name_tag + '/fold{}_ckpt.ckpt'.format(i)
        filepath_model = 'models/'+ model_name + model_name_tag + '/fold{}_model.json'.format(i)
        if not os.path.exists('models/'+ model_name+ model_name_tag + '/'):
            os.makedirs('models/'+ model_name+ model_name_tag + '/')

        model_checkpoint = ModelCheckpoint(filepath_checkpoint,
                                           monitor='val_loss',
                                           save_best_only=True,verbose=1)  ## change monitor according to what you want (val_mrs_mrs_accuracy'val_losss')


    '''
    CNN
    '''

    if 'train' in mode:
        # pdb.set_trace()

        model = hybrid_unet(**model_parameters) #dropout_unet

        model.summary()

        train_gen = gen.hybrid_input_generator(train_path, batch_size,sample_dim=sample_dim,
                                            output_dim=output_dim,numeric_dim=numeric_dim,
                                            numeric_dict=numeric_dict,
                                            add_dimension=cnn3d,slice_expand=slice_expand,
                                            output=1)
        print('finish loading training samples')
        val_gen = gen.hybrid_input_generator(val_path,batch_size,sample_dim=sample_dim,
                                            output_dim=output_dim,numeric_dim=numeric_dim,
                                            numeric_dict=numeric_dict,
                                            add_dimension=cnn3d,slice_expand=slice_expand,
                                            output=1)
        print('finish loading validation samples')
        # test_gen = input_generator(test_path, batch_size,sample_dim=sample_dim,output_dim=output_dim, add_dimension=cnn3d,slice_expand=slice_expand,output=1)
        print('finish loading test sample')


        history = model.fit_generator(
                train_gen,
                steps_per_epoch = len(train_path)*predict_dim[-1]/batch_size,
                epochs=epochs,
                verbose=1,
                use_multiprocessing=True,
                workers=16,
                max_queue_size = 32,
                callbacks=[model_checkpoint],
                validation_data=val_gen,
                validation_steps = len(val_path)*predict_dim[-1]/batch_size
                )
        '''
        IMPORTANT:
        use multi_processing=true and workers=1 in fit_generator
        other wise callback data will be different from evaluated data. Yannan 2020/4/
        but the 2nd epoch have different value from eval data...why?
        '''
        #save best model
        model_json = model.to_json()
        with open(filepath_model, "w") as json_file:
            json_file.write(model_json)
        model.save(filepath_model[:-5]+'.h5')

        print('last model validation results')
        score_lastmodel = model.evaluate_generator(val_gen,steps=len(val_path),use_multiprocessing=False,workers=1, verbose=0)
        for key in print_dict:
            print(key,score_lastmodel[model.metrics_names.index(print_dict[key])])

        del model

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model fold{} loss'.format(i))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('models/{}/fold{}_loss.png'.format(model_name+model_name_tag,i))
        plt.clf() # clear current figure
        plt.plot(history.history['dice_coef'])
        plt.plot(history.history['val_dice_coef'])
        plt.title('Model fold{} acc'.format(i))
        plt.ylabel('dice')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('models/{}/fold{}_dice.png'.format(model_name+model_name_tag,i))
        plt.close()


    if 'test' in mode:
        '''
        Need window sliding while testing.
        '''
        model = load_model(filepath_model[:-5]+'.h5',custom_objects={'recall': recall,
            'precision':precision,'dice_coef':dice_coef,'weighted_bce_loss':weighted_bce_loss,
            'weighted_ce_l1_bycase':weighted_ce_l1_bycase,'vol_diff':vol_diff}) ## load the final model (h5 format)


        ## last model performance

        # print('last model performance:')
        # evaluate_model_generator(model,val_path, batch_size,print_dict, sample_dim, output_dim, slice_expand,cnn3d=cnn3d,steps=len(val_path)*10)
        model.load_weights(filepath_checkpoint) # Comment this if you want last model instead of best model
        #print val results in best model.
        print('best model performance:')
        # evaluate_model_generator(model,val_path, batch_size,print_dict, sample_dim, output_dim, slice_expand,cnn3d=cnn3d,steps=len(val_path)*10)
        for input_path in test_path:
            test_hybrid_model(model,input_path, sample_dim,output_dim,numeric_dict,
                model_name,model_name_tag,pred_dim=predict_dim,slice_expand=slice_expand,
                print_dict=print_dict, read_dim=(sample_dim[0],sample_dim[1],predict_dim[-1],num_channel),model_output_layer_name='output_layer',
                name_tag='',save=save_test,add_dimension=cnn3d,source_path=source_path)

dwimask=True
mask_contrast=9
if 'metrics' in mode:
    threshold_true = 0.5 # thresholding has been done in preprocess. if preprocess is already 0.9, try low value here.
    rangelist = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # mean_metrics = {'thres':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'f1_score':[],'abs_volume_difference':[]}
    median_metrics = {'thres':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
    all_px_metrics = {'thres':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
    for thres in rangelist:
        print('true:',threshold_true,'prediction',thres)
        summary_list = {'subject':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
        for cv in range(0,folds):
            if thres == 0.5: # output csv file for every patient when thres == 0.5
                datawrite=False
            else:
                datawrite = False
            subj_list_test = subj_list[cv]
            subj_list_test = remove_from_list(subj_list_test,exclude_list)
            # pdb.set_trace()
            result_list, result_list_all,fpr,tpr, thresholds = stroke_test_metrics(printout=False, dir_result = output_path, dir_stroke = source_path,
                                                                                    dir_brainmask = dir_brainmask,subj_list = subj_list_test,
                                                                                    mask_contrast = mask_contrast, threshold_true =threshold_true, threshold_pred = thres,
                                                                                   lower_lim = 0, upper_lim = 60,savedata=datawrite,dwimask=dwimask,gt_filename=gt_filename,
                                                                                   dir_stroke2='/data/yannanyu/PWImasked185/')
            # create roc figure
            if thres == 0.5:
                create_roc(fpr, tpr, result_list_all['auc'], output_path, thresholds,figname='fold{0}_roc.png'.format(cv+1),tablename='fold{0}_roc.h5'.format(cv+1), datawrite=True)
            for key in summary_list:
                summary_list[key] += result_list[key]
        printout_list = ['median', np.median(summary_list['dice']), np.median(summary_list['auc']), np.median(summary_list['precision']), np.median(summary_list['recall']), np.median(summary_list['specificity']), np.median(summary_list['volume_difference']),np.median(summary_list['volume_predicted'])]
        for key_mean_metrics in median_metrics:
            if key_mean_metrics != 'thres':
                median_metrics[key_mean_metrics] += [[np.percentile(summary_list[key_mean_metrics],25),np.median(summary_list[key_mean_metrics]),np.percentile(summary_list[key_mean_metrics],75)]]
                all_px_metrics[key_mean_metrics] += [result_list_all[key_mean_metrics]]
            else:
                median_metrics[key_mean_metrics] += [[thres,thres,thres]]
                all_px_metrics[key_mean_metrics] += [thres]
        # print(summary_list)
        print(printout_list)
        save_dict(summary_list,output_path,filename='thres'+str(thres) + '.csv',summary=True)
    create_fig_for_model(median_metrics,output_path,figname = 'metrics_median.png', tablename = 'metrics_median.txt')
    create_fig_for_model(all_px_metrics, output_path, figname = 'metrics_allpx.png', tablename = 'metrics_allpx.txt')

K.clear_session()
