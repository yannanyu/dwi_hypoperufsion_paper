import os
import logging
import numpy as np
import nibabel as nib
from glob import glob
from sklearn.metrics import auc, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import pdb

def dice_score(y_true, y_pred, smooth=0.0000001, threshold_true=0.1, threshold_pred=0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def specificity(y_true, y_pred, smooth = 0.00001, threshold_true = 0.1, threshold_pred =0.5):
    y_neg_f = y_true.flatten() < threshold_true
    y_pred_pos_f = y_pred.flatten() >= threshold_pred
    false_pos = np.sum(y_neg_f * y_pred_pos_f)
    return np.sum(y_neg_f) / (np.sum(y_neg_f) + false_pos + smooth)
def vol_diff(y_true, y_pred, threshold_true = 0.1, threshold_pred =0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f) - np.sum(y_true_f)
def vol_pred(y_pred,threshold_pred=0.5):
    y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f)
def weighted_dice(y_true,y_pred,smooth = 0.00001,threshold_true = 0.1, threshold_pred =0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (1.7 * np.sum(y_true_f) + 0.3 * np.sum(y_pred_f) + smooth)
def get_max_for_each_slice(data):
    '''
    data has to be 3d
    '''
    assert len(data.shape) == 3 , 'input data is not 3d'
    max_list = []
    for slice_num in range(data.shape[2]):
        max_list.append(np.max(data[:,:,slice_num]))
    return max_list

def define_laterality(data,threshold):
    midline = int(data.shape[0] / 2)
    max_list_gt = get_max_for_each_slice(data)
    lesion_left = 0
    lesion_right = 0
    lesion_side = ''
    for slice_num in range(len(max_list_gt)):
        if max_list_gt[slice_num] > threshold:
            if np.sum(data[:midline, :, slice_num]) > np.sum(data[midline:, :,
                                                                slice_num]):  ## If stroke in Left side of the image and Right side of the brain
                lesion_left += 1
            elif np.sum(data[:midline, :, slice_num]) < np.sum(data[midline:, :,
                                                                  slice_num]):  ## If stroke in Right side of the image and Left side of the brain
                lesion_right += 1
    if (lesion_left > lesion_right and (lesion_right > 3)) or (lesion_left < lesion_right and (lesion_left > 3)):
        lesion_side = 'B'
    elif lesion_left > lesion_right:
        lesion_side = 'L'
    elif lesion_right > lesion_left:
        lesion_side = 'R'
    # print(lesion_left,lesion_right)
    return lesion_side

def metrics_output(y_true, y_pred,threshold_true,threshold_pred):
    '''
    output all the metrics including auc dice recall precision f1score and volume difference
    '''
    fpr, tpr, thresholds = roc_curve(y_true>threshold_true,y_pred)
    auc_hemisphere = auc(fpr, tpr)
    precision = precision_score(y_true>threshold_true, y_pred>threshold_pred)
    recall = recall_score(y_true>threshold_true, y_pred>threshold_pred)
    dice = dice_score(y_true,y_pred, threshold_true=threshold_true,threshold_pred=threshold_pred)
    spec = specificity(y_true, y_pred,threshold_true=threshold_true,threshold_pred=threshold_pred)
    voldiff = 0.008*vol_diff(y_true, y_pred,threshold_true=threshold_true,threshold_pred=threshold_pred)
    volpred = 0.008*vol_pred(y_pred,threshold_pred)
    f1score = 2 * precision * recall / (precision + recall + 0.0001)
    return auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, f1score, fpr, tpr, thresholds

def load_nii(path):
  data = nib.load(path)
  output = data.get_fdata()
  output = np.maximum(0, np.nan_to_num(output, 0))
  return output

def volume_from_mask(mask_path,label_threshold=0.5,option='>'):
    mask = load_nii(mask_path)
    if option == '>':
        volume = np.sum(mask>label_threshold)
    if option == '<':
        volume = np.sum(mask<label_threshold)
    return volume * 0.008

def calculate_volume_from_list(subj_list,dir_stroke='/data/yannanyu/newstroke_masked/',contrast='ADC',ext='.nii'):
    list_subject = sorted(subj_list)
    result_list = {'subject':[],'DWIvolume':[]}
    for subj in list_subject:
        print(subj)
        path = os.path.join(dir_stroke,subj,contrast+ext)
        volume = volume_from_mask(path)
        result_list['subject'].append(subj)
        result_list['DWIvolume'].append(volume)
    return result_list

def stroke_test_metrics(dir_result = '/data/yuanxie/Enhao/stroke_cv',dir_stroke = '/data/yuanxie/stroke_preprocess173/',
                dir_stroke2='/data/yannanyu/newstroke_masked/',dir_brainmask = '/Users/admin/controls_stroke_DL/001/inputs_aug0.hdf5',
                subj_list = ['30077A'], mask_contrast = 0,
                threshold_true =0.5, threshold_pred = 0.5,printout=True,upper_lim=91,lower_lim=0,
                savedata=True,dwimask=True,gt_filename='LESION'):
    '''

    :param dir_result: where to find predicted h5 file
    :param dir_stroke: where to find input h5 file
    :param dir_brainmask: where to find T1 template h5 file
    :param subj_list: list of the test cases
    :param model_name: model name
    :param mask_contrast: contrast for PWI masking. 4 = MTT, 5 = Tmax, corresponding to the input.h5 4th dimension.
    :param threshold_true: when ground truth > threshold_true, count as positive
    :param threshold_pred: when prediction > threshold_pred, count as positive
    :return: the list_result containing auc, precision, recall , dice, auc all.
    '''
    '''
    setup gpu
    '''

    list_subject = sorted(subj_list)


    list_result = {'subject':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
    all_y_true = np.array([])
    all_y_pred = np.array([])
    for subject in list_subject:
        path_subj = dir_stroke + '{}'.format(subject)
        if not os.path.exists(path_subj):
            print('did not find path for', subject)
            continue
        # pdb.set_trace()
        subject_file = glob(path_subj)[0].split('/')[-1]
        # load nifti
        path_gt = dir_stroke + "{}/{}.nii".format(subject_file,gt_filename)
        if not os.path.exists(path_gt):
            path_gt = dir_stroke2 + "{}/{}.nii".format(subject_file,gt_filename)
        if not os.path.exists(path_gt):
            print('no ground truth file for', subject)
            continue

        data_gt = load_nii(path_gt)

        name_output = '{0}.nii'.format(subject)
        path_output = os.path.join(dir_result, name_output)
        data_output = load_nii(path_output)
        # input brain masking from T1 template.
        T1temp = load_nii(dir_brainmask)
        T1temp = T1temp[:, :, lower_lim:upper_lim]

        # input brain masking from PWI (because some slices not covered by PWI)
        if mask_contrast == 4:
            mask_name = 'PWIMTT.nii'
        elif mask_contrast == 0:
            mask_name = 'DWI.nii'
        elif mask_contrast == 1:
            mask_name = 'ADC.nii'
        elif mask_contrast ==2:
            mask_name = 'TMAXthresholded.nii'
        # else:
        #     print('4=MTT, 0=DWI,1=ADC. please check code and input right mask_contrast')
        if mask_contrast < 9:
            path_PWImask = dir_stroke + '{0}/'.format(subject+file) + mask_name
            data_PWImask = load_nii(path_PWImask)
            data_PWImask = data_PWImask[:, :, lower_lim:upper_lim]

        #get DWI mask
        if dwimask:
            path_DWI = dir_stroke + '{0}/'.format(subject_file) + 'DWI.nii'
            dwi = load_nii(path_DWI)
            dwi = dwi[:, :, lower_lim:upper_lim] * (T1temp > 0)
            mean_dwi = np.mean(dwi[np.nonzero(dwi)])

        # compute auc
        y_true_data = []
        y_pred_data = []
        midline = int(data_gt.shape[0]/2)
        lesion_side = define_laterality(data_gt,threshold_true)

        if lesion_side != 'B':
            if lesion_side == 'L': ## If stroke in Left side of the image and Right side of the brain
                T1temp[midline:, :, :] = 0
            elif lesion_side == 'R': ## If stroke in Right side of the image and Left side of the brain
                T1temp[:midline, :, :] = 0
            else:
                print('check code and data. Left lesion  = Right lesion ')
        for index in range(data_gt.shape[2]):
            # if max_list_gt[index] > threshold_true or max_list_output[index] > threshold_pred: ##if include whole hemisphere, do not use this line.
            brain_mask = (T1temp[:, :, index] > 0) * 1.
            if mask_contrast < 9:
                PWImask = np.maximum(0, np.nan_to_num(data_PWImask[:, :, index], 0))
                brain_mask = brain_mask * (PWImask >0)
            if dwimask:
                brain_mask = brain_mask * (dwi[:, :, index] > (0.3 * mean_dwi))
            #### calculate AUC based on hemisphere!
            brain_mask[brain_mask == 0] = np.NaN  # 0 still be calculated in later steps, so convert to NaN
            y_true_masked = data_gt[:, :, index] * brain_mask
            y_true_data.append(y_true_masked)
            y_pred_masked = data_output[:, :, index] * brain_mask
            y_pred_data.append(y_pred_masked)

        y_true = np.array(y_true_data).flatten()
        y_true = y_true[~np.isnan(y_true)]
        y_pred = np.array(y_pred_data).flatten()
        y_pred = y_pred[~np.isnan(y_pred)]
        auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, _, _, _ ,_= metrics_output(y_true, y_pred, threshold_true, threshold_pred)
        # weighted_dice_score = weighted_dice(y_true, y_pred,threshold_true=threshold_true,threshold_pred=threshold_pred)
        if printout:
            print(subject, dice, auc_hemisphere, precision, recall, spec, voldiff, volpred, abs(voldiff))
        if savedata:
            label = (y_true > threshold_true) * 1.
            pred = y_pred
            id = range(len(y_true))
            output_csv = np.column_stack((id,label,pred))
            np.savetxt(dir_result + subject + '.csv', output_csv, fmt='%1.f,%1.f,%4.5f',delimiter=",")
        list_result['subject'].append(subject)
        list_result['auc'].append(auc_hemisphere)
        list_result['precision'].append(precision)
        list_result['recall'].append(recall)
        list_result['dice'].append(dice)
        list_result['specificity'].append(spec)
        list_result['volume_difference'].append(voldiff)
        list_result['volume_predicted'].append(volpred)
        list_result['abs_volume_difference'].append(abs(voldiff))
        # list_result['f1_score'].append(f1score)

        all_y_true = np.append(all_y_true,y_true)
        all_y_pred = np.append(all_y_pred,y_pred)

    if savedata:
        label_all = (all_y_true > threshold_true) * 1.
        output_csv_all = np.column_stack((label_all, all_y_pred))
        np.savetxt(dir_result + 'all_pred.csv', output_csv_all,fmt='%1.f,%4.5f',delimiter=',')

    all_auc_hemisphere, all_precision, all_recall, all_dice, all_spec, all_voldiff, all_volpred, all_f1score,fpr,tpr,thresholds = metrics_output(all_y_true, all_y_pred, threshold_true, threshold_pred)
    list_result_all = {'auc': all_auc_hemisphere, 'precision': all_precision, 'recall': all_recall,
                     'specificity': all_spec, 'dice': all_dice, 'volume_difference': all_voldiff,
                     'volume_predicted': all_volpred,
                     'abs_volume_difference': abs(all_voldiff)}
    return list_result, list_result_all,fpr,tpr,thresholds

def create_fig_for_model(results,output_path,figname = 'metrics.png',tablename = 'metrics.txt',comparable_level = []):
    '''
    this function plot the change of metrics as the threshold of the model changes
    inputs:
    results: should be a dictionary with keys that you need to plot
    output_path is the path you want figure to save.
    '''
    x = results['thres']
    y1 = results['auc']
    y2 = results['dice']
    y3 = results['volume_predicted']
    y4 = results['volume_difference']
    y5 = results['recall']
    y6 = results['precision']
    y7 = results['abs_volume_difference']
    y8 = results['specificity']

    fig, ((ax1,ax2),(ax3,ax4),(ax7,ax5),(ax6,ax8)) = plt.subplots(4,2,figsize= (8,12),sharex=False)


    ax1.set_xlabel('threshold of model')
    ax1.set_ylabel('auc')
    ax1.plot(x,y1,zorder=1, lw =1)
    ax1.scatter(x,y1,zorder=2,s=20)
    if 'auc' in comparable_level:
        ax1.axhline(y=comparable_level['auc'][1],color='lightslategrey', linestyle = '--')
        ax1.axhspan(ymin=comparable_level['auc'][0],ymax=comparable_level['auc'][2],color='lightslategrey', alpha=0.1)
    # ax2 = ax1.twinx()
    ax2.set_xlabel('threshold of model')
    ax2.set_ylabel('Volume Predicted (ml)')
    ax2.plot(x,y3, lw=1)
    ax2.scatter(x,y3,zorder=2,s=20)
    if 'volume_predicted' in comparable_level:
        ax2.axhline(y=comparable_level['volume_predicted'][1],color='lightslategrey', linestyle = '--')
        ax2.axhspan(ymin=comparable_level['volume_predicted'][0],ymax=comparable_level['volume_predicted'][2],color='lightslategrey', alpha=0.1)

    ax3.set_xlabel('threshold of model')
    ax3.set_ylabel('dice score')
    ax3.plot(x,y2,lw =1)
    ax3.scatter(x,y2,zorder=2,s=20)
    if 'dice' in comparable_level:
        ax3.axhline(y=comparable_level['dice'][1],color='lightslategrey', linestyle = '--')
        ax3.axhspan(ymin=comparable_level['dice'][0],ymax=comparable_level['dice'][2],color='lightslategrey', alpha=0.1)
    ax4.set_xlabel('threshold of model')
    ax4.set_ylabel('volume difference (ml)')
    ax4.plot(x,y4,lw =1)
    ax4.scatter(x,y4,zorder=2,s=20)
    if 'volume_difference' in comparable_level:
        ax4.axhline(y=comparable_level['volume_difference'][1],color='lightslategrey', linestyle = '--')
        ax4.axhspan(ymin=comparable_level['volume_difference'][0],ymax=comparable_level['volume_difference'][2],color='lightslategrey', alpha=0.1)

    ax5.set_xlabel('threshold of model')
    ax5.set_ylabel('recall')
    ax5.plot(x,y5,lw =1)
    ax5.scatter(x,y5,zorder=2,s=20)
    if 'recall' in comparable_level:
        ax5.axhline(y=comparable_level['recall'][1],color='lightslategrey', linestyle = '--')
        ax5.axhspan(ymin=comparable_level['recall'][0],ymax=comparable_level['recall'][2],color='lightslategrey', alpha=0.1)
    ax6.set_xlabel('threshold of model')
    ax6.set_ylabel('precision')
    ax6.plot(x,y6,lw =1)
    ax6.scatter(x,y6,zorder=2,s=20)
    if 'precision' in comparable_level:
        ax6.axhline(y=comparable_level['precision'][1],color='lightslategrey', linestyle = '--')
        ax6.axhspan(ymin=comparable_level['precision'][0],ymax=comparable_level['precision'][2],color='lightslategrey', alpha=0.1)
    ax7.set_xlabel('threshold of model')
    ax7.set_ylabel('absolute volume difference (ml)')
    ax7.plot(x,y7,lw =1)
    ax7.scatter(x,y7,zorder=2,s=20)
    if 'abs_volume_difference' in comparable_level:
        ax7.axhline(y=comparable_level['abs_volume_difference'][1],color='lightslategrey', linestyle = '--')
        ax7.axhspan(ymin=comparable_level['abs_volume_difference'][0],ymax=comparable_level['abs_volume_difference'][2],color='lightslategrey', alpha=0.1)
    ax8.set_xlabel('threshold of model')
    ax8.set_ylabel('specificity')
    ax8.plot(x,y8,lw =1)
    ax8.scatter(x,y8,zorder=2,s=20)
    if 'specificity' in comparable_level:
        ax8.axhline(y=comparable_level['specificity'][1],color='lightslategrey', linestyle = '--')
        ax8.axhspan(ymin=comparable_level['specificity'][0],ymax=comparable_level['specificity'][2],color='lightslategrey', alpha=0.1)
    fig.tight_layout()
    fig.savefig(output_path + figname)
    with open(output_path+tablename, 'w') as file:
      keylist = ','.join(str(key) for key in results.keys()) + '\n'
      file.write(keylist)
      for i in range(len(results['thres'])):
        row = []
        for key in results.keys():
          row += [results[key][i]]
          str1 = ','.join(str(n) for n in row) + '\n'
        file.write(str1)
    return print('figure and table saved at:', output_path + figname)


def create_roc(fpr,tpr,roc_auc,output_path, thresholds, figname = 'roc.png',tablename = 'table.csv',datawrite = True):
    youden = np.array(tpr) - np.array(fpr)
    max_youden = np.max(youden)
    cutoff_index = np.argmax(youden)
    # cutoff_index = youden.tolist().index(max_youden)

    cutoff = thresholds[cutoff_index]
    print(cutoff_index,cutoff, max_youden)
    plt.plot(fpr,tpr,color = 'darkorange', label = 'ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1],[0,1],color = 'navy', linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cutoff = {0:.2f}, Youden Index = {1:.2f}, sensitivity {2:.2f}, specificity {3:.2f}'.format(cutoff, max_youden,tpr[cutoff_index],1-fpr[cutoff_index]) )
    plt.legend(loc="lower right")
    plt.savefig(output_path + figname)
    plt.close()
    if datawrite:
        output_csv = np.column_stack((thresholds,fpr,tpr))
        np.savetxt(output_path + tablename, output_csv, fmt='%4.3f,%4.3f,%4.3f', delimiter=",")
    print('data output =', datawrite, ', figure saved at:', output_path + figname)
    return cutoff

def save_dict(dict,output_path,filename,summary=True):
    '''
    save dictionary into csv file
    :param dict: dictionary you want to save
    :param output_path:
    :param filename: e.g. file.csv
    :return:
    '''
    header =[0,]
    median = [0,'median']
    percentile25 = [0,'25th percentile']
    percentile75 = [0,'75th percentile']
    for key in dict:
        length = len(dict[key])
    output_csv = np.empty((length,))
    for key in dict:
        output_csv = np.column_stack((output_csv,np.array(dict[key])))
        header += [key]
        if summary:
            if not key == 'subject':
                median += [np.median(dict[key])]
                percentile25 += [np.percentile(dict[key],25)]
                percentile75 += [np.percentile(dict[key],75)]
    if summary:
        output_csv = np.row_stack((np.array(header),output_csv,median,percentile25,percentile75))
    else:
        output_csv = np.row_stack((np.array(header), output_csv))
    np.savetxt(output_path + filename, output_csv[:,1:], fmt='%s', delimiter=",")
    return print('dictionary saved at ', output_path + filename, 'with summary', summary)
