import pandas
import numpy as np
import os
import h5py
import nibabel as nib
import pdb
from glob import glob
from scipy.ndimage import rotate, affine_transform


def load_nii(path):
	data = nib.load(path)
	output = data.get_fdata()
	output = np.maximum(0, np.nan_to_num(output, 0))
	return output,data.affine

def read_csv(name_data,sep =','):
    load = pandas.read_csv(name_data, sep=',',dtype=str)
    data = load.values
    header = load.columns.values
    return data, header

def glob_filename_list(path,searchterm='*',ext=''):
  filename_list = []
  for filename in glob(path+searchterm+ext):
    fn = filename.split('/')
    filename_list.append(fn[-1])
  return filename_list

def remove_from_list(ref_list, remove_list):
	for i in remove_list:
		if i in ref_list:
			# pdb.set_trace()
			ref_list.remove(i)
	return ref_list

def load_pad_index(data_path,image_name,dimension_sample):
    padding_image = np.zeros(dimension_sample)
    image,_ = load_nii(data_path + image_name)
    xpadding = (dimension_sample[0]-image.shape[0])//2
    ypadding = (dimension_sample[1]-image.shape[1])//2
    padding_image[xpadding:xpadding+image.shape[0],ypadding:ypadding+image.shape[1],:] = image
    template_index = image_name[-5]
    return padding_image, template_index

def pad_image(image, target_dim):
	padding_image = np.zeros(target_dim)
	xpadding = (target_dim[0]-image.shape[0])//2
	ypadding = (target_dim[1]-image.shape[1])//2
	padding_image[xpadding:xpadding+image.shape[0],ypadding:ypadding+image.shape[1],...] = image
	return padding_image

def cut_image(image, target_dim):
	padding_image = np.zeros(target_dim)
	xpadding = (image.shape[0]-target_dim[0])//2
	ypadding = (image.shape[1]-target_dim[1])//2
	if target_dim[2] > image.shape[2]:
		zpadding = (target_dim[2]-image.shape[2])//2
		padding_image[:,:,zpadding:zpadding+image.shape[2]] = image[xpadding:xpadding+target_dim[0],ypadding:ypadding+target_dim[1],...]
	else:
		padding_image = image[xpadding:xpadding+target_dim[0],ypadding:ypadding+target_dim[1],...]
	return padding_image

def image_normalization(image,contrast):
	if 'threshold' in contrast or '_seg' in contrast or 'CORE' in contrast or 'PENUMBRA' in contrast: # because sometimes tmax has small lesion and the diff = 0
		if not np.max(image) ==0:
			value_mean = np.min(image[np.nonzero(image)])
			print(contrast, 'non zero min', value_mean)
			image /= value_mean
	else:
		if contrast == 'ADC':
			mask_rmventricle = image < 2000
			image *= mask_rmventricle
		value_mean = np.mean(image[np.nonzero(image)])
		image /= value_mean
	return image

def label_cleanup(image,threshold=0.5):
	output = (image > threshold) + 0.
	return output

def savenib(image, affine, output_path,export_name):
	new_image = nib.Nifti1Image(image, affine)
	nib.save(new_image, os.path.join(output_path, export_name))
	return print('saved nifty shape: ',image.shape)


def augmentation(img,option={'xzoom':0.05,'yzoom':0.05,'xshear':0.05,'yshear':0.05,'xzshear':0.05,'yzshear':0.05,'rotation':10}):
	new_affine = np.eye(3)
	new_affine[0,0] = 1 + option['xzoom']*np.random.randint(-10,10)*0.1 ### x axis zoom (0.95~1.05)
	new_affine[0,1] = option['xshear']*np.random.randint(-10,10)*0.1 ## x axis shear (range -0.05~0.05)
	new_affine[0,2] = option['xzshear']*np.random.randint(-10,10)*0.1 ## x and z axis shear? (--0.05~0.05)
	new_affine[1,0] = option['yshear']*np.random.randint(-10,10)*0.1 ## y axis shear (--0.05~0.05)
	new_affine[1,1] =1 + option['yzoom']*np.random.randint(-10,10)*0.1 ## y axis zoom (0.95~1.05)
	new_affine[1,2] = option['yzshear']*np.random.randint(-10,10)*0.1 ## y and z axis shear?(--0.05~0.05)
	rnd_angle = np.random.randint(-option['rotation'], option['rotation'])
	print('augmentation parameter: x zoom:{0}, x shear:{1}, xz shear: {2}, y zoom:{3}, y shear:{4}, yz shear: {5}, rotation:{6}'.format(new_affine[0,0],new_affine[0,1],new_affine[0,2],new_affine[1,0],new_affine[1,1],new_affine[1,2],rnd_angle))
	# pdb.set_trace()
	tf_img = affine_transform(img, new_affine, mode='nearest')
	tf_img = rotate(tf_img, rnd_angle, reshape=False, order =3, mode='nearest')
	tf_img = tf_img>0.5
	return tf_img

def category_to_ordinal(y_train,category=None):

    new_y_train = np.zeros((1,category-1))
    new_y_train[0,:int(y_train)] = 1
    # pdb.set_trace()
    return new_y_train

def array_category_to_ordinal(y_train,category=None):
	categories = np.unique(y_train)

	if category:
		num_category = category
		print('output have', categories, 'orders, now setting to', category, 'orders')
	else:
		num_category = len(categories)
		print('output have', categories, 'orders')

	new_y_train = np.zeros((len(y_train),num_category-1))
	# pdb.set_trace()
	for n in range(0,len(y_train)):
		new_y_train[n,:int(y_train[n])] = 1

	return new_y_train


def get_partition_labels(path_list, key_name, init_dict = {},init_labels= {}):
    init_dict[key_name] = []
    label_count = 0
    for path in path_list:
        init_dict[key_name] += [path]
        label = get_label_from_h5path(path)
        init_labels[path] = label
        label_count +=label
    return init_dict, init_labels, label_count



'''
export file
'''

def export_to_h5(data_export, path_export, key_h5='init', dtype=np.float32, verbose=0):
    '''Export numpy array to h5.

    Parameters
    ----------
    data_export (numpy array): data to export.
    path_export (str): path to h5 file.
    key_h5: key for h5 file.
    '''
    with h5py.File(path_export,'w') as f:
        f.create_dataset(key_h5, data=data_export.astype(dtype))

    print('H5 exported to: {}'.format(path_export))

def augment_data(data_xy, axis_xy=[1,2], augment={'flipxy':0,'flipx':0,'flipy':0,'flipc':0,'flipc_segment':1}):
    if 'flipxy' in augment and augment['flipxy']:
        data_xy = np.swapaxes(data_xy, axis_xy[0], axis_xy[1])
    if 'flipx' in augment and augment['flipx']:
        if axis_xy[0] == 0:
            data_xy = data_xy[::-1,...]
        if axis_xy[0] == 1:
            data_xy = data_xy[:, ::-1,...]
    if 'flipy' in augment and augment['flipy']:
        if axis_xy[1] == 1:
            data_xy = data_xy[:, ::-1,...]
        if axis_xy[1] == 2:
            data_xy = data_xy[:, :, ::-1,...]
    if 'shiftx' in augment and augment['shiftx']>0:
        if axis_xy[0] == 0:
            data_xy[:-augment['shiftx'],...] = data_xy[augment['shiftx']:,...]
        if axis_xy[0] == 1:
            data_xy[:,:-augment['shiftx'],...] = data_xy[:,augment['shiftx']:,...]
    if 'shifty' in augment and augment['shifty']>0:
        if axis_xy[1] == 1:
            data_xy[:,:-augment['shifty'],...] = data_xy[:,augment['shifty']:,...]
        if axis_xy[1] == 2:
            data_xy[:,:,:-augment['shifty'],...] = data_xy[:,:,augment['shifty']:,...]
    if 'shiftc' in augment and augment['shiftc']>0:
        if flipc_segment == 1:
            data_xy[:,:,:,:] = data_xy[:,:,:,::-1]
        elif flipc_segment == 2:
        # if augment['shiftc']==1:
            nc = int(data_xy.shape[-1]/flipc_segment)
            data_xy[:,:,:,:nc] = data_xy[:,:,:,(nc-1)::-1]
            data_xy[:,:,:,-nc:] = data_xy[:,:,:,-1:(nc-1):-1]
        # if augment['shiftc']==2:
    return data_xy
