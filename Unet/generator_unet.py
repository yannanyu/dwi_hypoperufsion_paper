import numpy as np
import pdb
from glob import glob
import os
import h5py
from random import shuffle, seed,randint
from tool_package import *
from keras.utils import to_categorical

# np.random.seed(1990)
def array_h5(h5_file,key):
    output = h5_file.get(key)
    output = np.array(output)
    return output

def get_h5_path(path,pt_list=[],dim=(128,128,6),include_mirror = False, include_aug=False):

    if pt_list == []:
        h5_path = glob(path+'*/*.hdf5')
    else:
        h5_path = []
        search_term_mir = '*.hdf5'
        for pt in pt_list:
            search_term_pt = '{}/inputs'.format(pt)
            if not include_mirror:
                search_term_mir = '*0.hdf5'
            search_term = search_term_pt + search_term_mir
            pt_path = glob(path+search_term)
            h5_path.extend(pt_path)
    # shuffle(h5_path)
    print('finish getting path.')
    return h5_path

def generator_load_h5(read_path: str,train_data='',h5key='init'):
    assert os.path.exists(read_path), 'file path does not exists:{}'.format(read_path)
    train = h5py.File(read_path +  train_data,'r')
    x_train = array_h5(train,h5key)
    train.close()

    output_name = 'output' + os.path.basename(read_path)[6:]
    y_path = os.path.join(os.path.dirname(read_path),output_name)
    train = h5py.File(y_path +  train_data,'r')
    y_train = array_h5(train,h5key)
    train.close()
    return x_train, y_train

def generator_hybrid_load_h5(read_path: str,
                                number_list,
                                train_data='',
                                h5key='init',
                                numeric_dict={}):
    assert os.path.exists(read_path), 'file path does not exists:{}'.format(read_path)
    train = h5py.File(read_path +  train_data,'r')
    x_train = array_h5(train,h5key)
    for key in numeric_dict:
        # print('key',key,numeric_dict[key])
        # print(array_h5(train,key))
        number_list[numeric_dict[key]] = array_h5(train,key)
    train.close()


    output_name = 'output' + os.path.basename(read_path)[6:]
    y_path = os.path.join(os.path.dirname(read_path),output_name)
    train = h5py.File(y_path +  train_data,'r')
    y_train = array_h5(train,h5key)
    train.close()
    return x_train, number_list, y_train


def hybrid_input_generator(inputpath,bs,
                sample_dim=(128,128,6),
                output_dim =(128,128,1),
                numeric_dim=4,
                numeric_dict={},
                add_dimension=False,
                slice_expand=2,
                output=1):
    '''
    bs = batch_size
    '''
    while True:
        x_train = np.empty((bs,)+sample_dim)
        y_train = np.empty((bs,)+output_dim)
        number_train = np.empty((bs,numeric_dim))

        index = np.random.choice(inputpath,bs,replace=False)
        for i in range(bs):
            #choose random in samples
            x, number,y = generator_hybrid_load_h5(index[i],number_list=number_train[i],numeric_dict=numeric_dict)
            slice = np.random.randint(slice_expand,x.shape[-1]-slice_expand)
            x_image = x[:,:,:,slice-slice_expand:slice+slice_expand+1]
            x_image = np.transpose(x_image,[1,2,3,0])
            if not add_dimension:
                x_image = np.reshape(x_image, [x_image.shape[0],x_image.shape[1],-1]) #not concate on z axis,穿插z&4th dimension
            y_image = y[:,:,slice]
            number_train[i] = number
            if len(y_image.shape)<=2:
                y_image = np.expand_dims(y_image,axis=4)
            # print(x_image.shape,y_image.shape, sample_dim)
            if sample_dim != x_image.shape:
                x_train[i] = pad_image(x_image,sample_dim)
                y_train[i] = pad_image(y_image,output_dim)
            else:
                x_train[i] = x_image
                y_train[i] = y_image

        if add_dimension:
            x_train = np.expand_dims(x_train,axis=5)
        yield [x_train, number_train], y_train


def input_generator(inputpath,bs, sample_dim=(128,128,6),output_dim =(128,128,1),add_dimension=False,slice_expand=2, output=1):
    '''
    bs = batch_size
    '''
    while True:
        x_train = np.empty((bs,)+sample_dim)
        y_train = np.empty((bs,)+output_dim)


        index = np.random.choice(inputpath,bs,replace=False)
        for i in range(bs):
            #choose random in samples
            x, y = generator_load_h5(index[i])
            # print(x.shape,y.shape)
            slice = np.random.randint(slice_expand,x.shape[-1]-slice_expand)
            x_image = x[:,:,:,slice-slice_expand:slice+slice_expand+1]
            x_image = np.transpose(x_image,[1,2,3,0])
            if not add_dimension:
                x_image = np.reshape(x_image, [x_image.shape[0],x_image.shape[1],-1]) #not concate on z axis,穿插z&4th dimension
            y_image = y[:,:,slice]
            if len(y_image.shape)<=2:
                y_image = np.expand_dims(y_image,axis=4)
            # print(x_image.shape,y_image.shape, sample_dim)
            if sample_dim != x_image.shape:
                x_train[i] = pad_image(x_image,sample_dim)
                y_train[i] = pad_image(y_image,output_dim)
            else:
                x_train[i] = x_image
                y_train[i] = y_image
        if add_dimension:
            x_train = np.expand_dims(x_train,axis=5)
        yield x_train, y_train



def load_each_slice_h5(input_path, input_dim, output_dim, slice, slice_expand=2):
    x,y = generator_load_h5(input_path)
    x_image = x[:,:,slice-slice_expand:slice+slice_expand+1]
    y_image = y[:,:,slice]
    if len(y_image.shape)<=2:
        y_image = np.expand_dims(y_image,axis=4)
    if input_dim != x_image.shape:
        x_test = pad_image(x_image,input_dim)
        y_test = pad_image(y_image,output_dim)
    else:
        x_test = x_image
        y_test = y_image
    return x_test,y_test

def windowslide_input(x, y, sample_dim,output_dim,slice_expand=2):
    input = np.empty((x.shape[2]- 2*slice_expand,) + sample_dim)
    output = np.empty((x.shape[2]- 2*slice_expand,) + output_dim)
    for slice in range(slice_expand,x.shape[2]-slice_expand):
        x_image = x[:,:,slice-slice_expand:slice+slice_expand+1,...]
        if len(x_image.shape) >3:
            x_image = np.reshape(x_image, [x_image.shape[0],x_image.shape[1],-1])
        y_image = y[:,:,slice]
        if len(y_image.shape)<=2:
            y_image = np.expand_dims(y_image,axis=4)
        if sample_dim != x_image.shape:
            input[slice-slice_expand] = pad_image(x_image,sample_dim)
            output[slice-slice_expand] = pad_image(y_image,output_dim)
        else:
            input[slice-slice_expand] = x_image
            output[slice-slice_expand] = y_image
    return input,output

def load_volume_h5(input_path, input_dim=(128,128,91), output_dim=(128,128,91),transpose_factor=[1,2,3,0]):
    x,y = generator_load_h5(input_path)
    x = np.transpose(x,transpose_factor)
    if input_dim != x.shape:
        x_test = pad_image(x,input_dim)
        y_test = pad_image(y,output_dim)
    else:
        x_test = x
        y_test = y
    return x_test,y_test
