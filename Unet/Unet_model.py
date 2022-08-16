import numpy as np
import pdb
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Flatten, Conv2D,Conv3D,MaxPooling3D,MaxPooling2D,Dropout, Concatenate, Input, Reshape, Activation,SpatialDropout2D, Conv2DTranspose, Add, multiply, concatenate,Lambda
from keras.regularizers import l1,l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import keras.metrics
import keras.losses
import keras.optimizers
from keras.utils import to_categorical
from glob import glob
from random import shuffle, seed, choice
from skimage.io import imsave
from keras.callbacks import ModelCheckpoint
from keras.backend import set_session
from tool_package import *
from generator_unet import *
from model_utils import *

def numeric_branch(inputs):
    x = Dense(16, activation='relu')(inputs)
    x = Dense(8, activation = 'relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(4, activation = 'relu')(x)

    return x

def create_attention_block_2D(g, x, output_channel, padding='same'):
    g1 = Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(g)
    g1 = BatchNormalization(axis=-1)(g1)
    x1 = Conv2D(output_channel, kernel_size=1, strides=1, padding=padding)(x)
    x1 = BatchNormalization(axis=-1)(x1)
    psi = Activation("relu")(Add()([g1, x1]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding=padding)(psi)
    psi = BatchNormalization(axis=-1)(psi)
    psi = Activation("sigmoid")(psi)
    return multiply([x, psi])
def lambda_bn(x,batch_norm):
    if batch_norm:
        output = BatchNormalization()(x)
    else:
        output = x
    return output

def dropout_hybrid_unet(numeric_dim):
    number_input = Input(shape=numeric_dim)
    return number_input

def unet_first_conv(conv1,
                num_conv_per_pooling
                ,batch_norm,
                num_channel_first,
                activation_conv,
                kernel_initializer,
                pool_size,
                dropout_rate,
                verbose):
    '''
    return convolution results and pooling + dropout results
    '''
    conv_identity = []
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        conv1 = lambda_bn(conv1,batch_norm)
        conv1 = Conv2D(num_channel_first, (3, 3),
                       padding="same",
                       activation=activation_conv,
                       kernel_initializer=kernel_initializer)(conv1)

        if (i + 1) % 2 == 0 and i != 1:
            conv1 = keras_add([conv_identity[-1], conv1])
            # pdb.set_trace() # jiahong apr
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)
    pool1 = SpatialDropout2D(rate=dropout_rate)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)
    return conv1, pool1

def unet_encoders(conv_encoders,
                pool_encoders,
                num_channel_input,
                num_channel_first,
                num_poolings,
                num_conv_per_pooling,
                batch_norm,
                activation_conv,
                kernel_initializer,
                pool_size,
                dropout_rate,
                verbose):
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            conv_encoder = lambda_bn(conv_encoder,batch_norm)
            conv_encoder = Conv2D(
                num_channel, (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_encoder = keras_add([conv_identity[-1], conv_encoder])
                pdb.set_trace() # jiahong apr
        pool_encoder = MaxPooling2D(pool_size=pool_size)(conv_encoder)
        pool_encoder = SpatialDropout2D(rate=dropout_rate)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)
    return list_num_features, pool_encoders, conv_encoders

def tile_numeric_info(x):
    expand_number = K.expand_dims(x[0],axis=1)
    expand_number = K.expand_dims(expand_number,axis=1)
    conv_center_shape = x[1].get_shape().as_list()
    return K.tile(expand_number,[1,conv_center_shape[1],conv_center_shape[2],1])

def hybrid_unet(num_channel_input=1, num_channel_output=1,numeric_dim=4,
                   img_rows=128, img_cols=128, epochs=10,
                   y=np.array([0, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function='mean_squared_error',
                   metrics_monitor=['mean_absolute_error', 'mean_squared_error'],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   dropout_rate=0.25, pool_size=(2,2),batch_norm=True,
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    inputs = Input((img_rows, img_cols, num_channel_input))
    number_input = Input(shape=(numeric_dim,))

    conv1 = inputs
    conv1, pool1 = unet_first_conv(conv1,
                    num_conv_per_pooling,
                    batch_norm,
                    num_channel_first,
                    activation_conv,
                    kernel_initializer,
                    pool_size,
                    dropout_rate,
                    verbose)


    # encoder layers with pooling
    conv_encoders = [inputs,conv1]
    pool_encoders = [inputs,pool1]
    list_num_features, pool_encoders, conv_encoders = unet_encoders(conv_encoders,
                                                                    pool_encoders,
                                                                    num_channel_input,
                                                                    num_channel_first,
                                                                    num_poolings,
                                                                    num_conv_per_pooling,
                                                                    batch_norm,
                                                                    activation_conv,
                                                                    kernel_initializer,
                                                                    pool_size,
                                                                    dropout_rate,
                                                                    verbose)

    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros')(pool_encoders[-1])
    # conv_center = lambda_bn(conv_center,batch_norm) #add BN for bottleneck
    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros',name='output_layer_before_merge')(conv_center)
    # conv_center = lambda_bn(conv_center,batch_norm) #add BN for bottleneck
    ## merge with numeric input
    tile = Lambda(tile_numeric_info)([number_input, conv_center])
    conv_center = Concatenate(axis=-1)([conv_center, tile])

    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros',name='output_layer_after_merge')(conv_center)
    # conv_center = lambda_bn(conv_center,batch_norm) #add BN for bottleneck
    ## above edited by Yannan 2020/6/21
    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):

        attention_gated = create_attention_block_2D(Conv2DTranspose(
            list_num_features[-i], (2,2), strides=pool_size, padding="same",
            activation=activation_conv,
            kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i],list_num_features[-i])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2,2), strides=pool_size, padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), attention_gated])


        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            conv_decoder = lambda_bn(conv_decoder,batch_norm)
            conv_decoder = Conv2D(
                list_num_features[-i], (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
        conv_decoder = SpatialDropout2D(rate=dropout_rate)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])
        pdb.set_trace() # jiahong apr

    # construct model
    model = Model(outputs=conv_output, inputs=[inputs,number_input])
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = keras.optimizers.Adam(lr=lr_init, decay = lr_init/epochs) #add decay by yannan 2020/3/9
    else:
        optimizer = keras.optimizers.Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model

def dropout_unet(num_channel_input=1, num_channel_output=1,
                   img_rows=128, img_cols=128, epochs=10,
                   y=np.array([0, 1]),  # change to output_range in the future
                   output_range=None,
                   lr_init=None, loss_function='mean_squared_error',
                   metrics_monitor=['mean_absolute_error', 'mean_squared_error'],
                   num_poolings=4, num_conv_per_pooling=3, num_channel_first=32,
                   with_baseline_concat=True, with_baseline_addition=-1,  # -1 means no
                   dropout_rate=0.25, pool_size=(2,2),batch_norm=True,
                   activation_conv='relu',  # options: 'elu', 'selu'
                   activation_output=None,  # options: 'tanh', 'sigmoid', 'linear', 'softplus'
                   kernel_initializer='zeros',  # options: 'he_normal'
                   verbose=1):
    inputs = Input((img_rows, img_cols, num_channel_input))


    '''
    Modification descriptioin (Charles 11/16/18)

    Added residual blocks to the encoding and decoding sides of the Unet
        See if statements within the for loops
    Added drop out (can also try SpatialDropout2D)
        See drop out layer at end of for loops
    '''
    '''
    Below was modified by Charles
    '''
    # step1
    conv_identity = []
    conv1 = inputs
    for i in range(num_conv_per_pooling):
        if i % 2 == 0 and i != 0:
            conv_identity.append(conv1)
        conv1 = lambda_bn(conv1,batch_norm)
        conv1 = Conv2D(num_channel_first, (3, 3),
                       padding="same",
                       activation=activation_conv,
                       kernel_initializer=kernel_initializer)(conv1)

        if (i + 1) % 2 == 0 and i != 1:
            conv1 = keras_add([conv_identity[-1], conv1])
            # pdb.set_trace() # jiahong apr
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)
    pool1 = SpatialDropout2D(rate=dropout_rate)(pool1)
    if verbose:
        print('conv1:', conv1, pool1)

    # encoder layers with pooling
    conv_encoders = [inputs,conv1]
    pool_encoders = [inputs,pool1]
    conv_identity = []
    list_num_features = [num_channel_input, num_channel_first]
    for i in range(1, num_poolings):
        # step2
        conv_encoder = pool_encoders[-1]
        num_channel = num_channel_first * (2 ** (i))
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                conv_identity.append(conv_encoder)
            conv_encoder = lambda_bn(conv_encoder,batch_norm)
            conv_encoder = Conv2D(
                num_channel, (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_encoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_encoder = keras_add([conv_identity[-1], conv_encoder])
                pdb.set_trace() # jiahong apr
        pool_encoder = MaxPooling2D(pool_size=pool_size)(conv_encoder)
        pool_encoder = SpatialDropout2D(rate=dropout_rate)(pool_encoder)
        if verbose:
            print('encoding#{0}'.format(i), conv_encoder, pool_encoder)
        pool_encoders.append(pool_encoder)
        conv_encoders.append(conv_encoder)
        list_num_features.append(num_channel)

    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros')(pool_encoders[-1])
    conv_center = Conv2D(list_num_features[-1]*2, (3, 3), padding="same", activation="relu",
                        kernel_initializer=kernel_initializer,
                        bias_initializer='zeros',name='output_layer')(conv_center)

    conv_decoders = [conv_center]
    if verbose:
        print('centers', conv_center)

    # decoder steps
    deconv_identity = []
    for i in range(1, num_poolings + 1):

        attention_gated = create_attention_block_2D(Conv2DTranspose(
            list_num_features[-i], (2,2), strides=pool_size, padding="same",
            activation=activation_conv,
            kernel_initializer=kernel_initializer)(conv_decoders[-1]), conv_encoders[-i],list_num_features[-i])
        upsample_decoder = concatenate(
            [Conv2DTranspose(
                list_num_features[-i], (2,2), strides=pool_size, padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoders[-1]), attention_gated])


        conv_decoder = upsample_decoder
        for j in range(num_conv_per_pooling):
            if j % 2 == 0 and j != 0:
                deconv_identity.append(conv_decoder)
            conv_decoder = lambda_bn(conv_decoder,batch_norm)
            conv_decoder = Conv2D(
                list_num_features[-i], (3, 3), padding="same",
                activation=activation_conv,
                kernel_initializer=kernel_initializer)(conv_decoder)

            if (j + 1) % 2 == 0 and j != 1:
                conv_decoder = keras_add([deconv_identity[-1], conv_decoder])
                # pdb.set_trace() # jiahong apr
        conv_decoder = SpatialDropout2D(rate=dropout_rate)(conv_decoder)
        conv_decoders.append(conv_decoder)
        if verbose:
            print('decoding#{0}'.format(i), conv_decoder, upsample_decoder)
    '''
    Above was modified by Charles
    '''
    # concatenate with baseline
    if with_baseline_concat:
        conv_decoder = conv_decoders[-1]
        conv_decoder = concatenate([conv_decoder, inputs])
        if verbose:
            print('residual concatenate:', conv_decoder)

    '''
    '''
    # output layer activation
    if output_range is None:
        output_range = np.array(y).flatten()
    if activation_output is None:
        if max(output_range) <= 1 and min(output_range) >= 0:
            activation_output = 'sigmoid'
        elif max(output_range) <= 1 and min(output_range) >= -1:
            activation_output = 'tanh'
        else:
            activation_output = 'linear'
    conv_output = Conv2D(num_channel_output, (1, 1),
                         padding="same",
                         activation=activation_output)(conv_decoder)
    if verbose:
        print('output:', conv_output)

    # add baselind channel
    if with_baseline_addition > 0:
        print('add residual channel {0}#{1}'.format(
            with_baseline_addition, num_channel_input // 2))
        conv_output = keras_add(
            [conv_output, inputs[:, :, :, num_channel_input // 2]])
        pdb.set_trace() # jiahong apr

    # construct model
    model = Model(outputs=conv_output, inputs=inputs)
    if verbose:
        print('model:', model)

    # optimizer and loss
    if lr_init is not None:
        # ,0.001 rho=0.9, epsilon=1e-08, decay=0.0)
        optimizer = keras.optimizers.Adam(lr=lr_init, decay = lr_init/epochs) #add decay by yannan 2020/3/9
    else:
        optimizer = keras.optimizers.Adam()
    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=metrics_monitor)

    return model



def test_model(model,input_path,sample_dim,output_dim,model_name,model_name_tag,pred_dim=(91,109,60),read_dim =(128,128,60,1),
               slice_expand=2,print_dict={'Test loss':'loss'},model_output_layer_name='output_layer',
               name_tag='',save=True,source_path='/data/yannanyu/PWImasked185/',source_path2='/data/yannanyu/newstroke_masked/'):
    x,y = load_volume_h5(input_path,input_dim=read_dim, output_dim=(128,128,60))
    # pdb.set_trace()
    input,output = windowslide_input(x,y, sample_dim, output_dim)
    pred =  model.predict(input)
    pred = np.array(pred)
    pred = np.squeeze(pred.transpose(1,2,0,3))
    pred_pad = cut_image(pred,pred_dim)
    score_bestmodel = model.evaluate(input, output, verbose=0)
    for key in print_dict:
        print(key,score_bestmodel[model.metrics_names.index(print_dict[key])])

    intermediate_layer_model = Model(inputs=model.input,
                         outputs=model.get_layer('output_layer').output) ##('dense_{}'.format(11*(i-1)+5)).output)
    last_image_layer = intermediate_layer_model.predict(input) ## shape = (slice, row, col, feature)
    last_layer_output = last_image_layer.reshape(-1,last_image_layer.shape[0]) ##shape=(total feature number, slice)
    # pdb.set_trace()

    subject = input_path.split('/')[-2]
    #save results into txt
    affine_path = source_path + '{}/LESION.nii'.format(subject)
    if not os.path.exists(affine_path):
        affine_path = source_path2 + '{}/LESION.nii'.format(subject)
    _,affine = load_nii(affine_path)
    if save:
        output_path = 'models/'+ model_name +model_name_tag + '/'
        np.savetxt(output_path + 'bottleneck_{}.csv'.format(subject),last_layer_output, delimiter=",",fmt = '%.3f')
        savenib(pred_pad,affine,output_path, export_name=subject+'.nii')
    del model
    return None

def test_hybrid_model(model,input_path,sample_dim,output_dim,
                    numeric_dict,model_name,model_name_tag,
                    pred_dim=(91,109,60),read_dim =(128,128,60,1),slice_expand=2,
                    print_dict={'Test loss':'loss'},
                    model_output_layer_name='output_layer',
                    name_tag='',save=True,
                    add_dimension=False,source_path='',source_path2='/data/yannanyu/newstroke_masked/'):
    x,number,y = generator_hybrid_load_h5(input_path,number_list=np.empty((len(numeric_dict),)),numeric_dict=numeric_dict)
    x = np.transpose(x,[1,2,3,0])
    # pdb.set_trace()
    # if not add_dimension:
    #     x = np.reshape(x, [x.shape[0],x.shape[1],x.shape[2]*x.shape[3]])
    if read_dim != x.shape:
        x_test = pad_image(x,read_dim)
        y_test = pad_image(y,read_dim)
    else:
        x_test = x
        y_test = y
    input,output = windowslide_input(x_test,y_test, sample_dim, output_dim)
    number = np.tile(number, (input.shape[0],1))
    pred =  model.predict([input,number])
    pred = np.array(pred)
    pred = np.squeeze(pred.transpose(1,2,0,3))
    pred_pad = cut_image(pred,pred_dim)
    score_bestmodel = model.evaluate([input,number], output, verbose=1)
    for key in print_dict:
        print(key,score_bestmodel[model.metrics_names.index(print_dict[key])])

    intermediate_layer_before_merge = Model(inputs=model.input,
                         outputs=model.get_layer('output_layer_before_merge').output) ##('dense_{}'.format(11*(i-1)+5)).output)
    intermediate_layer_after_merge = Model(inputs=model.input,
                         outputs=model.get_layer('output_layer_after_merge').output) ##('dense_{}'.format(11*(i-1)+5)).output)
    before_merge_layer = intermediate_layer_before_merge.predict([input,number]) ## shape = (slice, row, col, feature)
    before_merge_output = before_merge_layer.reshape(-1,before_merge_layer.shape[0]) ##shape=(total feature number, slice)
    after_merge_layer = intermediate_layer_after_merge.predict([input,number]) ## shape = (slice, row, col, feature)
    after_merge_output = after_merge_layer.reshape(-1,after_merge_layer.shape[0]) ##shape=(total feature number, slice)
    # pdb.set_trace()

    subject = input_path.split('/')[-2]
    print(subject)
    #save results into txt
    read_source_path = source_path + subject
    if not os.path.exists(read_source_path):
        read_source_path =source_path2 + subject
    subj_source_path=glob(read_source_path)[0]
    _,affine = load_nii(subj_source_path+'/LESION.nii'.format(subject))
    if save:
        output_path = 'models/'+ model_name +model_name_tag + '/'
        np.savetxt(output_path + 'bottleneck_{}_before_merge.csv'.format(subject),before_merge_output, delimiter=",",fmt = '%.3f')
        np.savetxt(output_path + 'bottleneck_{}_after_merge.csv'.format(subject),after_merge_output, delimiter=",",fmt = '%.3f')
        savenib(pred_pad,affine,output_path, export_name=subject+'.nii')
    del model
    return None


def evaluate_model(model,input_path,input_dim,output_dim,slice,print_dict={'Test loss':'loss'},cnn3d=False,mode='val'):
    x_test, y_test= load_each_slice_h5(input_path, input_dim, output_dim,slice)
    if cnn3d:
        x_test = np.expand_dims(x_test,axis=4)
    score_model = model.evaluate(x_test, y_test, verbose=0)
    for key in print_dict:
        print(key,score_model[model.metrics_names.index(print_dict[key])])

    del model

def evaluate_model_generator(model,input_path,batch_size,print_dict, sample_dim, output_dim, slice_expand,cnn3d=False, steps=50):
    data_gen = input_generator(input_path, batch_size,sample_dim=sample_dim,output_dim=output_dim,add_dimension=cnn3d,slice_expand=slice_expand,output=1)
    score_model = model.evaluate_generator(data_gen,steps=steps,use_multiprocessing=False,workers=1, verbose=0)
    for key in print_dict:
        print(key,score_model[model.metrics_names.index(print_dict[key])])
