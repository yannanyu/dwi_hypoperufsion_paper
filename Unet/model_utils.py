import pandas
import numpy as np
import os
import h5py
import nibabel as nib
import pdb
from glob import glob
from tensorflow import math
from keras import backend as K
import keras.metrics
import keras.losses
from scipy.ndimage import rotate, affine_transform
import tensorflow as tf
# import tflearn
from sklearn.metrics import roc_auc_score

def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def vol_diff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = tf.cast(K.greater(K.flatten(y_pred), 0.5),'float32') # changed from just y_pred
    weight = 1 / (K.sum(y_true_f) + 1)  # weighted edditted by YAnnan 3/18. original value is 1/100000.
    difference = K.abs(K.sum(y_pred_f) - K.sum(y_true_f)) * weight
    return difference/4

def weighted_ce_l1_bycase(y_true, y_pred):
    # return seg_crossentropy_weighted_bycase(y_true, y_pred) + mean_absolute_error(y_true,y_pred) + sd_weights[0] * dice_coef_loss(y_true,y_pred) + sd_weights[0]*vol_diff(y_true, y_pred)
    return weighted_bce_loss(y_true, y_pred) + keras.losses.mean_absolute_error(y_true,y_pred) + 0.5 * (1 - dice_coef(y_true,y_pred)) +vol_diff(y_true, y_pred)
    ## added dice and vol loss on 2/4 by Yannan.

def mrs_accuracy(y_true,y_pred):
	diff = K.round(y_pred)-K.round(y_true)
	y_true_f = K.flatten(y_true)
	total = K.cast(tf.size(y_true_f),tf.int32)
	# upperlimit = tf.constant([1])
	# lowerlimit = tf.constant([-1])
	# pdb.set_trace()
	accuracy = K.sum(K.cast(math.logical_and(math.greater_equal(diff, -1), math.less_equal(diff, 1)),tf.int32))
	return accuracy/total

def cate_acc_plus_minus_one(y_true,y_pred):
    diff  = K.cast(K.argmax(y_true, axis=-1)- K.argmax(y_pred, axis=-1),K.floatx())
    accuracy = K.cast(math.logical_and(math.greater_equal(diff, -1), math.less_equal(diff, 1)),K.floatx())
    return  accuracy
# def mse_1_acc(y_true,y_pred):
# 	diff = tf.cast(tf.logical_and((y_true - y_pred) < 1.5, (y_true - y_pred) > -1.5),tf.float32) + 0.
# 	# pdb.set_trace()
# 	accuracy = K.sum(diff) / 128
# 	return 1 - accuracy
def bce_recall_accuracy(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred) + 0.5 * (1 - recall(y_true,y_pred))+ 0.5 * (1 - keras.metrics.binary_accuracy(y_true,y_pred))
def accuracy_loss(y_true,y_pred):
    return keras.losses.binary_crossentropy(y_true,y_pred) + 1 - keras.metrics.binary_accuracy(y_true,y_pred)
def weighted_bce_loss(y_true,y_pred):
    epsilon = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    pos_sample = K.sum(y_true_f)
    neg_sample = tf.cast(tf.size(y_true_f),tf.float32) - pos_sample
    ratio = neg_sample / (pos_sample + 1)
    ratio_one = ratio /(ratio + 1)
    ratio_zero = 1 / (ratio + 1)
    loss = -2*K.mean(ratio_one*y_true_f*tf.log(y_pred_f+epsilon) + ratio_zero*(1-y_true_f)*tf.log(1-y_pred_f+epsilon))
    return loss

def mse_accuracy(y_true,y_pred):
	# pdb.set_trace()
	loss = 1 - tf.cast(mrs_accuracy(y_true,y_pred), tf.float32) + 0.5 * keras.losses.mean_squared_error(y_true,y_pred)
	return loss

def mce_accuracy(y_true,y_pred):
	loss = 0.5 * keras.losses.mean_squared_error(y_true,y_pred) * keras.losses.mean_absolute_error(y_true,y_pred) + 1 - tf.cast(mrs_accuracy(y_true,y_pred), tf.float32)
	return loss
