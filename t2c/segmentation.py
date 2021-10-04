"""
Created by Michele Bianco, 9 July 2021
"""

import numpy as np
import pkg_resources
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K
except:
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops 
from tensorflow.python.ops import array_ops 
from tensorflow.python.ops import math_ops 

def sigmoid_balanced_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, beta=None, name=None):
    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel,labels, logits)
    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name: 
        logits = ops.convert_to_tensor(logits, name="logits") 
        labels = ops.convert_to_tensor(labels, name="labels") 
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %(logits.get_shape(), labels.get_shape())) 
        zeros = array_ops.zeros_like(logits, dtype=logits.dtype) 
        cond = (logits >= zeros) 
        relu_logits = array_ops.where(cond, logits, zeros) 
        neg_abs_logits = array_ops.where(cond, -logits, logits) 
        balanced_cross_entropy = relu_logits*(1.-beta)-logits*labels*(1.-beta)+math_ops.log1p(math_ops.exp(neg_abs_logits))*((1.-beta)*(1.-labels)+beta*labels)
        return tf.reduce_mean(balanced_cross_entropy)

def balanced_cross_entropy(y_true, y_pred):
    """
    To decrease the number of false negatives, set beta>1. To decrease the number of false positives, set beta<1.
    """
    beta = tf.maximum(tf.reduce_mean(1 - y_true), tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = K.log(y_pred / (1 - y_pred))
    return sigmoid_balanced_cross_entropy_with_logits(logits=y_pred, labels=y_true, beta=beta)


def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """

    intersection = K.sum(K.abs(y_true * y_pred))
    #intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1, otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



################################################################

class segunet21cm:
    def __init__(self, tta=1, verbose=False):
        """ SegU-Net: segmentation of 21cm images with U-shape network (Bianco et al. 2021, https://arxiv.org/abs/2102.06713)
            - tta (int): default 0 (super-fast, no pixel-error map) implement the error map
                         with time-test aumentated techique in the prediction process
            - verbose (bool): default False, activate verbosity
         
         Description:
            tta = 0 : fast (~7 sec), it tends to be a few percent less accurate (<2%) then the other two cases, no pixel-error map (no TTA manipulation)
            tta = 1 : medium (~17 sec), accurate and preferable than tta=0, with pixel-error map (3 samples)
            tta = 2 : slow (~10 min), accurate, with pixel-error map (~100 samples)
        
         Returns:
            - X_seg (ndarray) : recovered binary field (1 = neutral and 0 = ionized regions)
            - X_err (ndarray) : pixel-error map of the recovered binary field
        
         Example:
          $ from tools21cm import segmentation
          $ seg = segmentation.segunet21cm(tta=1, verbose=True)   # load model (need to be done once)
          $ Xseg, Xseg_err = seg.prediction(x=dT3)

         Print of the Network's Configuration file:
          [TRAINING]
          BATCH_SIZE = 64
          AUGMENT = NOISESMT
          IMG_SHAPE = 128, 128
          CHAN_SIZE = 256
          DROPOUT = 0.05
          KERNEL_SIZE = 3
          EPOCHS = 100
          LOSS = balanced_cross_entropy
          METRICS = iou, dice_coef, binary_accuracy, binary_crossentropy
          LR = 1e-3
          RECOMP = False
          GPUS = 2
          PATH = /home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/data2D_128_030920/
 
          [RESUME]
          RESUME_PATH = /home/michele/Documents/PhD_Sussex/output/ML/dataset/outputs/new/02-10T23-52-36_128slice/
          BEST_EPOCH = 56
          RESUME_EPOCH = 66

        """
        self.TTA = tta
        self.VERBOSE = verbose

        if(self.TTA == 2):
            # slow
            self.MANIP = self.IndependentOperations(verbose=self.VERBOSE)
        elif(self.TTA == 1):
            # fast
            self.MANIP = {'opt0': [lambda a: a, 0, 0]}
        elif(self.TTA == 0):
            # super-fast
            self.MANIP = {'opt0': [lambda a: a, 0, 0]}
        
        self.NR_MANIP = len(self.MANIP)

        # load model
        MODEL_NAME = pkg_resources.resource_filename('tools21cm', 'input_data/segunet_02-10T23-52-36_128slice_ep56.h5')
        MODEL_EPOCH = 56
        METRICS = {'balanced_cross_entropy':balanced_cross_entropy, 'iou':iou, 'dice_coef':dice_coef} 
        self.MODEL_LOADED = load_model(MODEL_NAME, custom_objects=METRICS)
        print(' Loaded model: %s' %MODEL_NAME)

    def UniqueRows(self, arr):
        """ Remove duplicate row array in 2D data 
                - arr (narray): array with duplicate row
            
            Example:
            >> d = np.array([[0,1,2],[0,1,2],[0,0,0],[0,0,2],[0,1,2]])
            >> UniqueRows(d) 
            
            array([[0, 0, 0],
                    [0, 0, 2],
                    [0, 1, 2]])
        """
        arr = np.array(arr)

        if(arr.ndim == 2):
            arr = np.ascontiguousarray(arr)
            unique_arr = np.unique(arr.view([('', arr.dtype)]*arr.shape[1]))
            new_arr = unique_arr.view(arr.dtype).reshape((unique_arr.shape[0], arr.shape[1]))
        elif(arr.ndim == 1):
            new_arr = np.array(list(dict.fromkeys(arr)))

        return new_arr


    def IndependentOperations(self, verbose=False):
        ''' How many unique manipulations (horzontal and vertical flip, rotation, etc...) 
            can we operate on a cube? 
            Each indipendent operation is considered as an additional rappresentation
            of the same coeval data, so that it can be considered for errorbar with SegU-Net '''

        data = np.array(range(3**3)).reshape((3,3,3)) 

        func = [lambda a: a,
                np.fliplr, 
                np.flipud, 
                lambda a: np.flipud(np.fliplr(a)),
                lambda a: np.fliplr(np.flipud(a))]
        axis = [0,1,2] 
        angl_rot = [0,1,2,3] 


        tot_manipl_data_flat = np.zeros((len(func)*len(axis)*len(angl_rot), data.size)) 
        tot_operations = {'opt%d' %k:[] for k in range(0,len(func)*len(axis)*len(angl_rot))} 

        i = 0 
        for f in func: 
            cube = f(data)
            for rotax in axis: 
                ax_tup = [0,1,2] 
                ax_tup.remove(rotax)
                for rot in angl_rot:
                    tot_manipl_data_flat[i] = np.rot90(cube, k=rot, axes=ax_tup).flatten() 
                    # function, axis of rotation, angle of rotation, slice index
                    tot_operations['opt%d' %i] = [f, rotax, rot] 
                    i += 1 

        uniq_manipl_data_flat = self.UniqueRows(tot_manipl_data_flat).astype(int)
        uniq_operations = {}

        for iumdf, uniq_mdf in enumerate(uniq_manipl_data_flat):
            for itmdf, tot_mdf in enumerate(tot_manipl_data_flat):
                if(all(uniq_mdf == tot_mdf)):
                    uniq_operations['opt%d' %iumdf] = tot_operations['opt%d' %itmdf]
                    break
                    
        assert uniq_manipl_data_flat.shape[0] == len(uniq_operations)
        if(verbose): print('tot number of (unique) manipulation we can do on a cube: %d' %(len(uniq_operations)))

        return uniq_operations


    def prediction(self, x):
        img_shape = x.shape
        if(self.TTA == 2):
            X_tta = np.zeros((np.append(3*len(self.MANIP), img_shape)))
        elif(self.TTA == 1):
            X_tta = np.zeros((np.append(3*len(self.MANIP), img_shape)))
        elif(self.TTA == 0):
            X_tta = np.zeros((np.append(len(self.MANIP), img_shape)))
        
        if(self.VERBOSE):
            loop = tqdm(range(len(self.MANIP)))
        else:
            loop = range(len(self.MANIP))

        for iopt in loop:
            opt, rotax, rot = self.MANIP['opt%d' %iopt]
            ax_tup = [0,1,2] 
            ax_tup.remove(rotax)

            cube = np.rot90(opt(x), k=rot, axes=ax_tup) 
            X = cube[np.newaxis, ..., np.newaxis]

            for j in range(img_shape[0]):
                if(self.TTA == 0):
                    X_tta[iopt,j,:,:] = self.MODEL_LOADED.predict(X[:,j,:,:,:], verbose=0).squeeze()
                else:
                    X_tta[iopt,j,:,:] = self.MODEL_LOADED.predict(X[:,j,:,:,:], verbose=0).squeeze()
                    X_tta[iopt+len(self.MANIP),:,j,:] = self.MODEL_LOADED.predict(X[:,:,j,:,:], verbose=0).squeeze()
                    X_tta[iopt+len(self.MANIP)*2,:,:,j] = self.MODEL_LOADED.predict(X[:,:,:,j,:], verbose=0).squeeze()

        for itta in range(X_tta.shape[0]):
            opt, rotax, rot = self.MANIP['opt%d' %(itta%len(self.MANIP))]
            ax_tup = [0,1,2] 
            ax_tup.remove(rotax)
            X_tta[itta] = opt(np.rot90(X_tta[itta], k=-rot, axes=ax_tup))

        X_seg = np.round(np.mean(X_tta, axis=0))
        X_err = np.std(X_tta, axis=0)

        return X_seg, X_err
