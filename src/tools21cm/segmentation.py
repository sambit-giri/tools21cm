"""
Created by Michele Bianco, 9 July 2021
"""

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# import tensorflow as tf
# try:
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras import backend as K
# except:
#     from tensorflow.python.keras.models import load_model
#     from tensorflow.python.keras import backend as K
# from tensorflow.python.ops import nn_ops
# from tensorflow.python.framework import ops 
# from tensorflow.python.ops import array_ops 
# from tensorflow.python.ops import math_ops 

def sigmoid_balanced_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, beta=None, name=None):
    import tensorflow as tf
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.framework import ops 
    from tensorflow.python.ops import array_ops 
    from tensorflow.python.ops import math_ops 

    nn_ops._ensure_xent_args("sigmoid_cross_entropy_with_logits", _sentinel, labels, logits)
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
    import tensorflow as tf
    from tensorflow.keras import backend as K

    beta = tf.maximum(tf.reduce_mean(1 - y_true), tf.keras.backend.epsilon())
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = K.log(y_pred / (1 - y_pred))
    return sigmoid_balanced_cross_entropy_with_logits(logits=y_pred, labels=y_true, beta=beta)

def iou(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def dice_coef(y_true, y_pred, smooth=1):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def precision(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    y_true, y_pred = K.clip(y_true, K.epsilon(), 1-K.epsilon()), K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    TP = K.sum(y_pred * y_true)
    FP = K.sum(y_pred * (1 - y_true))
    return TP/(TP + FP + K.epsilon())

def recall(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    y_true, y_pred = K.clip(y_true, K.epsilon(), 1-K.epsilon()), K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    TP = K.sum(y_pred * y_true)
    FN = K.sum((1 - y_pred) * y_true)
    return TP/(TP + FN + K.epsilon())

def r2score(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res/(SS_tot + K.epsilon())

################################################################

class segunet21cm:
    def __init__(self, tta=1, verbose=False):
        self.TTA = tta
        self.VERBOSE = verbose

        if self.TTA == 2:
            self.MANIP = self.IndependentOperations(verbose=self.VERBOSE)
        else:
            self.MANIP = {'opt0': [lambda a: a, 0, 0]}

        self.NR_MANIP = len(self.MANIP)
        self.MODEL_LOADED = None  # Placeholder for the model

    def load_model(self):
        import tensorflow as tf
        import importlib.resources as pkg_resources

        METRICS = {
            'balanced_cross_entropy': balanced_cross_entropy,
            'r2score': r2score,
            'iou': iou,
            'precision': precision,
            'recall': recall
        }

        MODEL_NAME = pkg_resources.resource_filename('tools21cm', 'input_data/segunet_03-11T12-02-05_128slice_ep35.h5')
        self.MODEL_LOADED = tf.keras.models.load_model(MODEL_NAME, custom_objects=METRICS)
        print('Loaded model: %s' % MODEL_NAME)

    def UniqueRows(self, arr):
        arr = np.array(arr)

        if arr.ndim == 2:
            arr = np.ascontiguousarray(arr)
            unique_arr = np.unique(arr.view([('', arr.dtype)]*arr.shape[1]))
            new_arr = unique_arr.view(arr.dtype).reshape((unique_arr.shape[0], arr.shape[1]))
        elif arr.ndim == 1:
            new_arr = np.array(list(dict.fromkeys(arr)))

        return new_arr

    def IndependentOperations(self, verbose=False):
        data = np.array(range(3**3)).reshape((3,3,3)) 

        func = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a))]
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
                    tot_operations['opt%d' %i] = [f, rotax, rot] 
                    i += 1 

        uniq_manipl_data_flat = self.UniqueRows(tot_manipl_data_flat).astype(int)
        uniq_operations = {}

        for iumdf, uniq_mdf in enumerate(uniq_manipl_data_flat):
            for itmdf, tot_mdf in enumerate(tot_manipl_data_flat):
                if all(uniq_mdf == tot_mdf):
                    uniq_operations['opt%d' %iumdf] = tot_operations['opt%d' %itmdf]
                    break
                    
        assert uniq_manipl_data_flat.shape[0] == len(uniq_operations)
        if verbose: print('tot number of (unique) manipulation we can do on a cube: %d' %(len(uniq_operations)))

        return uniq_operations

    def prediction(self, x):
        if self.MODEL_LOADED is None:
            self.load_model()

        img_shape = x.shape
        X_tta = np.zeros((np.append(3*len(self.MANIP), img_shape)))

        if self.VERBOSE:
            loop = tqdm(range(len(self.MANIP)))
        else:
            loop = range(len(self.MANIP))

        for iopt in loop:
            opt, rotax, rot = self.MANIP['opt%d' % iopt]
            ax_tup = [0, 1, 2] 
            ax_tup.remove(rotax)

            cube = np.rot90(opt(x), k=rot, axes=ax_tup) 
            X = cube[np.newaxis, ..., np.newaxis]

            for j in range(img_shape[0]):
                X_tta[iopt, j, :, :] = self.MODEL_LOADED.predict(X[:, j, :, :, :], verbose=0).squeeze()
                X_tta[iopt+len(self.MANIP), :, j, :] = self.MODEL_LOADED.predict(X[:, :, j, :, :], verbose=0).squeeze()
                X_tta[iopt+len(self.MANIP)*2, :, :, j] = self.MODEL_LOADED.predict(X[:, :, :, j, :], verbose=0).squeeze()

        for itta in range(X_tta.shape[0]):
            opt, rotax, rot = self.MANIP['opt%d' % (itta % len(self.MANIP))]
            ax_tup = [0, 1, 2] 
            ax_tup.remove(rotax)
            X_tta[itta] = opt(np.rot90(X_tta[itta], k=-rot, axes=ax_tup))

        X_seg = np.round(np.mean(X_tta, axis=0))
        X_err = np.std(X_tta, axis=0)

        return X_seg, X_err
