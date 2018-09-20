import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tflearn.initializations import truncated_normal 
from tflearn.activations import relu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape,dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def a_layer(x,units):
    W = weight_variable([x.get_shape().as_list()[1],units])
    b = bias_variable([units])
    return relu(tf.matmul(x, W) + b)


def bi_layer(x0,x1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        W1p = weight_variable([x1.get_shape().as_list()[1],dim_pred])
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W1p),transpose_b=True)
    else:
        W0p = weight_variable([x0.get_shape().as_list()[1],dim_pred])
        return tf.matmul(tf.matmul(x0, W0p), 
                            tf.matmul(x1, W0p),transpose_b=True)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix,0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1)+1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix

def calculate_auc_and_draw(ground_truth, pred_list, draw=False):
    auc_score = roc_auc_score(ground_truth, pred_list)
    print 'auc_score: ', auc_score
    return auc_score
    fpr, tpr, thresholds = roc_curve(ground_truth, pred_list)
    # plt.plot(fpr, tpr)
    # plt.show()

