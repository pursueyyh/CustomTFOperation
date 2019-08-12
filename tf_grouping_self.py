import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))
def query_and_interpolation(xyz1, xyz2, lh):
    '''
    Input:
        L: float32, search space
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        space_weight:(batch_size,npoint,27,3)
    '''
    return grouping_module.query_and_interpolation(xyz1, xyz2, lh)
@tf.RegisterGradient('QueryAndInterpolation')
def _query_and_interpolation_grad(op,grad_out):
    xyz1=op.input[0]
    xyz2=op.input[1]
    lh=op.input[2]
    return  [grouping_module.query_and_interpolation_grad(xyz1, xyz2, lh, grad_out), None]

if  __name__=='__main__':
    import numpy as np
    import time
    np.random.seed(100)
    tmp1 = np.random.random((32,512,3)).astype('float32')
    tmp2 = np.random.random((32,128,3)).astype('float32')
    with tf.device('/gpu:1'):
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        lh=0.1
	space_weight=query_and_interpolation(xyz1, xyz2, lh)
    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(space_weight)
        print(time.time() - now)
	print(xyz2.eval())
        print(ret.shape, ret.dtype)
        print(ret)
