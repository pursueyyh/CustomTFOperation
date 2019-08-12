import tensorflow as tf
import numpy as np
from tf_grouping_self import query_and_interpolation

class QueryAndInterpolationTest(tf.test.TestCase):
	def test(self):
		import time
		print(time.time())
		np.random.seed(100)
		tmp1 = np.random.random((32,512,3)).astype('float32')
		tmp2 = np.random.random((32,128,3)).astype('float32')
		with tf.device('/gpu:1'):
			xyz1 = tf.constant(tmp1)
			xyz2 = tf.constant(tmp2)
			L_=0.1
			L=tf.constant(L_)
			space_weight=query_and_interpolation(xyz1, xyz2, L)
		with tf.Session('') as sess:
			now = time.time() 
			for _ in range(100):
				ret = sess.run(space_weight)
			print(time.time() - now)
			print(ret.shape, ret.dtype)
			print(ret)
	def test_grad(self):
		with tf.device('/gpu:0'):
			points = tf.constant(np.random.random((1,128,16)).astype('float32'))
			print(points)
			xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
			xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
			L_ = 0.1
			L = tf.constant(L_)
			weight_space = query_and_interpolation(xyz1, xyz2, L)
			print(weight_space)

		with self.test_session():
			print("---- Going to compute gradient error")
			err = tf.test.compute_gradient_error(points, (1,128,16), weight_space, (1,8,27,3))
			print(err)
			self.assertLess(err, 1e-4) 

if __name__=='__main__':
  tf.test.main() 
