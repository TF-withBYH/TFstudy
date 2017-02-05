import tensorflow as tf
import numpy as np
X =[1.,2.,3.]
Y =[1.,2.,3.]
m = n_samples = len(X)

W=tf.placeholder(tf.float32)

#set model
hypothesis = tf.mul(X,W)

#set coset
cost = tf.reduce_sum(tf.pow(hypothesis-Y,2))/(m)

#init variables
init = tf.initialize_all_variables()

#for graph
W_val = []
cost_val = []

#launch
sess=tf.Session()
sess.run(init)

for i in (-30,50):
	print i*0.1,sess.run(cost,feed_dict={W:i*0.1})
	W_val.append(i*0.1)
	cost_val.append(sess.run(cost,feed_dict={W:i*0.2}))
