import tensorflow as tf
import numpy as np
X1 =[1.,2.,3.]
X2=[1.,2.,3.]
Y =[1.,2.,3.]
m = n_samples = len(X)

W1=tf.placeholder(tf.float32)
W2=tf.placeholder(tf.float32)
b =tf.Variable(tf.random_uniform([1],-10,10))
#set model
hypothesis = W1*X1+W2*X2+b

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
