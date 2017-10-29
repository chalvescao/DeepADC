import numpy as np
import tensorflow as tf
import circuitActivation as ca
import target as tg

customActiv = ca.customActivation()
nHidden=24
nBits=8

# Activation of hidden layer
def activ1(x):
    return tf.sigmoid(x)

# "Hard" / non-differentiable activation of final layer
def activ2(x):
    return tf.cast(x>0,tf.float32)

# Differentiable activation+loss wrt true value being positive or negative
def activ2L(x):
    return [tf.nn.softplus(-x), tf.nn.softplus(x)]

# Define a circuit activation function
def activCircuit(x):
    return customActiv.tf_circuitActivation(x)
    

wts = {}
wts['w0'] =  tf.Variable(tf.random_uniform([1,nHidden],minval=-np.sqrt(3.0),maxval=np.sqrt(3.0),dtype=tf.float32))
wts['b0'] =  tf.Variable(tf.constant(0,shape=[nHidden],dtype=tf.float32))
wts['w1'] =  tf.Variable(tf.random_uniform([nHidden,nBits],minval=-np.sqrt(3.0/nHidden),maxval=np.sqrt(3.0/nHidden),dtype=tf.float32))
wts['b1'] =  tf.Variable(tf.constant(0,shape=[nBits],dtype=tf.float32))

data = tg.Target(nBits=nBits)
sigs,bits = data.rSamp(4096)


y = tf.matmul(sigs,wts['w0'])+wts['b0']
# y = tf.matmul(activ1(y),wts['w1']) + wts['b1']
y = tf.matmul(activCircuit(y),wts['w1']) + wts['b1']


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(y.eval()[0])
    print len(y.eval())

