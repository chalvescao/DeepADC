#-- Ayan Chakrabarti <ayan@wustl.edu>

import numpy as np
import tensorflow as tf
import csvact as ca

_vtc = ca.VTC()

# Activation of hidden layer
def activ1(x):
    #return tf.sigmoid(x)
    return _vtc.act(x)

# "Hard" / non-differentiable activation of final layer
def activ2(x):
    return tf.cast(x>0,tf.float32)

# Differentiable activation+loss wrt true value being positive or negative
def activ2L(x):
    return [tf.nn.softplus(-x), tf.nn.softplus(x)]


class Circuit:
    def __init__(self,nHidden=4,nBits=8):
        self.wts = {}
        self.wts['w0'] =  tf.Variable(tf.random_uniform([1,nHidden],minval=-1.0,maxval=1.0,dtype=tf.float32))
        # Change bias init to center based on our new activation
        self.wts['b0'] =  tf.Variable(tf.constant(0.48,shape=[nHidden],dtype=tf.float32))
        
        self.wts['wf'] =  tf.Variable(tf.random_uniform([nHidden,nBits],minval=-np.sqrt(3.0/nHidden),maxval=np.sqrt(3.0/nHidden),dtype=tf.float32))
        self.wts['bf'] =  tf.Variable(tf.constant(0,shape=[nBits],dtype=tf.float32))

    def encode(self,x):
        w0 = self.wts['w0']
        w0b = tf.stop_gradient(w0)
        w0 = w0 + tf.clip_by_value(w0b,-1.0,1.0) - w0b
        
        y = tf.matmul(x,w0)+self.wts['b0']
        y = tf.matmul(activ1(y)-0.6,self.wts['wf']) + self.wts['bf']
        return [activ2(y), activ2L(y)]
