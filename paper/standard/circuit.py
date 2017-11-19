#-- Ayan Chakrabarti <ayan@wustl.edu>

import numpy as np
import tensorflow as tf
import csvact as ca

_vtc = ca.VTC()
VDD=1.2
MID_IN = 0.48
MID_OUT = 0.5281


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


        w0 = np.float32(np.random.uniform(-0.5,0.5,[1,nHidden]))
        self.wts['w0'] =  tf.Variable(tf.constant(w0,shape=[1,nHidden]),dtype=tf.float32)
        self.wts['b0'] =  tf.Variable(tf.constant(MID_IN,shape=[nHidden]),dtype=tf.float32)
        
        wf = np.float32(np.random.uniform(-np.sqrt(3.0/nHidden),np.sqrt(3.0/nHidden),[nHidden,nBits]))
        self.wts['wf'] =  tf.Variable(tf.constant(wf,shape=[nHidden,nBits]),dtype=tf.float32)
        bf = np.float32(-MID_OUT*np.sum(wf,axis=0))
        self.wts['bf'] =  tf.Variable(tf.constant(bf,shape=[nBits]),dtype=tf.float32)

    def encode(self,x):
        w0 = tf.nn.tanh(self.wts['w0'])
        y = tf.matmul(x,w0)+self.wts['b0']
        y = tf.matmul(activ1(y),self.wts['wf']) + self.wts['bf']
        return [activ2(y), activ2L(y)]
