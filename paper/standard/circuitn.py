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
    def __init__(self,nHidden=4,nBits=8,nzstd=0.01):
        self.nHidden = nHidden
        self.nBits = nBits
        self.nstd = nzstd
        
        self.wts = {}
        w0 = np.float32(np.random.uniform(-0.5,0.5,[1,nHidden]))
        
        self.wts['w0'] =  tf.Variable(tf.constant(w0,shape=[1,nHidden]),dtype=tf.float32)
        b0 = -0.5*w0 + MID_IN
        self.wts['b0'] =  tf.Variable(tf.constant(b0,shape=[nHidden]),dtype=tf.float32)
        
        wf = np.float32(np.random.uniform(-np.sqrt(3.0/nHidden),np.sqrt(3.0/nHidden),[nHidden,nBits]))
        self.wts['wf'] =  tf.Variable(tf.constant(wf,shape=[nHidden,nBits]),dtype=tf.float32)
        bf = np.float32(-MID_OUT*np.sum(wf,axis=0))
        self.wts['bf'] =  tf.Variable(tf.constant(bf,shape=[nBits]),dtype=tf.float32)

        v = tf.clip_by_value(self.wts['w0'],-1.0,1.0)
        v1 = tf.abs(v) / tf.maximum(1e-8,tf.abs(self.wts['w0']))
        v1 = (self.wts['b0'] - MID_IN)*tf.reshape(v1,[-1]) + MID_IN
        v1 = tf.clip_by_value(v1,-VDD,VDD)
        
        with tf.control_dependencies([v,v1]):
            self.cOp = [tf.assign(self.wts['w0'],v),tf.assign(self.wts['b0'],v1)]

    def encode(self,x):
        w0 = self.wts['w0']*tf.exp(tf.random_normal([1,self.nHidden],stddev=self.nstd))
        wf = self.wts['wf']*tf.exp(tf.random_normal([self.nHidden,self.nBits],stddev=self.nstd))
        y = tf.matmul(x,w0)+self.wts['b0']
        y = tf.matmul(activ1(y),wf) + self.wts['bf']
        return [activ2(y), activ2L(y)]
