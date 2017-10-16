#-- Ayan Chakrabarti <ayan@wustl.edu>


import numpy as np
import tensorflow as tf


class Target:
    def __init__(self,imin=0,imax=1.0,omin=0.25,omax=0.75,nBits=8):

        self.imin = imin
        self.imax = imax
        self.omin = omin
        self.omax = omax
        self.nBits = nBits

    def rSamp(self,bsz=64):
        sig = np.float32(self.imin + np.random.sample([bsz,1])*(self.imax-self.imin))
        ovec = self.s2o(sig)

        return sig,ovec


    def uSamp(self,bsz=8192):
        sig = np.linspace(self.imin,self.imax,bsz+1)[:-1]
        sig = np.float32(np.reshape(sig,[bsz,1]))
        ovec = self.s2o(sig)

        return sig,ovec

    # Ground truth quantization function
    def s2o(self,sig):
        out = np.maximum(0.,sig-self.omin)
        out = np.minimum(1.,out/(self.omax-self.omin))
        out = np.int64(out*(2**self.nBits-1))
        
        ovec = []
        for i in range((self.nBits+1)//8):
            ovec = ovec + [np.unpackbits(np.uint8(out%256),axis=1)[:,::-1]]
            out = out // 256
        if len(ovec) == 1:
            ovec = ovec[0]
        else:
            ovec = np.stack(ovec,axis=1)
        ovec = ovec[:,:self.nBits]    
        return ovec
        


    # Computes encoded value, error, and loss nodes
    # Call with tensors / placeholders
    def Eval(self,gt,hpred,actl=None):
        wt = np.float32(2**np.arange(self.nBits))
        wt = tf.constant(np.reshape(wt,[1,self.nBits]))

        # Encoded value from predicted bits
        enc = tf.reduce_sum(wt*hpred,1,keep_dims=True)

        # Compute soft-loss
        loss = None
        if actl is not None:
            #loss = tf.reduce_mean((gt*actl[0] + (1-gt)*actl[1])) * np.float32(self.nBits)
            loss = tf.reduce_mean((wt)*(gt*actl[0] + (1-gt)*actl[1])) * np.float32(self.nBits)
            
        # Error in encoded numerical value
        egt = tf.reduce_sum(wt*gt,1,keep_dims=True)
        err1 = tf.reduce_mean(tf.abs(egt-enc))
        err2 = tf.sqrt(tf.reduce_mean(tf.square(egt-enc)))

        return [enc,loss,err1,err2]
            
