#-- Ayan Chakrabarti <ayan@wustl.edu>


import numpy as np
import tensorflow as tf


# tensorflow version compatibility
def _unpack(x,num=None):
    try:
        return tf.unstack(x,num)
    except:
        return tf.unpack(x,num)
    

class Target:
    def __init__(self,imin=0,imax=1.0,omin=0.0,omax=1.0,nBits=5):

        self.imin = imin
        self.imax = imax
        self.omin = omin
        self.omax = omax
        self.nBits = nBits

    def rSamp(self,bsz=64,cg=16):
        dl = np.random.uniform(-4.,4.,size=[cg,bsz//cg])*2.0**(-self.nBits)
        
        sig = np.float32(self.imin + np.random.sample([1,bsz//cg])*(self.imax-self.imin))
        sig = np.maximum(self.imin,np.minimum(self.imax,np.reshape(sig+dl,[bsz,1])))
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
        for i in range((self.nBits+1)//self.nBits):
            ovec = ovec + [np.unpackbits(np.uint8(out%(2**self.nBits)),axis=1)[:,::-1]]
            out = out // (2**self.nBits) 
        if len(ovec) == 1:
            ovec = ovec[0]
        else:
            ovec = np.stack(ovec,axis=1)
        ovec = ovec[:,:self.nBits]    
        return ovec

    # Computes encoded value, error, and loss nodes
    # Call with tensors / placeholders
    def Eval(self,sig,gt,hpred,actl=None,cwt=1.0):
        wt = np.float32(2**np.arange(self.nBits))
        wt = tf.constant(np.reshape(wt,[1,self.nBits]))

        # Encoded value from predicted bits
        hpred = tf.stop_gradient(hpred) # just in case
        enc = tf.reduce_sum(wt*hpred,1,keep_dims=True)
        egtQ = tf.reduce_sum(wt*gt,1,keep_dims=True)

        #egt = tf.clip_by_value(tf.stop_gradient(sig),self.omin,self.omax)-self.omin
        egt = (sig - self.omin) * ((2**self.nBits)/(self.omax-self.omin))-0.5
        egt = tf.clip_by_value(egt,-0.5,2**self.nBits-0.5)
        
        # Compute soft-loss
        loss = None
        if actl is not None:
            lossC = tf.reduce_mean(wt*wt*(gt*actl[0] + (1-gt)*actl[1])) * np.float32(self.nBits) * cwt

            enc0 = enc-wt*hpred
            enc1 = enc0+wt

            # Direct Loss
            wtpn = tf.abs(enc1-egt)-tf.abs(enc0-egt)
            wtp = tf.square(tf.nn.relu(-wtpn))
            wtn = tf.square(tf.nn.relu(wtpn))
            loss0 = tf.reduce_mean(wtp*actl[0] + wtn*actl[1])

            loss = lossC+loss0
                
        # Error in encoded numerical value
        err1 = tf.reduce_mean(tf.abs(egtQ-enc))
        err2 = tf.sqrt(tf.reduce_mean(tf.square(egtQ-enc)))

        return [enc,loss,err1,err2]
            
