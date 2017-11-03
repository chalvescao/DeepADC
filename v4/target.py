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
    def __init__(self,imin=0,imax=1.0,omin=0.25,omax=0.75,nBits=5):

        self.imin = imin
        self.imax = imax
        self.omin = omin
        self.omax = omax
        self.nBits = nBits

    def rSamp(self,bsz=64,cg=16):
        #dl = np.random.uniform(1.,2.)*2.0**(-self.nBits) 
        #dl = np.float32(np.reshape([-dl,dl],[2,1]))*0.49*(self.omax-self.omin)
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
    def Eval(self,gt,hpred,actl=None):
        wt = np.float32(2**np.arange(self.nBits))
        wt = tf.constant(np.reshape(wt,[1,self.nBits]))

        # Encoded value from predicted bits
        hpred = tf.stop_gradient(hpred) # just in case
        enc = tf.reduce_sum(wt*hpred,1,keep_dims=True)
        egt = tf.reduce_sum(wt*gt,1,keep_dims=True)

        # Compute soft-loss
        loss = None
        if actl is not None:
            enc0 = enc-wt*hpred
            enc1 = enc0+wt

            # Direct Loss
            wtpn = tf.abs(enc1-egt)-tf.abs(enc0-egt)
            wtp = tf.square(tf.nn.relu(-wtpn))
            wtn = tf.square(tf.nn.relu(wtpn))
            loss0 = tf.reduce_mean(wtp*actl[0] + wtn*actl[1])

            # Derivative Loss
            egt2 = _unpack(tf.reshape(egt,[2,-1,1]))
            hpd2 = _unpack(tf.reshape(enc,[2,-1,1]))
            en02 = _unpack(tf.reshape(enc0,[2,-1,self.nBits]))
            en12 = _unpack(tf.reshape(enc1,[2,-1,self.nBits]))

            at02 = _unpack(tf.reshape(actl[0],[2,-1,self.nBits]))
            at12 = _unpack(tf.reshape(actl[1],[2,-1,self.nBits]))


            gdiff = egt2[0]-egt2[1]

            wtpn = tf.abs(en12[0]-hpd2[1] - gdiff) - tf.abs(en02[0]-hpd2[1] - gdiff)
            wtp = tf.square(tf.nn.relu(-wtpn))
            wtn = tf.square(tf.nn.relu(wtpn))
            loss1a = tf.reduce_mean(wtp*at02[0] + wtn*at12[0])

            wtpn = tf.abs(hpd2[0]-en12[1] - gdiff) - tf.abs(hpd2[0] - en02[1] - gdiff)
            wtp = tf.square(tf.nn.relu(-wtpn))
            wtn = tf.square(tf.nn.relu(wtpn))
            loss1b = tf.reduce_mean(wtp*at02[1] + wtn*at12[1])

            ###
            loss = loss0+0.5*(loss1a+loss1b)
            
        # Error in encoded numerical value
        err1 = tf.reduce_mean(tf.abs(egt-enc))
        err2 = tf.sqrt(tf.reduce_mean(tf.square(egt-enc)))

        return [enc,loss,err1,err2]
            
