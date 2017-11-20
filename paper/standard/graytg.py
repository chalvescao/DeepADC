#-- Ayan Chakrabarti <ayan@wustl.edu>


import numpy as np
import tensorflow as tf


# tensorflow version compatibility
def _unpack(x,num=None):
    try:
        return tf.unstack(x,num)
    except:
        return tf.unpack(x,num)


def makeCode(nb1,nb2):
    levels = 2**nb1
    lchg = np.zeros(nb2)
    
    def tostr(b):
        return ' '.join([str(int(i)) for i in b])

    codes = [np.zeros(nb2)]
    strs = [tostr(codes[0])]

    for id in range(1,levels):
        idx = np.argsort(lchg)
        for k in range(len(idx)):
            nb = codes[-1].copy()
            nb[idx[k]] = 1-nb[idx[k]]
            st = tostr(nb)
            if st not in strs:
                lchg[idx[k]] = id
                codes.append(nb)
                strs.append(st)
                break
        assert len(strs) == id+1

    return np.stack(codes,0)
    

class Target:
    def __init__(self,imin=0,imax=1.0,omin=0.0,omax=1.0,nBits1=8,nBits2=10):

        self.imin = imin
        self.imax = imax
        self.omin = omin
        self.omax = omax
        self.nBits1 = nBits1
        self.nBits2 = nBits2
        self.code = makeCode(self.nBits1,self.nBits2)

    def rSamp(self,bsz=64,cg=16):
        dl = np.random.uniform(-4.,4.,size=[cg,bsz//cg])*2.0**(-self.nBits1)
        
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
        out = np.int64(np.round(out*(2**self.nBits1-1)))
        
        ovec = self.code[out.flatten(),:]
        return ovec

    # Computes encoded value, error, and loss nodes
    # Call with tensors / placeholders
    def Eval(self,gt,hpred,actl=None):
        err1 = 1.-tf.reduce_mean(tf.cast(tf.equal(gt,hpred),tf.float32))
        
        loss = None
        if actl is not None:
            loss = tf.reduce_mean(gt*actl[0] + (1-gt)*actl[1])
                
        return [loss,err1]
            
