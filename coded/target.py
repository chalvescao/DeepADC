#-- Ayan Chakrabarti <ayan@wustl.edu>


import numpy as np
import tensorflow as tf


def _tostr(b):
    return ' '.join([str(int(i)) for i in b])

def makecode(levels,bits):
    cur = np.zeros(bits)
    bits = [cur]
    bstrs = [_tostr(cur)]

    for id in range(1,levels):
        idx = np.argsort(lchg)
        for k in range(len(idx)):
            nb = cur.copy()
            nb[idx[k]] = 1-nb[idx[k]]
            st = tostr(nb)
            if st not in bstrs:
                lchg[idx[k]] = id
                cur = nb
                bstrs.append(_tostr(cur))
                bits.append(cur)
                break
        assert len(bits) == id+1


   
        

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
    def Eval(self,sig,gt,hpred,actl=None):
        

        
        return [enc,loss,err1,err2]
            
