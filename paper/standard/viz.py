#-- Ayan Chakrabarti <ayan@wustl.edu>

import numpy as np

VDD=1.2


############## Inverter characteristics
#########################################################################################
MID_IN = 0.48
MID_OUT = 0.5281
FNAME = 'INV_VTC.csv'
COL = 'M'; SKIP = 2; XMIN=0;XMAX=1.2;XSTEP=0.01
class VTC:
    def __init__(self):
        lines = open(FNAME).readlines()[SKIP:]
        data = [[float(x) if len(x) > 0 else 0.0 for x in l.split(',')] for l in lines]
        data=np.float32(data)

        self.GT = data[:,[0,ord(COL)-ord('A')]]
        
        data = data[:,ord(COL)-ord('A')]

        assert(data.shape[0] == int((XMAX-XMIN)/XSTEP)+1)

        slope = (data[1:]-data[:-1])/XSTEP
        offs = data[:-1] - slope*(XMIN+np.arange(slope.shape[0])*XSTEP)

        self.slope = slope
        self.offs = offs
        self.midx = slope.shape[0]-1

    def act(self,x):
        x = np.minimum(XMAX,np.maximum(XMIN,x))
        idx = floor((tf.stop_gradient(x) - XMIN)/XSTEP)
        idx = int(np.minimum(self.midx,np.maximum(0.,idx)))

        return self.slope[idx]*x+self.offs[idx]
#########################################################################################


#########################################################################################
class Circuit:
    def __init__(self,fname):
        data = np.load(fname)

        w0 = data['w0'] 


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
