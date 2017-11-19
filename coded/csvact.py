import numpy as np
import tensorflow as tf

FNAME = 'INV_VTC.csv'
COL = 'M'
SKIP = 2 # Ignore first two rows

XMIN=0
XMAX=1.2
XSTEP=0.01


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

        self.slope = tf.constant(slope,dtype=tf.float32)
        self.offs = tf.constant(offs,dtype=tf.float32)
        self.midx = slope.shape[0]-1

    def act(self,x):
        x = tf.clip_by_value(x,XMIN,XMAX)
        idx = tf.floor((tf.stop_gradient(x) - XMIN)/XSTEP)
        idx = tf.to_int64(tf.clip_by_value(idx,0,self.midx))

        return tf.gather(self.slope,idx)*x + tf.gather(self.offs,idx)

