#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

import numpy as np

VDD=1.2
RGS=0.1


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
        idx = np.floor((x - XMIN)/XSTEP)
        idx = np.int64(np.minimum(self.midx,np.maximum(0.,idx)))

        return self.slope[idx]*x+self.offs[idx]

vtc = VTC()
#########################################################################################


#########################################################################################
class Circuit:
    def __init__(self,fname):
        data = np.load(fname)
        self.data = data

        w0 = (data['w0'])
        w0 = np.maximum(-(1-1e-4),np.minimum((1-1e-4),w0))
        w0 = RGS * w0 / (1-np.abs(w0))

        self.R0p = np.maximum(0,w0)
        self.R0n = np.maximum(0,-w0)
        self.V0 = np.maximum(-VDD,np.minimum(VDD,data['b0']))

        w1 = data['wf']
        w1p = np.maximum(0,w1)
        w1n = np.maximum(0,-w1)
        div = np.maximum(np.sum(w1p,0),np.sum(w1n,0))*1.01

        w1p = w1p / div
        w1n = w1n / div
        b1 = data['bf'] / div
        div = np.maximum(1.,np.abs(b1)/VDD)
        w1p = w1p / div; w1n = w1n / div; b1 = b1 / div

        self.R1p = w1p * RGS / (1-np.sum(w1p,0))
        self.R1n = w1n * RGS / (1-np.sum(w1n,0))
        self.V1 = b1


        self.nBits = self.V1.shape[0]
        self.nHidden = self.V0.shape[0]

    def pert(self,sigma):
        for v in [self.R0p,self.R1p,self.R0n,self.R1n]:
            v[...] = v*np.exp(np.random.randn(*v.shape)*sigma)

    def encode(self,x):
        x = np.reshape(x,[-1,1])
        R0p = self.R0p; R0n = self.R0n; V0 = self.V0
        R1p = self.R1p; R1n = self.R1n; V1 = self.V1

        y = np.matmul(x,R0p)/(R0p+RGS) - np.matmul(x,R0n)/(R0n+RGS) + V0
        y = vtc.act(y)
        y  = np.matmul(y,R1p)/(RGS+np.sum(R1p,0)) - np.matmul(y,R1n)/(RGS+np.sum(R1n,0)) + V1
        return np.float32(y > 0)
        

#################################################


import sys

tm =  np.linspace(0.,2*np.pi,2**20)
signal = 0.5 + 0.5 * np.sin(tm)


enobs = []
for it in range(100):
    ckt = Circuit(sys.argv[1])
    ckt.pert(float(sys.argv[2]))

    quant = ckt.encode(signal)

    wts = 2**np.float32(np.arange(ckt.nBits))
    val = np.float32(np.sum(wts*quant,1))
    vs = np.unique(val)
    val2 = val.copy()
    for v in vs:
        o = np.mean(signal[val2 == v])
        val[val2 == v] = o

    gt = signal

    mse = np.mean((gt-val)**2)
    SINAD = 10*np.log10( np.var(gt) + mse) - 10*np.log10(mse)
    ENOB = (SINAD-1.76)/6.02
    enobs.append(ENOB)

enobs = np.sort(np.float32(enobs))
q25 = enobs[25]
q50 = enobs[50]
q75 = enobs[75]

print("ENOB: (%.3f, %.3f, %.3f)" % (q25,q50,q75))
