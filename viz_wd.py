#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

#this is visualization for conductance clip

import numpy as np

VDD=1.2
C = 0.00002 # The summation for all conductance in first layer
C_1 = 0.000005 # The summation for a pair of conductance in second layer
############## Inverter characteristics
#########################################################################################
MID_IN = 0.47261 # This should be around 0.474
MID_OUT = 0.5281

R_min = 1/C
R_max = 5e5

r = 1000 # Reram resistance resolution clip
r1 = 1000

C_min = 1/(10000000*R_max)

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

        w0_p_s = [] #to store the scaled positive weight
        w0_n_s = [] #to store the scaled negative weight

        b0_p_i = [] #to store the initial trained bias belongs to positive weight
        b0_n_i = [] #to store the initial trained bias belongs to negative weight

        index_P = [] #to store the index of positive weight and corresponding bias
        index_N = [] #to store the index of negative weight and corresponding bias

        w0 = (data['w0'])
        b0 = (data['b0'])
        l = b0.shape[0]

        for i in range (l): ## classify the weight by negative and positive value
            if w0[0,i] > 0:
                w0_p_s.append(w0[0,i])
                b0_p_i.append(b0[i])
                index_P.append(i)
            else:
                w0_n_s.append(w0[0,i])
                b0_n_i.append(b0[i])
                index_N.append(i) ## end

        lp = np.array(index_P).shape[0] #the length of positive weight array
        ln = np.array(index_N).shape[0] #the length of negative weight array

        b0_p = np.array(b0_p_i)
        b0_n = np.array(b0_n_i)

        alpha_p1 = 2 * (MID_IN - b0_p) / VDD # based on the Eq.(2) in report;
        alpha_p2 = 2 * (np.array(b0_p) - MID_IN) / VDD + 2 * np.array(w0_p_s) # based on the Eq.(2) in report;
        a_scalep1 = np.max(alpha_p1)
        a_scalep2 = np.max(alpha_p2)
        a_scalep = np.maximum(a_scalep1,a_scalep2) # get the maximum scaling factor for positive side

        alpha_n1 = 2 * (b0_n - MID_IN) / VDD # based on the Eq.(4) in report;
        alpha_n2 = 2 * (MID_IN - np.array(b0_n)) / VDD - 2 * np.array(w0_n_s) # based on the Eq.(4) in report;
        a_scalen1 = np.max(alpha_n1)
        a_scalen2 = np.max(alpha_n2)
        a_scalen = np.maximum(a_scalen1,a_scalen2) # get the maximum scaling factor for negative side
        a_scale = 1.1 * np.maximum(a_scalep, a_scalen) # get the maximum scaling factor for crossbar. Here, 1.1 is a factor to avoid some weights are zero

        V_bias = VDD * a_scale
        VDD_new = V_bias # new power supply
        w0_p_s = w0_p_s / a_scale # to scale positive weight
        w0_n_s = w0_n_s / a_scale # to scale negative weight
        V_shift = VDD_new / 2 - MID_IN # This is voltage shift for inverter
        V_H = V_shift + VDD # This is the highest output voltage of shifted inverter
        V_L = V_shift # This is the lowest output voltage of shifted inverter

        # get conductance from trained positive weights, # based on the Eq.(a) in report;
        G_j1BP = np.ones(lp, int) * C_min # instead of letting G_j1BP be 0, just giving it a very small value.
        G_j1AP = C * np.array(w0_p_s) + G_j1BP
        w_sb = C_min / C # the weight contributed by G_j1BP
        G_j0AP = C * (V_shift + b0_p) / V_bias - w_sb
        G_j0BP = C - G_j1AP - G_j0AP - G_j1BP
        #end

        # get conductance from trained negative weights
        G_j1AN = np.ones(ln, int) * C_min # instead of letting G_j1AN be 0, just giving it a very small value.# based on the Eq.(b) in report;
        G_j1BN = -C * np.array(w0_n_s) + G_j1AN
        G_j0AN = C * (V_shift + b0_n + w0_n_s * V_bias - w_sb* V_bias) / V_bias
        G_j0BN = C - G_j1AN - G_j0AN - G_j1BN
        # end

        G_j1A = w0.copy() # to store the conductance for signal side of sub-crossbar array A
        G_j1B = w0.copy() # to store the conductance for signal side of sub-crossbar array B
        G_j0A = w0.copy() # to store the conductance for bias side of sub-crossbar array A
        G_j0B = w0.copy() # to store the conductance for bias side of sub-crossbar array B
        Bias = w0.copy()

        index_P = np.reshape(np.array(index_P),(1,lp))
        index_N = np.reshape(np.array(index_N), (1, ln))

        G_j1A[0,index_P] = np.reshape(np.array(G_j1AP),(1,lp))
        G_j1A[0,index_N] = np.reshape(np.array(G_j1AN),(1,ln))
        G_j1B[0,index_P] = np.reshape(np.array(G_j1BP),(1,lp))
        G_j1B[0,index_N] = np.reshape(np.array(G_j1BN),(1,ln))

        G_j0A[0,index_P] = np.reshape(np.array(G_j0AP),(1,lp))
        G_j0A[0,index_N] = np.reshape(np.array(G_j0AN),(1,ln))
        G_j0B[0,index_P] = np.reshape(np.array(G_j0BP),(1,lp))
        G_j0B[0,index_N] = np.reshape(np.array(G_j0BN),(1,ln))

        Bias[0,index_P] = V_bias
        Bias[0,index_N] = V_bias

        self.G0SA = np.maximum(0,G_j1A) ##conductance for singal side
        self.G0SB = np.maximum(0,G_j1B) ##conductance for signal side
        self.G0BA = np.maximum(0,G_j0A) ##conductance for bias side
        self.G0BB = np.maximum(0,G_j0B) ##conductance for bias side
        self.Bias = Bias
        self.VDD_new = VDD_new
        self.V_shift = V_shift
        self.V_H = V_H
        self.V_L = V_L
        # end

        # This is to normalize the weight and bias
        w1 = data['wf']
        b1 = data['bf']

        self.nBits = b1.shape[0]
        self.nHidden = b0.shape[0]
        h = self.nHidden

        w_max = np.max(np.abs(w1))
        w1 = w1 / ((h + 1) * 1.5 * w_max)
        b1 = b1 / ((h + 1) * 1.5 * w_max)

        div = np.maximum(1.,np.abs(b1)/(V_H + V_L))
        w1 = w1 / div
        b1 = b1 / div
        # end

        # This is to get conductance from trained weights
        G1SA = C_1 * (1 + (h + 1) * w1) / 2 # Conductance for signal side,# based on the Eq.(8) in report;
        G1SB = C_1 * (1 - (h + 1) * w1) / 2
        w1sb = G1SB / ((1 + h) * C_1)

        ## This is to get the weight of bias.
        w1ba = (b1 + (V_H + V_L) / 2 - np.sum(w1, axis=0) * V_shift - np.sum(w1sb, axis=0) * (V_H + V_L)) / (V_H + V_L)
        #end

        self.G1BA = (1 + h) * C_1 * w1ba # Conductance for bias side
        self.G1BB = C_1 - self.G1BA
        self.G1SA = G1SA
        self.G1SB = G1SB
        #end

    def encode(self,x):
        x = np.reshape(x,[-1,1])# input for positive side

        G0SA = self.G0SA; G0SB = self.G0SB; G0BA = self.G0BA; G0BB = self.G0BB; V_shift = self.V_shift; VDD_new = self.VDD_new
        G1SA = self.G1SA; G1SB = self.G1SB; G1BA = self.G1BA; G1BB = self.G1BB; V_H = self.V_H; V_L = self.V_L

        # This is to get the resistance of first layer.
        R0SA = np.round(1 / (G0SA * r)) * r
        R0SB = np.round(1 / (G0SB * r)) * r
        R0BA = np.round(1 / (G0BA * r)) * r
        R0BB = np.round(1 / (G0BB * r)) * r
        # end

        # To get the resolution of ReRAM resistance in first layer
        N0SA = np.min(R0SA)
        N0SB = np.min(R0SB)
        N0BA = np.min(R0BA)
        N0BB = np.min(R0BB)
        N0 = np.minimum(np.minimum(N0SA,N0SB),np.minimum(N0BA,N0BB))
        Res_0 = (R_max - N0)/r
        # end

        # This is to get the circuit model of first layer.
        C_sum = 1/R0SA + 1/R0SB + 1/R0BA + 1/R0BB
        w0bp = (1/R0BA) / C_sum
        V0bp = ((1/R0SB) / C_sum * VDD_new) + w0bp * VDD_new
        w0sp = (1/R0SA - 1/R0SB) / C_sum

        w0bn = 1/R0BB / C_sum
        V0bn = ((1/R0SA) / C_sum) * VDD_new + w0bn * VDD_new
        w0sn = (1/R0SB - 1/R0SA) / C_sum

        # Equivalently to VTC shift / # Differential outputs of hidden layer
        y_p0 = np.matmul(x,w0sp) + V0bp - V_shift
        y_p0ac = vtc.act(y_p0) + V_shift # Positive path
        y_n0 = np.matmul(x,w0sn) + V0bn - V_shift
        y_n0ac = vtc.act(y_n0) + V_shift # Negative path
        yac = y_p0ac + y_n0ac
        #end/ By this step, we get the output of hidden layer
        #end

        # This is to get the resistance of second layer.
        R1SA = np.round(1/(G1SA*r1))*r1
        R1SB = np.round(1/(G1SB*r1))*r1
        R1BA = np.round(1/(G1BA*r1))*r1
        R1BB = np.round(1/(G1BB*r1))*r1

        N1SA = np.min(np.min(R1SA))
        N1SB = np.min(np.min(R1SB))
        N1BA = np.min(np.min(R1BA))
        N1BB = np.min(np.min(R1BB))

        N1 = np.minimum(np.minimum(N1SA, N1SB), np.minimum(N1BA, N1BB))

        Res_1 = (R_max - N1) / r1
        # end

        V_BIAS = V_H + V_L # This is the bias input for the second layer.
        biasp =  V_BIAS * (1/R1BA) / (np.sum(1/R1SA, 0) + np.sum(1/R1SB, 0) + 1/R1BA + 1/R1BB)
        biasn =  V_BIAS * (1/R1BB) / (np.sum(1/R1SA, 0) + np.sum(1/R1SB, 0) + 1/R1BA + 1/R1BB)
        y_p1 = (np.matmul(y_p0ac, 1/R1SA) + np.matmul(y_n0ac, 1/R1SB)) / (np.sum(1/R1SA, 0) + np.sum(1/R1SB, 0) + 1/R1BA + 1/R1BB) + biasp # Positive path
        y_n1 = (np.matmul(y_p0ac, 1/R1SB) + np.matmul(y_n0ac, 1/R1SA)) / (np.sum(1/R1SA, 0) + np.sum(1/R1SB, 0) + 1/R1BA + 1/R1BB) + biasn # Negative path

        return np.float32((y_p1 - y_n1) > 0)

#################################################


import sys

ckt = Circuit(sys.argv[1])

tm =  np.linspace(0.,2*np.pi,2**20)

signal = ckt.Bias[0,1]/2.4 + ckt.Bias[0,1]/2.4 * np.sin(tm)# interesting, the signal scale is different with power scale. 1.2~1

quant = ckt.encode(signal)

wts = 2**np.float32(np.arange(ckt.nBits))
val = np.float32(np.sum(wts*quant,1)) # Binary Representation

# Best decoder
vs = np.unique(val)
val2 = val.copy()
for v in vs:
    o = np.mean(signal[val2 == v])
    val[val2 == v] = o

gt = signal

mse = np.mean((gt-val)**2)
SINAD = 10*np.log10( np.var(gt) + mse) - 10*np.log10(mse)
ENOB = (SINAD-1.76)/6.02
res = 1/(C * r)
print("ENOB = %.2f (%d)" % (ENOB,ckt.nBits))
print("res",  res)

if len(sys.argv) == 3:
    import matplotlib as mp
    mp.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(tm,gt,'-g',linewidth=1.5)
    plt.plot(tm,val,'-r',linewidth=1.5)

    plt.xlim([-0.1,2*np.pi+0.1])
    plt.ylim([-0.1,3])

    #end
    plt.title("%d Bits, %d Hidden Neurons: ENOB = %.2f" % (ckt.nBits,ckt.nHidden,ENOB))
    plt.savefig(sys.argv[2],dpi=120)

