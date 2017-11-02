#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

import tensorflow as tf
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt

import utils as ut
import circuit as ct
import target as tg

# Copy from train
nBits = 8
nHidden = 64


#########################################################################

# Actual model
data = tg.Target(nBits=nBits)
ckt = ct.Circuit(nHidden=nHidden,nBits=nBits)

gt = tf.placeholder(dtype=tf.float32)
sig = tf.placeholder(dtype=tf.float32)
hpred, acl = ckt.encode(sig)

enc_t = data.Eval(gt,hpred,acl)[0]

#########################################################################
# Start TF session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

sigs,bits = data.uSamp(100000)
bits = np.sum(bits*np.reshape(2**np.arange(nBits),[1,nBits]),1)


mfiles = glob('wts/*.model.npz')
for m in mfiles:
    if os.path.isfile(m+'.png'):
        continue
    ut.netload(ckt,m,sess)
    enc = sess.run(enc_t,feed_dict={sig: sigs})
    plt.Figure()
    plt.hold(True)
    plt.plot(sigs.flatten(),bits.flatten(),'-g',linewidth=2)
    plt.plot(sigs.flatten(),enc.flatten(),'-r',linewidth=2)
    plt.title(m)
    plt.savefig(m+'.png',dpi=120)
    plt.close()
