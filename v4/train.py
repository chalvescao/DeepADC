#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

import sys
import os
import time
import tensorflow as tf
import numpy as np
import ctrlc

import utils as ut
import circuit as ct
import target as tg

#########################################################################
# Training Parameters
#########################################################################
maxiter=4e4
def getlr(iter):
    if iter < 2e4:
        return 1e-2
    elif iter < 3e4:
        return 3.16e-3
    else:
        return 1e-3

mom = 0.9

bsz = 4096
bgroup = 256

sfreq = 1e3      # How frequently to save
#########################################################################


#########################################################################
# Model / Cicuit Parameters
#########################################################################
nBits = 8
nHidden = 64
#########################################################################



#########################################################################
# Model save & logging setup
ockp = ut.ckpter('wts/iter_*.state.npz') # Optimization state
ockp_s = 'wts/iter_%d.state.npz'

mckp = ut.ckpter('wts/iter_*.model.npz') # Model Files
mckp_s = 'wts/iter_%d.model.npz'

log = open('wts/train.log','a')


def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()
    
#########################################################################
# Actual model
data = tg.Target(nBits=nBits)
ckt = ct.Circuit(nHidden=nHidden,nBits=nBits)

gt = tf.placeholder(dtype=tf.float32)
sig = tf.placeholder(dtype=tf.float32)
hpred, acl = ckt.encode(sig)

_,loss,err1,err2 = data.Eval(gt,hpred,acl)

lr = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(lr,mom)
tStep = opt.minimize(loss+1e-3*tf.reduce_sum(tf.abs(ckt.wts['w1'])))


#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
# Restore model if necessary

gopt = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gopt))
try:
    sess.run(tf.global_variables_initializer())
except:
    sess.run(tf.initialize_all_variables())

if ockp.latest is not None:
    mprint("Loading model")
    ut.netload(ckt,'wts/iter_%d.model.npz'%ockp.iter,sess)
    mprint("Loading state")
    ut.loadopt(opt,ckt.wts,[],'wts/iter_%d.state.npz'%ockp.iter,sess)
    mprint("Done!")

niter = ockp.iter


#########################################################################
# Main training loop

mprint("Starting from Iteration %d" % niter)
while niter < maxiter:

    e1v_a, e2v_a, lv_a = 0., 0., 0.
    for i in range(bgroup):
        sigs,bits = data.rSamp(bsz)
        e1v,e2v,lv,_ = sess.run([err1,err2,loss,tStep],feed_dict={gt: bits, sig: sigs,lr: getlr(niter)})
        e1v_a = e1v_a+e1v
        e2v_a = e2v_a+e2v
        lv_a = lv_a+lv
    e1v_a = e1v_a / bgroup
    e2v_a = e2v_a / bgroup
    lv_a = lv_a / bgroup
    mprint("[%09d] lr = %.2e, Err1 = %.4e, Err2 = %.4e, Loss = %.4e" % (niter, getlr(niter), e1v_a, e2v_a, lv_a))
    niter=niter+1
        
    if ctrlc.stop:
        break

    if sfreq > 0 and niter % sfreq == 0:
        ut.netsave(ckt,mckp_s%niter,sess)
        ut.saveopt(opt,ckt.wts,[],ockp_s%niter,sess)
        ockp.clean(every=sfreq)
        mckp.clean(every=sfreq)
        mprint("Saved model and state.")


mprint("Done!")
ut.netsave(ckt,mckp_s%niter,sess)
ut.saveopt(opt,ckt.wts,[],ockp_s%niter,sess)
ockp.clean(every=sfreq)
mckp.clean(every=sfreq)
mprint("Saved model and state.")
log.close()
