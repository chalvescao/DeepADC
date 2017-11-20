#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

import sys
import os
import time
import tensorflow as tf
import numpy as np
import ctrlc

import utils as ut
import circuitn as ct
import graytg as tg

#########################################################################
# Training Parameters
#########################################################################
BASE_LR = 1e-3
SNAPITER = 1
RESETITER = 5e2
maxiter=2e4
def getlr(iter):
    if iter < 5e3:
        return BASE_LR
    elif iter < 1e4:
        return 0.316*BASE_LR
    else:
        return 0.1*BASE_LR

WD=1e-4
mom = 0.9

bsz = 4096
bgroup = 256

sfreq = 1e3      # How frequently to save
#########################################################################


#########################################################################
# Get params from command line
#########################################################################
if len(sys.argv) != 6:
    sys.exit("USAGE: train.py nBits1 nBits2 nHidden nzstd wtsdir")
nBits1 = int(sys.argv[1])
nBits2 = int(sys.argv[2])
nHidden = int(sys.argv[3])
nzstd = float(sys.argv[4])
wtdir = sys.argv[5]
#########################################################################



#########################################################################
# Model save & logging setup
ockp = ut.ckpter(wtdir+'/iter_*.state.npz') # Optimization state
ockp_s = wtdir+'/iter_%d.state.npz'

mckp = ut.ckpter(wtdir+'/iter_*.model.npz') # Model Files
mckp_s = wtdir+'/iter_%d.model.npz'

log = open(wtdir+'/train.log','a')


def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()
    
#########################################################################
# Actual model
data = tg.Target(nBits1=nBits1,nBits2=nBits2)
ckt = ct.Circuit(nHidden=nHidden,nBits=nBits2,nzstd=nzstd)

gt = tf.placeholder(dtype=tf.float32)
sig = tf.placeholder(dtype=tf.float32)
hpred, acl = ckt.encode(sig)

loss,err1 = data.Eval(gt,hpred,acl)

lr = tf.placeholder(dtype=tf.float32)
opt = tf.train.AdamOptimizer(lr,mom)
tStep = opt.minimize(loss+ WD * ( \
                            tf.nn.l2_loss(ckt.wts['w0']) ))
rStep = ut.resetopt(opt,ckt.wts)

#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
# Restore model if necessary

#gopt = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
gopt = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gopt))
try:
    sess.run(tf.global_variables_initializer())
except:
    sess.run(tf.initialize_all_variables())

if ockp.latest is not None:
    mprint("Loading model")
    ut.netload(ckt,wtdir+'/iter_%d.model.npz'%ockp.iter,sess)
    mprint("Loading state")
    ut.loadopt(opt,ckt.wts,[],wtdir+'/iter_%d.state.npz'%ockp.iter,sess)
    mprint("Done!")

niter = ockp.iter


#########################################################################
# Main training loop

mprint("Starting from Iteration %d" % niter)
while niter < maxiter:

    e1v_a, lv_a = 0., 0.
    for i in range(bgroup):
        sigs,bits = data.rSamp(bsz)
        e1v,lv,_ = sess.run([err1,loss,tStep],feed_dict={gt: bits, sig: sigs,lr: getlr(niter)})
        
        e1v_a = e1v_a+e1v
        lv_a = lv_a+lv
    e1v_a = e1v_a / bgroup
    lv_a = lv_a / bgroup
    mprint("[%09d] lr = %.2e, Err1 = %.4e, Loss = %.4e" % (niter, getlr(niter), e1v_a, lv_a))
    niter=niter+1
    if niter % RESETITER == 0:
        sess.run(rStep)
    if niter % SNAPITER == 0:
        sess.run(ckt.cOp)
        
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
