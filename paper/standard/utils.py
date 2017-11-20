#-- Ayan Chakrabarti <ayan@wustl.edu>

import re
import os
from glob import glob
import numpy as np
import tensorflow as tf


# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
# Init with wildcard e.g. 'wts/*.model.npz'
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])


## Read weights from npz file
def netload(net,fname,sess):
    wts = np.load(fname)
    ph = tf.placeholder(tf.float32)
    for k in wts.keys():
        wvar = net.wts[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(ph),feed_dict={ph: wk})

# Save weights to npz file
def netsave(net,fname,sess):
    wts = {}
    for k in net.wts.keys():
        wts[k] = net.wts[k].eval(sess)
    np.savez(fname,**wts)

# Reset Optimizer state
def resetopt(opt,vdict):
    ops = []
    ops.append(tf.assign(opt._beta1_power,opt._beta1))
    ops.append(tf.assign(opt._beta2_power,opt._beta2))
    for nm in vdict.keys():
        v = vdict[nm]
        ops.append(opt.get_slot(v,'m').initializer)
        ops.append(opt.get_slot(v,'v').initializer)
    return ops
    
    
# Save Optimizer state (Assume Adam)
def saveopt(opt,vdict,others,fname,sess):
    weights = {}

    weights['opt:b1p'] = opt._beta1_power.eval(sess)
    weights['opt:b2p'] = opt._beta2_power.eval(sess)
    for nm in vdict.keys():
        v = vdict[nm]
        weights['opt:m_%s' % nm] = opt.get_slot(v,'m').eval(sess)
        weights['opt:v_%s' % nm] = opt.get_slot(v,'v').eval(sess)

    weights.update(others)
    np.savez(fname,**weights)


# Load Optimizer state (Assume Adam)
def loadopt(opt,vdict,others,fname,sess):
    if not os.path.isfile(fname):
        return None
    weights = np.load(fname)

    ph = tf.placeholder(tf.float32)

    sess.run(opt._beta1_power.assign(ph),
             feed_dict={ph: weights['opt:b1p']})
    sess.run(opt._beta2_power.assign(ph),
             feed_dict={ph: weights['opt:b2p']})

    for nm in vdict.keys():
        v = vdict[nm]
        sess.run(opt.get_slot(v,'m').assign(ph),
                 feed_dict={ph: weights['opt:m_%s' % nm]})
        sess.run(opt.get_slot(v,'v').assign(ph),
                 feed_dict={ph: weights['opt:v_%s' % nm]})

    oval = [weights[k] for k in others]
    return oval
