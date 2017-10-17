#!/usr/bin/env python
#-- Ayan Chakrabarti <ayan@wustl.edu>

import re
import numpy as np
from matplotlib import pyplot as plt

lines = open('wts/train.log').readlines()

it = []
err1 = []
err2 = []
loss = []

for i in range(len(lines)):
    m = re.search('.*Iteration (\d+).*Err1 = ([^ ,]*).*Err2 = ([^ ,]*).*Loss = ([^ ,]*)',lines[i])
    if m is not None:
        it.append(float(m.groups(0)[0]))
        err1.append(float(m.groups(0)[1]))
        err2.append(float(m.groups(0)[2]))
        loss.append(float(m.groups(0)[3]))

it = np.float32(it)
err1 = np.float32(err1)
err2 = np.float32(err2)
loss = np.float32(loss)

plt.hold(True)
plt.plot(it,err1,'-k',label='Err1')
plt.plot(it,err2,'-b',label='Err2')
plt.plot(it,loss,'-r',label='Loss')
plt.legend()
plt.show()
