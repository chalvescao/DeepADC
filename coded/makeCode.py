#-- Ayan Chakrabarti <ayan@wustl.edu>

import numpy as np
import sys

if len(sys.argv) != 3:
    sys.exit('Call as: python makeCode.py bin_bits gray_bits')

levels = 2**int(sys.argv[1])
bits = int(sys.argv[2])

lchg = np.ones(bits)-1

def tostr(b):
    return ' '.join([str(int(i)) for i in b])


cur = np.zeros(bits)

bits = [tostr(cur)]

for id in range(1,levels):
    idx = np.argsort(lchg)
    for k in range(len(idx)):
        nb = cur.copy()
        nb[idx[k]] = 1-nb[idx[k]]
        st = tostr(nb)
        if st not in bits:
            lchg[idx[k]] = id
            cur = nb
            bits.append(st)
            break
    if len(bits) != id+1:
        sys.exit('Could not find gray code, need more gray bits.')

for id in range(len(bits)):    
    print("%4d: %s" % (id,bits[id]))
