import numpy as np
import tensorflow as tf
import csvact as ca
import matplotlib.pyplot as plt    

XMIN = ca.XMIN
XMAX = ca.XMAX


vtc = ca.VTC()
x = tf.placeholder(dtype=tf.float32)
y = vtc.act(x)

s = tf.Session()

xx = np.linspace(XMIN-0.1,XMAX+0.1,1e4).reshape((100,100))
yy = s.run(y,feed_dict={x:xx})

plt.hold(True)
plt.plot(xx,yy,'-r',linewidth=10)
plt.plot(vtc.GT[:,0],vtc.GT[:,1],'-g',linewidth=5)
plt.show()
