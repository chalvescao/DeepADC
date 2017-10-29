import csv
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline


#########################################################################
#Read the data from the CSV file
#Normalize the x and y data to [0 1]

file_path = 'INV_VTC.csv'

#columnName is name of yAxis in CSV file
def readCSVFile(columnName):
    #with open(file_path,'rb') as csvfile:
    with open(file_path,'r') as csvfile:
      reader = csv.reader(csvfile)
      columnY = [row[ord(columnName)-65] for row in reader][2:-1]
      lastIndex = -1 if '' not in columnY else columnY.index('')
      columnY = columnY[:lastIndex]
      
    #with open(file_path,'rb') as csvfile:
    with open(file_path,'r') as csvfile:
      reader = csv.reader(csvfile)
      columnX = [row[0] for row in reader][2:-1]   
      columnX = columnX[:lastIndex] 
    return [columnX, columnY]

#Normalize Data
def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x));
    return x;


#Normalize CSV x and y
#Return the x and y after normalization
def normalizeXY(columnName):
    x = np.float32(readCSVFile(columnName)[0])
    x = MaxMinNormalization(x)
    
    y = np.float32(readCSVFile(columnName)[1])
    y = MaxMinNormalization(y)    
    
    return [x,y];


#Calculate Derivative for each xPoint
def CalculateDerivative(x,y,xPoint):
    # Get a function that evaluates the linear spline at any x
    f = InterpolatedUnivariateSpline(x, y, k=1)

    return f.derivative()(xPoint);

#########################################################################
#Define the circuit activation function class
class customActivation:
    def __init__(self,columnName = 'M'):
      
        #piecewise linear interpolation            
        self.circuitActivationArray = {}
        self.circuitActivationArray['x'] = np.linspace(0, 1, 1000)
        self.circuitActivationArray['y'] = np.interp(self.circuitActivationArray['x'], 
                                             normalizeXY(columnName)[0], 
                                             normalizeXY(columnName)[1]) 
         
    #plot circuit activation function and piecewise linear interpolation
    def plotChart(self,columnName):  
          
        #Calculate the normalized x and y
        x = normalizeXY(columnName)[0]
        y = normalizeXY(columnName)[1]
        
        #piecewise linear interpolation
        xd = np.linspace(0, 1, 1000) 
        yinterp = np.interp(xd, x, y)
        
        plt.hold(True)
        plt.plot(x,y,'-k',label='circuitActivation')
        plt.plot(xd,yinterp,'-r',label='piecewise linear interpolation')
        plt.legend()
        plt.show() 
        
    # Define a circuit activation function
    # Then convert it to numpy arrays
    def circuitActivation(self,x):

        if x < 0:
            return self.circuitActivationArray['y'][0]
        elif x > 1:
            return self.circuitActivationArray['y'][-1]
        else:
            return np.interp(x, 
                             self.circuitActivationArray['x'], 
                             self.circuitActivationArray['y'])
    
    
    # Define gradient of circuit activation function
    # Then convert it to numpy arrays
    def d_circuitActivation(self,x):
        
        return CalculateDerivative(self.circuitActivationArray['x'],
                                   self.circuitActivationArray['y'],
                                   x)

    
    # define gradients of a function using tf.RegisterGradient and tf.Graph.gradient_override_map
    def py_func(self,func, inp, Tout, stateful=True, name=None, grad=None):
    
        # Need to generate a unique name to avoid duplicates
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
        tf.RegisterGradient(rnd_name)(grad)  
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
    
 
    # Making a numpy function to a tensorflow funtion
    def tf_d_circuitActivation(self,x,name=None):    
        
        np_d_circuitActivation = np.vectorize(self.d_circuitActivation)      
        np_d_circuitActivation_32 = lambda x: np_d_circuitActivation(x).astype(np.float32)
        
        with ops.name_scope(name, "d_circuitActivation",[x]) as name:
            y = tf.py_func(np_d_circuitActivation_32,
                            [x],
                            [tf.float32],
                            name=name,
                            stateful=False)
            return y[0]
      
    # The activation function has only one input      
    def circuitActivationGrad(self,op, grad):
      
        x = op.inputs[0]    
        n_gr = self.tf_d_circuitActivation(x)
        return grad * n_gr 

    
    # Finally we get a custom activation function accept by tensorflow
    def tf_circuitActivation(self, x, name=None):
      
        np_circuitActivation = np.vectorize(self.circuitActivation)
        np_circuitActivation_32 = lambda x: np_circuitActivation(x).astype(np.float32)   
           
        with ops.name_scope(name, "circuitActivation",[x]) as name:
            y = self.py_func(np_circuitActivation_32,
                            [x],
                            [tf.float32],
                            name=name,
                            grad=self.circuitActivationGrad)  
            return y[0]

#########################################################################
#Test the activation function
# with tf.Session() as sess:
#      ca = customActivation(columnName = 'M')
#      x = tf.constant([-0.2,0.3,1.7,0.7])
#      y = ca.tf_circuitActivation(x)
#      tf.global_variables_initializer().run()
#      print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())
#      ca.plotChart('M')

    

            
