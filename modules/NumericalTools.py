import tensorflow as tf
from keras import backend as K
import math
from numpy import sqrt,arange
from scipy.special import erf

def rangeTransform(x,a,b,inverse=False):

    """transform to a limited range as Minuit does (cf. https://root.cern.ch/download/minuit.pdf Sec. 1.2.1)"""

    return tf.asin(2*(x-a)/(b-a)-1.0) if inverse else a+0.5*(b-a)*(tf.sin(x)+1.0)


def gdLikelihood(y_true,mu,sigma,a1,a2,gausConstraints={}):

    """ implements the gaussian-double exponential likelihood """

    #reduce the variable
    t = (y_true - mu)/sigma
    
    #normalization term
    Norm = 0
    try:
        Norm = sqrt(math.pi/2)*sigma*(tf.erf(a2/sqrt(2)) - tf.erf(-a1/sqrt(2))) 
    except:
        Norm = sqrt(math.pi/2)*sigma*(erf(a2/sqrt(2)) - erf(-a1/sqrt(2))) 
    Norm += K.exp(-0.5*K.pow(a1,2))*sigma/a1 
    Norm += K.exp(-0.5*K.pow(a2,2))*sigma/a2

    #add gaussian constraints
    aux_gaus=0
    if 'mu' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['mu'][1]
        aux_gaus +=  0.5*K.pow((a1-gausConstraints['mu'][0])/gausConstraints['mu'][1],2)
    if 'sigma' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['sigma'][1]
        aux_gaus +=  0.5*K.pow((a1-gausConstraints['sigma'][0])/gausConstraints['sigma'][1],2)
    if 'a1' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['a1'][1]
        aux_gaus +=  0.5*K.pow((a1-gausConstraints['a1'][0])/gausConstraints['a1'][1],2)
    if 'a2' in gausConstraints:
        Norm     +=  sqrt(0.5/math.pi)/gausConstraints['a2'][1]
        aux_gaus +=  0.5*K.pow((a1-gausConstraints['a2'][0])/gausConstraints['a2'][1],2)

        
    #make sure the normalization is not 0
    Norm  = tf.clip_by_value(Norm,1e-5,9e12)
    
    #negative log-likelihood
    nll = tf.where(K.greater_equal(t, a2),
                   K.log(Norm) -0.5*K.pow(a2, 2) + a2*t + aux_gaus,
                   tf.where(K.greater_equal(t, -a1),
                            K.log(Norm) + 0.5*K.pow(t,2) + aux_gaus,
                            K.log(Norm) -0.5*K.pow(a1, 2) - a1*t + aux_gaus
                            )
                   )
    
    return nll

def gdOffsetLikelihood(y_true,e2,sigma_e2,a1_e2,a2_e2,n_e2,gausConstraints={}):

    """ implements the gaussian-double exponential+offset likelihood """    

    #reduced variables
    t = (y_true - e2)/sigma_e2
    t1 = (math.pi + e2)/sigma_e2
    t2 = (math.pi - e2)/sigma_e2

    n1 = (sigma_e2/a1_e2)*K.exp(0.5*tf.pow(a1_e2,2))*(K.exp(-tf.pow(a1_e2,2)) - K.exp(-a1_e2*t1))
    n2 = (sigma_e2/a2_e2)*K.exp(0.5*tf.pow(a2_e2,2))*(K.exp(-tf.pow(a2_e2,2)) - K.exp(-a2_e2*t2))

    N = tf.where(tf.logical_and(tf.greater_equal(a1_e2, t1), tf.greater_equal(a2_e2, t2)),
                 sqrt(math.pi/2)*sigma_e2*(tf.erf(t2/sqrt(2)) - tf.erf(-t1/sqrt(2))),
                 tf.where(tf.logical_and(tf.greater(t1, a1_e2), tf.greater_equal(a2_e2, t2)),
                          sqrt(math.pi/2)*sigma_e2*(tf.erf(t2/sqrt(2)) - tf.erf(-a1_e2/sqrt(2))) + n1,
                          tf.where(tf.logical_and(tf.greater_equal(a1_e2, t1), tf.greater(t2, a2_e2)),
                                   sqrt(math.pi/2)*sigma_e2*(tf.erf(a2_e2/sqrt(2)) - tf.erf(-t1/sqrt(2))) + n2,
                                   sqrt(math.pi/2)*sigma_e2*(tf.erf(a2_e2/sqrt(2)) - tf.erf(-a1_e2/sqrt(2))) + n1 + n2
                                   )
                          )
                 )
        
    f = tf.where(tf.greater_equal(t, a2_e2),
                 K.exp(0.5*tf.pow(a2_e2, 2) - a2_e2*t),
                 tf.where(tf.greater_equal(t, -a1_e2),
                          K.exp(-0.5*tf.pow(t,2)),
                          K.exp(0.5*tf.pow(a1_e2, 2) + a1_e2*t)
                          )
                 )
    N = tf.clip_by_value(N,1e-5,9e12)

    nll = -K.log(n_e2 + f*(1-2*math.pi*n_e2)/N)
    nll = tf.where(tf.is_nan(nll), 500*tf.ones_like(nll), nll)
    nll = tf.where(tf.is_inf(nll), 500*tf.ones_like(nll), nll)

    return nll
