import tensorflow as tf
from keras import backend as K
import math
from numpy import sqrt,arange
from scipy.special import erf

from NumericalTools import *

def gd_loss_generator(addConstraints=False,fixedTails=False):

    """generator for a gd_loss function with or without constraints"""

    def aux_fun(y_true, y_pred):

        y_true = tf.unstack(y_true, axis=1, num=1)[0]

        mu    = rangeTransform( tf.unstack(y_pred, axis=1)[0], -3,   3)
        sigma = rangeTransform( tf.unstack(y_pred, axis=1)[1], 1e-3, 5)
        a1=1.6
        a2=1.6
        if not fixedTails:
            a1    = rangeTransform( tf.unstack(y_pred, axis=1)[2], 1e-3, 5)
            a2    = rangeTransform( tf.unstack(y_pred, axis=1)[3], 1e-3, 5)

        gausConstraints={}
        if addConstraints:
            gausConstraints['mu']    = (0.8,1.0)
            gausConstraints['sigma'] = (0.8,1.0)
            if not fixedTails:
                gausConstraints['a1']    = (1.6,1.0)
                gausConstraints['a2']    = (1.6,1.0)
                
        aux=gdLikelihood(y_true,mu,sigma,a1,a2,gausConstraints)
        return K.sum(aux)

    name='gd_loss_gauss' if addConstraints else 'gd_loss'
    name += '_fixedtails' if fixedTails else ''
    aux_fun.__name__ = name
    return aux_fun


def gd_offset_loss_generator(addConstraints=False):

    """customized loss for a gauss+double expo tails + offset """

    def aux_fun(y_true, y_pred):
        y_true   = tf.unstack(y_true, axis=1, num=1)[0]
        
        e2       = rangeTransform( tf.unstack(y_pred, axis=1)[0],  -3,   3)
        sigma_e2 = rangeTransform( tf.unstack(y_pred, axis=1)[1],  1e-3, 5)
        a1_e2    = rangeTransform( tf.unstack(y_pred, axis=1)[2],  1e-3, 5)
        a2_e2    = rangeTransform( tf.unstack(y_pred, axis=1)[3],  1e-3, 5)
        n_e2     = rangeTransform( tf.unstack(y_pred, axis=1)[4],  0,    1)

        gausConstraints={}
        if addConstraints:
            print 'Not yet implemented for gd_offset'

        aux=gdOffsetLikelihood(y_true,e2,sigma_e2,a1_e2,a2_e2,n_e2,gausConstraints)
        return K.sum(aux)

    name='gd_offset_loss_gauss' if addConstraints else 'gd_offset_loss'
    aux_fun.__name__ = name
    return aux_fun


def logcosh(y_true, y_pred):
    """logcosh losse should be implemented in newer Keras versions"""
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)

def quantile_loss_generator(q):
    """generator for a q-quantile loss"""
    def aux_fun(y_true, y_pred):
        e = (y_true - y_pred)
        aux = tf.where(tf.greater_equal(e, 0.), q*e, (q-1)*e)
        return K.mean(aux)
    qkey='%3.2f'%q
    aux_fun.__name__ = 'quantile_loss_'+qkey.replace(".", "")
    return aux_fun


def huber(y_true,y_pred,delta=1.0):
    """Huber loss"""
    z = K.abs(y_true[:,0] - y_pred[:,0])
    mask = K.cast(K.less(z,delta),K.floatx())
    return K.mean( 0.5*mask*K.square(z) + (1.-mask)*(delta*z - 0.5*delta**2) )

def ahuber(y_true,y_pred,dm=0.5,dp=1.0):
    """asymmetric Huber loss"""
    z = y_true[:,0] - y_pred[:,0]
    aux = tf.where( tf.greater_equal(z,dp),
                    dp*z-0.5*dp**2,
                    tf.where( tf.greater_equal(z,-dm),
                              0.5*K.square(z),
                              -dm*z-0.5*dm**2) 
                )
    return K.mean(aux)

def ahuber_q(y_true,y_pred,dm=0.8,dp=1.2,qm=0.16,qp=0.84):
    """asymmetric Huber loss with quantiles"""

    mu_pred  = tf.unstack(y_pred, axis=1)[0]
    e_mu   = (y_true-mu_pred)
    aux_mu = tf.where( tf.greater_equal(e_mu,dp),
                       dp*e_mu-0.5*dp**2,
                       tf.where( tf.greater_equal(e_mu,-dm),
                                 0.5*K.square(e_mu),
                                 -dm*e_mu-0.5*dm**2) )
    aux_mu = K.mean(aux_mu)
                
    qm_pred  = tf.unstack(y_pred, axis=1)[1]
    em = (y_true-qm_pred)
    aux_m = tf.where(tf.greater_equal(em, 0.), qm*em, (qm-1)*em)
    aux_m = K.mean(aux_m)

    qp_pred  = tf.unstack(y_pred, axis=1)[2]
    ep = (y_true-qp_pred)
    aux_p = tf.where(tf.greater_equal(ep, 0.), qp*ep, (qp-1)*ep)
    aux_p = K.mean(aux_p)

    return aux_m+aux_p+aux_mu
    

# list of all the losses
global_loss_list={'gd_loss'                  : gd_loss_generator(addConstraints=False,fixedTails=False),
                  'gd_loss_gauss'            : gd_loss_generator(addConstraints=True, fixedTails=False),
                  'gd_loss_gauss_fixedtails' : gd_loss_generator(addConstraints=True, fixedTails=True),
                  'gd_offset_loss'           : gd_offset_loss_generator(addConstraints=False),
                  'gd_offset_loss_gauss'     : gd_offset_loss_generator(addConstraints=True),
                  'logcosh'                  : logcosh,
                  'huber'                    : huber,
                  'ahuber'                   : ahuber,
                  'ahuber_q'                 : ahuber_q}

for q in arange(0.05,1.0,0.05):
    global_loss_list['%3.2f'%q]=quantile_loss_generator(q)
