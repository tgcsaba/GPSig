import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings, transforms
from gpflow.features import InducingPointsBase, InducingFeature, Kuu, Kuf
from gpflow.dispatch import dispatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.params import Parameter, ParamList
from gpflow.kernels import Kernel, Combination, Sum, Product

from .kernels import SignatureKernel

class SignatureInducing(InducingPointsBase):
    """
    Base class for inducing variables for use with signature kernel in sparse variational GPs.
    # Input
    :Z:             Array of locations of inducing variables
    :num_levels:    The same as the num_levels argument in SignatureKernel
    :learn_weights: True or False, if True, an additional linear combination layer is added to the inducing variables, separately on each tensor algebra level 
    """
    def __init__(self, Z, num_levels, learn_weights=False):
        super().__init__(Z)
        self.learn_weights = learn_weights
        if learn_weights:
            self.W = Parameter(np.tile(np.eye(self.__len__())[None, ...], [num_levels, 1, 1]), dtype=settings.float_type)

class InducingTensors(SignatureInducing):
    """
    Inducing class for using sparse tensors as inducing variable locations.
    # Input
    :Z:             np data array of initial locations of size ((num_levels+1)*num_levels/2, num_tensors, num_features*(num_lags+1)) if not increments
                    else ((num_levels+1)*num_levels/2, num_tensors, 2, num_features*(num_lags+1))
    :num_levels:    The same as the num_levels argument in the SignatureKernel
    :increments:    Along each axis in the tensor product that specifies the sparse tensors,
                    a difference of two elements of 2 reproducing kernels is used (instead of just a reproducing kernel)
    """
    def __init__(self, Z, num_levels, increments=False, **kwargs):
        len_tensors = int(num_levels*(num_levels+1)/2)
        assert Z.shape[0] == len_tensors
        if increments:
            assert Z.ndim == 4
            assert Z.shape[2] == 2
        super().__init__(Z, num_levels, **kwargs)
        self.len_tensors = len_tensors
        self.increments = increments

    def __len__(self):
        return self.Z.shape[1]

@dispatch(InducingTensors, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzz, Kzx, Kxx = kern.K_tens_n_seq_covs(feat.Z, X_new, full_X_cov=full_f_cov, return_levels=True, increments=feat.increments)
            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
            Kxx = tf.reduce_sum(Kxx, axis=0)
        else:
            Kzz, Kzx, Kxx = kern.K_tens_n_seq_covs(feat.Z, X_new, full_X_cov=full_f_cov, increments=feat.increments)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
        if full_f_cov:
            Kxx += jitter * tf.eye(tf.shape(X)[0], dtype=settings.dtypes.float_type)
        else:
            Kxx += jitter
    return Kzz, Kzx, Kxx

@dispatch(InducingTensors, SignatureKernel, object)
def Kuf(feat, kern, X_new):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzx = kern.K_tens_vs_seq(feat.Z, X_new, return_levels=True, increments=feat.increments)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
        else:
            Kzx = kern.K_tens_vs_seq(feat.Z, X_new, increments=feat.increments)
    return Kzx

@dispatch(InducingTensors, SignatureKernel)
def Kuu(feat, kern, *, jitter=0.0, full_f_cov=False):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzz = kern.K_tens(feat.Z, return_levels=True, increments=feat.increments)
            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
        else:
            Kzz = kern.K_tens(feat.Z, increments=feat.increments)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
    return Kzz

class InducingSequences(SignatureInducing):
    """
    Inducing class for using sequences as inducing variable locations.
    # Input
    :Z:             np data array of initial sequenes of size (num_inducing, len_inducing, num_features)
    :num_levels:    The same as the num_levels argument in the SignatureKernel
    """
    def __init__(self, Z, num_levels, **kwargs):
        super().__init__(Z, num_levels, **kwargs)
        self.len_inducing = Z.shape[1]
        

@dispatch(InducingSequences, SignatureKernel)
def Kuu(feat, kern, *, jitter=0.0):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzz = kern.K(feat.Z, return_levels=True, presliced=True)
            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
        else:
            Kzz = kern.K(feat.Z, presliced=True)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
    return Kzz

@dispatch(InducingSequences, SignatureKernel, object)
def Kuf(feat, kern, X_new):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzx = kern.K(feat.Z, X_new, presliced_X=True, return_levels=True)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
        else:
            Kzx = kern.K(feat.Z, X_new, presliced_X=True)
    return Kzx

@dispatch(InducingSequences, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzz, Kzx, Kxx = kern.K_seq_n_seq_covs(feat.Z, X_new, full_X2_cov=full_f_cov, return_levels=True)
            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
            Kxx = tf.reduce_sum(Kxx, axis=0)
        else:
            Kzz, Kzx, Kxx = kern.K_seq_n_seq_covs(feat.Z, X_new, full_X2_cov=full_f_cov)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
        if full_f_cov:
            Kxx += jitter * tf.eye(tf.shape(X)[0], dtype=settings.dtypes.float_type)
        else:
            Kxx += jitter
    return Kzz, Kzx, Kxx