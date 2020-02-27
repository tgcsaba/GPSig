import numpy as np
import tensorflow as tf
import gpflow

from gpflow import settings, transforms
from gpflow.features import InducingPointsBase
from gpflow.dispatch import dispatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.params import Parameter

from .kernels import SignatureKernel

class GradedInducingVariables(InducingPointsBase):
    def __init__(self, Z, num_levels, learn_weights=False):
        
        super().__init__(Z)
        
        self.learn_weights = learn_weights

        if learn_weights:
            self.W = Parameter(np.tile(np.eye(self.__len__())[None, ...], [num_levels, 1, 1]), dtype=settings.float_type)


class InducingTensors(GradedInducingVariables):
    def __init__(self, Z, num_levels, recursive_tensors=False, increments=False, **kwargs):

        if recursive_tensors: raise NotImplementedError('not yet implemented recursive_tensors')
        
        if recursive_tensors:
            len_tensors = num_levels
        else:
            len_tensors = int(num_levels*(num_levels+1)/2)

        assert Z.shape[0] == len_tensors

        if increments:
            assert Z.ndim == 4
            assert Z.shape[2] == 2

        super().__init__(Z, num_levels, **kwargs)

        self.len_tensors = len_tensors
        self.recursive_tensors = recursive_tensors
        self.increments = increments

    def __len__(self):
        return self.Z.shape[1]

@dispatch(InducingTensors, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False):
    with params_as_tensors_for(feat):
        if feat.learn_weights:
            Kzz, Kzx, Kxx = kern.K_tens_n_stream_covs(feat.Z, X_new, full_X_cov=full_f_cov, return_levels=True, increments=feat.increments)
            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
            Kxx = tf.reduce_sum(Kxx, axis=0)
        else:
            Kzz, Kzx, Kxx = kern.K_tens_n_stream_covs(feat.Z, X_new, full_X_cov=full_f_cov, increments=feat.increments)
        
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
            Kzx = kern.K_tens_vs_stream(feat.Z, X_new, return_levels=True, increments=feat.increments)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
        else:
            Kzx = kern.K_tens_vs_stream(feat.Z, X_new, increments=feat.increments)
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

class InducingStreams(GradedInducingVariables):
    def __init__(self, Z, num_levels, first_coord_is_time=False, **kwargs):
        
        if first_coord_is_time:
            times = Z[..., 0]
            Z = Z[..., 1:]
            super().__init__(Z, num_levels, **kwargs)
            self.starting_times = Parameter(times[:, 0], dtype=settings.float_type)
            self.delta_times = Parameter(times[:, 1:] - times[:, :-1], transform=transforms.positive, dtype=settings.float_type)
        else:
            super().__init__(Z, num_levels, **kwargs)

        self.first_coord_is_time = first_coord_is_time
        self.len_streams = Z.shape[1]

    @params_as_tensors
    def _build_inducing_streams(self):
        if self.first_coord_is_time:
            times = tf.cumsum(tf.concat((self.starting_times[:, None], self.delta_times), axis=1), axis=1)
            Z = tf.concat((times[..., None], self.Z), axis=2)
            return Z
        else:
            return self.Z

    @property
    def _inducing_streams(self):
        return self._build_inducing_streams()

    @property
    @autoflow()
    def inducing_streams(self):
        return self._inducing_streams


@dispatch(InducingStreams, SignatureKernel, object)
def Kuu_Kuf_Kff(feat, kern, X_new, *, jitter=0.0, full_f_cov=False):
    with params_as_tensors_for(feat):
        Z = feat._inducing_streams
        if feat.learn_weights:
            Kzz, Kzx, Kxx = kern.K_stream_n_stream_covs(Z, X_new, full_X2_cov=full_f_cov, return_levels=True)

            Kzz = Kzz[0] + tf.reduce_sum(tf.matmul(tf.matmul(feat.W, Kzz[1:]), feat.W, transpose_b=True), axis=0)
            Kzx = Kzx[0] + tf.reduce_sum(tf.matmul(feat.W, Kzx[1:]), axis=0)
            Kxx = tf.reduce_sum(Kxx, axis=0)
        else:
            Kzz, Kzx, Kxx = kern.K_stream_n_stream_covs(Z, X_new, full_X2_cov=full_f_cov)

        Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
        if full_f_cov:
            Kxx += jitter * tf.eye(tf.shape(X)[0], dtype=settings.dtypes.float_type)
        else:
            Kxx += jitter
    return Kzz, Kzx, Kxx