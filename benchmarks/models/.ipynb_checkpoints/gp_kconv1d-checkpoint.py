from functools import lru_cache

import numpy as np
import tensorflow as tf

from gpflow.decors import autoflow, params_as_tensors, params_as_tensors_for
from gpflow.dispatch import dispatch
from gpflow.features import InducingPointsBase
from gpflow.kernels import Kernel
from gpflow.models import SVGP
from gpflow import settings


class Conv1D(Kernel):

    def __init__(self, base_kern, len_streams, len_windows, num_features):
        Kernel.__init__(self, len_streams*num_features)
        self.len_streams = len_streams
        self.len_windows = len_windows
        self.base_kern = base_kern
        self.num_features = num_features

        if self.base_kern.input_dim != len_windows*num_features:
            raise ValueError("Base_kern input dimensions must be consistent with window length.")

    @lru_cache()
    @params_as_tensors
    def get_windows(self, X):
        """
        Extracts windows from a batch of time-series
        :param X: (num_streams, len_streams, num_features)
        :return: windows (num_streams, num_windows, len_windows*num_features)
        """
        X = tf.reshape(X, (tf.shape(X)[0], -1, 1, self.num_features)) # expand last-1th dim since tf extract image patches only supports "2D images", while a sequence is 1D
        Xw = tf.extract_image_patches(X, (1, self.len_windows, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), "VALID")
        return tf.reshape(Xw, (tf.shape(X)[0], -1, self.len_windows*self.num_features))

    def _filter_nans_and_reduce_mean_along_axes(self, M, axes):
        """
        :M:     a tensor
        :axes:  a list or tuple of integers meaning along which axes to compute the mean, only counting the non-nan elements (e.g. np.nanmean)
        """
        
        mask_not_nans = tf.logical_not(tf.is_nan(M))
        
        M = tf.where(mask_not_nans, M, tf.zeros(tf.shape(M), dtype=settings.float_type))
        
        K = tf.reduce_sum(M, axes) / tf.reduce_sum(tf.cast(mask_not_nans, settings.float_type), axes)

        return K

    def _replace_nans_and_get_windows_mask(self, Xw):
        """
        """
        mask_nans_w = tf.is_nan(Xw)
        Xw = tf.where(mask_nans_w, tf.zeros(tf.shape(Xw), dtype=settings.float_type), Xw)
        mask_nans_w = tf.reduce_any(mask_nans_w, axis=2)
        return Xw, mask_nans_w

    @params_as_tensors
    def K(self, X, X2=None):
        """
        """
        num_streams = tf.shape(X)[0]        
        Xw = self.get_windows(X)  # (num_streams, num_windows, len_windows*num_features)
        num_windows = tf.shape(Xw)[1]
        Xw, mask_nans_w = self._replace_nans_and_get_windows_mask(Xw)
        Xw = tf.reshape(Xw, (-1, self.base_kern.input_dim))
        
        if X2 is None:
            mask_nans_w2 = mask_nans_w
            M = tf.reshape(self.base_kern.K(Xw), (num_streams, num_windows, num_streams, num_windows))
        else:
            num_streams2 = tf.shape(X2)[0]
            X2w = self.get_windows(X2)  # (num_streams2, num_windows2, len_windows*num_features)
            num_windows2 = tf.shape(X2w)[1]
            
            X2w, mask_nans_w2 = self._replace_nans_and_get_windows_mask(X2w)
            X2w = tf.reshape(X2w, (-1, self.base_kern.input_dim))
            M = tf.reshape(self.base_kern.K(Xw, X2w), (num_streams, num_windows, num_streams2, num_windows2))

        mask_nans_M = tf.logical_or(mask_nans_w[:, :, None, None], mask_nans_w2[None, None, :, :])
        M = tf.where(mask_nans_M, tf.zeros(tf.shape(M), dtype=settings.float_type), M)
        K = tf.reduce_sum(M, axis=(1, 3)) / tf.reduce_sum(tf.cast(tf.logical_not(mask_nans_M), settings.float_type), axis=(1, 3))
        
        K = K + settings.jitter * tf.eye((num_streams), dtype=settings.float_type) if X2 is None else K
        return K

#     @params_as_tensors
#     def K(self, X, X2=None):
#         """
#         """
#         num_streams = tf.shape(X)[0]        
#         Xw = self.get_windows(X)  # (num_streams, num_windows, len_windows*num_features)
#         num_windows = tf.shape(Xw)[1]
#         Xw = tf.reshape(Xw, (-1, self.base_kern.input_dim))
        
#         if X2 is None:
#             M = tf.reshape(self.base_kern.K(Xw), (num_streams, num_windows, num_streams, num_windows))
#         else:
#             num_streams2 = tf.shape(X2)[0]
#             X2w = self.get_windows(X2)  # (num_streams2, num_windows2, len_windows*num_features)
#             X2w = tf.reshape(X2w, (-1, self.base_kern.input_dim))
#             M = tf.reshape(self.base_kern.K(Xw, X2w), (num_streams, num_windows, num_streams2, num_windows))

#         K = self._filter_nans_and_reduce_mean_along_axes(M, axes=(1,3))
#         return K

    @params_as_tensors
    def Kdiag(self, X):
        """
        """
        num_streams = tf.shape(X)[0]
        Xw = self.get_windows(X)
        num_windows = tf.shape(Xw)[1]
        Xw, mask_nans_w = self._replace_nans_and_get_windows_mask(Xw)

        M = self.base_kern.K(Xw)
        mask_nans_M = tf.logical_or(mask_nans_w[:, :, None], mask_nans_w[:, None, :])
        M = tf.where(mask_nans_M, tf.zeros(tf.shape(M), dtype=settings.float_type), M)
        K = tf.reduce_sum(M, axis=(1, 2)) / tf.reduce_sum(tf.cast(tf.logical_not(mask_nans_M), settings.float_type), axis=(1, 2))
        return K + settings.jitter

#     @params_as_tensors
#     def Kdiag(self, X):
#         """
#         """
#         num_streams = tf.shape(X)[0]
#         Xw = self.get_windows(X)
#         num_windows = tf.shape(Xw)[1]
#         Xw = tf.reshape(Xw, (num_streams, num_windows, -1))  # (num_streams, num_windows, len_windows*num_features)
        
#         M = self.base_kern.K(Xw)
#         K = self._filter_nans_and_reduce_mean_along_axes(M, axes=(1,2))
#         return K

    @property
    def num_windows(self):
        return (self.len_streams - self.len_windows + 1)

    @autoflow((settings.float_type,))
    def compute_windows(self, X):
        return self.get_windows(X)

class InducingWindows(InducingPointsBase):
    """
    Analogous to inducing patches, but for 1D conv kernel
    """
    pass

@dispatch(InducingWindows, Conv1D, object)
def Kuf(feat, kern, Xnew):
    """
    """
    with params_as_tensors_for(feat, kern):
#         num_streams = tf.shape(Xnew)[0]
#         Xw = kern.get_windows(Xnew)  # (num_streams, num_windows, len_windows * num_features)
#         Xw, mask_nans_w = kern._replace_nans_and_get_windows_mask(Xw)
#         Xw = tf.reshape(Xw, (num_streams * kern.num_windows, -1))
#         Xw = tf.reshape(Xw, (-1, kern.base_kern.input_dim))
        
#         M = tf.reshape(kern.base_kern.K(tf.reshape(feat.Z, (-1, kern.base_kern.input_dim)), Xw), (len(feat), num_streams, kern.num_windows))  # (num_inducing, num_streams, num_windows)
        
#         mask_nans_M = tf.logical_or(tf.zeros((tf.shape(feat.Z)[0], 1, 1), dtype=bool), mask_nans_w[None, :, :])
        
#         M = tf.where(mask_nans_M,  tf.zeros(tf.shape(M), dtype=settings.float_type), M)
#         K = tf.reduce_sum(M, axis=(2)) / tf.reduce_sum(tf.cast(mask_nans_M, settings.float_type), axis=(2))
#         K = kern._filter_nans_and_reduce_mean_along_axes(M, axes=(2))
        K = kern.K(tf.reshape(feat.Z, (len(feat), -1)), Xnew)
    return K


@dispatch(InducingWindows, Conv1D)
def Kuu(feat, kern, jitter=0.0):
    """
    """
    with params_as_tensors_for(feat, kern):
        return kern.K(tf.reshape(feat.Z, (len(feat), -1)))
#         return kern.base_kern.K(tf.reshape(feat.Z, (len(feat), -1))) + (jitter + settings.jitter) * tf.eye(len(feat), dtype=settings.float_type)

class SVGP(SVGP):
    """
    re-define SVGP so that it pulls the new dispatch functions
    """
    # def __init__(self, X, Y, kern, likelihood, **kwargs):
    #     SVGP.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
    pass    