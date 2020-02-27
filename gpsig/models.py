import numpy as np
import tensorflow as tf

from gpflow import settings, transforms, models, likelihoods, mean_functions
from gpflow.params import Parameter, DataHolder, Minibatch
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow.kullback_leiblers import gauss_kl
from gpflow.conditionals import base_conditional, _expand_independent_outputs
from gpflow.features import Kuu, Kuf

from .inducing_variables import InducingTensors, InducingSequences, Kuu_Kuf_Kff, Kuu, Kuf

class SVGP(models.SVGP):
    """
    Re-implementation of SVGP from GPflow with a few minor tweaks. Slightly more efficient with SignatureKernels, and when using the low-rank option with signature kernels, this code must be used.
    """
    def __init__(self, X, Y, kern, likelihood, feat, mean_function=None, num_latent=None, q_diag=False, whiten=True, minibatch_size=None, num_data=None, q_mu=None, q_sqrt=None, shuffle=True, **kwargs):

        if not isinstance(feat, InducingTensors) and not isinstance(feat, InducingSequences):
            raise ValueError('feat must be of type either InducingTensors or InducingSequences')

        num_inducing = len(feat)

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, shuffle=shuffle, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, shuffle=shuffle, seed=0)

        models.GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = feat
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        
        return
    
    @params_as_tensors
    def _build_likelihood(self):

        X = self.X
        Y = self.Y

        num_samples = tf.shape(X)[0]

        if self.whiten:
            f_mean, f_var = self._build_predict(X, full_cov=False, full_output_cov=False)
            KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0))
        else:
            f_mean, f_var, Kzz = self._build_predict(X, full_cov=False, full_output_cov=False, return_Kzz=True)
            KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0), K=Kzz)
        
        # compute variational expectations
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)

        # scaling for batch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(num_samples, settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):
        
        num_samples = tf.shape(X_new)[0]
        Kzz, Kzx, Kxx = Kuu_Kuf_Kff(self.feature, self.kern, X_new, jitter=settings.jitter, full_f_cov=full_cov)
        f_mean, f_var = base_conditional(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=self.whiten)
        f_mean += self.mean_function(X_new)
        f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        
        if return_Kzz:
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var