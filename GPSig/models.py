import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow.params import Parameter, DataHolder, Minibatch
from gpflow import transforms
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.models import Model, GPModel, GPR, VGP, SVGP
from gpflow.kullback_leiblers import gauss_kl
from gpflow.conditionals import base_conditional, _expand_independent_outputs

from .low_rank_calculations import gauss_kl_feature_simplify_maybe, gauss_conditional_feature_simplify_maybe
from . import low_rank_calculations

import numpy as np
import tensorflow as tf

class SeqGPModel(GPModel):
    def __init__(self, X, Y, kern, likelihood, static_kern=None, mean_function=None, num_latent=None, name=None):
        
        Model.__init__(self, name=name)
        
        if num_latent is None:
            num_latent = Y.shape[1]

        self.num_latent = num_latent
        self.mean_function = mean_function or Zero(output_dim=self.num_latent)
        self.kern = kern
        self.static_kern = static_kern
        self.likelihood = likelihood
        if isinstance(X, np.ndarray):
            X = DataHolder(X)
        if isinstance(Y, np.ndarray):
            Y = DataHolder(Y)
        self.X, self.Y = X, Y

class SeqIDSVGP(SeqGPModel):
    """
    """

    def __init__(self, X, Y, kern, likelihood, Z, minibatch_size=None, static_kern=None, mean_function=None, num_latent=None, white=0.05, q_mu=None, q_sqrt=None, **kwargs):
        """
        """
        if Y.ndim == 1:
            Y = Y[:, None]
        
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)


        SeqGPModel.__init__(self, X, Y, kern, likelihood, static_kern, mean_function, num_latent, **kwargs)
        
        self.num_data = X.shape[0]

        if static_kern is None:
            seq_Z = np.squeeze(Z)
            self.seq_Z = Parameter(seq_Z, dtype=settings.float_type)
            self.static_Z = None
        else:
            assert Z[0].shape[1] == Z[1].shape[0]
            self.seq_Z = Parameter(Z[0], dtype=settings.float_type)
            self.static_Z = Parameter(Z[1], dtype=settings.float_type) 
        self.num_inducing = self.seq_Z.shape[1] # (inducing_length, num_inducing, num_features*(num_lags+1))
        self.white = Parameter(white, trainable=True, transform=transforms.positive, dtype=settings.float_type)
        
        q_mu = np.zeros((self.num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P
        
        q_sqrt = np.array([np.eye(self.num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)]) if q_sqrt is None else q_sqrt
        self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(self.num_inducing, self.num_latent)) # L/P x M x M


    @params_as_tensors
    def _build_likelihood(self):

        X_batch, Y_batch = self.X, self.Y
        num_samples = tf.shape(X_batch)[0]

        fmean, fvar, Kzz = self._build_predict(X_batch, full_cov=False, full_output_cov=False, return_Kzz=True)
        KL =  tf.identity(gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0), Kzz), name='kl_tensor')
        var_exp = tf.identity(self.likelihood.variational_expectations(fmean, fvar, Y_batch), name='varexp_tensor')

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(num_samples, settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):
        
        num_samples = tf.shape(X_new)[0]
        Kzz, Kzx, Kxx = self.kern.Kinter(self.seq_Z, X_new, full_cov=full_cov)
        if self.static_kern is not None:
            Kzz += self.static_kern.K(self.static_Z, presliced=True)
            X_new_static, _ = self.static_kern._slice(X_new, None)
            Kzx += self.static_kern.K(self.static_Z, X_new_static, presliced=True)
            if full_cov:
                Kxx += self.static_kern.K(X_new_static, presliced=True)
            else:
                Kxx += self.static_kern.Kdiag(X_new_static, presliced=True)
            
        Kzz += (self.white + settings.numerics.jitter_level) * tf.eye(self.num_inducing, dtype=settings.float_type)
        if full_cov:
            Kxx += self.white * tf.eye(num_samples, dtype=settings.float_type)
        else:
            Kxx += self.white
        f_mean, f_var = base_conditional(Kzx, Kzz, Kxx, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=False)
        f_mean += self.mean_function(X_new)
        f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        if return_Kzz:
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var

class SeqSVGP(SVGP):
    """
    """

    def __init__(self, X, Y, kern, likelihood, white = 0.05, inducing_length = None, **kwargs):
        """
        """
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.ndim == 3:
            X = tf.squeeze(X)
 
        SVGP.__init__(self, X, Y, kern, likelihood, **kwargs)
        self.white = Parameter(white, trainable=False, transform=transforms.positive, dtype=settings.float_type)


    @params_as_tensors
    def _build_likelihood(self):

        X_batch, Y_batch = self.X, self.Y
        num_inducing = len(self.feature)
        
        X_batch = tf.identity(X_batch, name='X_batch')
        
        num_samples = tf.shape(X_batch)[0]


        fmean, fvar, Kzz = self._build_predict(X_batch, full_cov=False, full_output_cov=False, return_Kzz=True)

        fmean = tf.identity(fmean, name='SVGP_fmean')
        fvar = tf.identity(fvar, name='SVGP_fvar')
        Kzz = tf.identity(Kzz, name='SVGP_Kzz')

        KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0), Kzz)
        KL = tf.identity(KL, name='SVGP_KL')
        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y_batch)
        var_exp = tf.identity(var_exp, name='SVGP_var_exp')

        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(num_samples, settings.float_type)
        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False, full_output_cov=False, return_Kzz=False):
        num_inducing = len(self.feature)
        num_samples = tf.shape(X_new)[0]
        Kzz, Kzx_, Kx_x_ = self.kern.Kinducing(self.feature.Z, X_new, full_cov=full_cov)
        Kzz += (self.white + settings.numerics.jitter_level) * tf.eye(num_inducing, dtype=settings.float_type)
        if full_cov:
            Kx_x_ += self.white * tf.eye(num_samples, dtype=settings.float_type)
        else:
            Kx_x_ += self.white
        
        Kzx_ = tf.identity(Kzx_, name='SVGP_Kzx_')
        f_mean, f_var = base_conditional(Kzx_, Kzz, Kx_x_, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=False)
        f_mean += self.mean_function(X_new)
        f_var = _expand_independent_outputs(f_var, full_cov, full_output_cov)
        if return_Kzz:
            return f_mean, f_var, Kzz
        else:
            return f_mean, f_var

class SeqVGP(VGP):
    """
    VGP model for compatibility with sequentialised kernels and low-rank calculations.
    Includes an additional diagonal noise parameter to make the low-rank approximation well-posed,
    and shift parameters for lagged time-streams.  
    """
    def __init__(self, X, Y, kern, likelihood, mean_function=None, num_latent=None, white=0.2, use_feature_map=False, **kwargs):
        """
        Variational GP for sequentialized kernels with low-rank calculations.
        Input
        :X:     data matrix of covariates as a numpy array of size (N, D)
        :Y:     data matrix of outcomes of size (N, R)
        :kern, likelihood, mean_function: are appropriate GPflow objects
        :num_latent:        number of dimensions in the latent space, should be the same as R
        :white:             initialiser for a diagonal noise added to the kernel
        :use_feature_map:   boolean flag indicating whether to use feature map to simplify calculations when possible.
                            Only works with the sequential kernels that support feature map output, i.e. when kern is only a sequential kernel.
        """
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.ndim == 3:
            X = tf.squeeze(X)
        VGP.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.white = Parameter(white, trainable=False, transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def _build_likelihood(self):
        
        num_samples = tf.shape(self.X)[0]
        K = self.kern.K(self.X) + (settings.numerics.jitter_level + self.white) * tf.eye(num_samples, dtype=settings.float_type)
        KL =  gauss_kl(self.q_mu, tf.matrix_band_part(self.q_sqrt, -1, 0), K)

        fmean = self.q_mu + self.mean_function(self.X)
        fvar = tf.matrix_transpose(tf.reduce_sum(tf.square(tf.matrix_band_part(self.q_sqrt, -1, 0)), axis=-1))
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        return tf.reduce_sum(var_exp) - KL


    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False):
        
        num_samples, num_samples_ = tf.shape(self.X)[0], tf.shape(X_new)[0]
        K = self.kern.K(tf.concat((self.X, X_new), axis=0))
        Kmm, Kmn, Knn = K[:num_samples, :num_samples], K[:num_samples, num_samples:], K[num_samples:, num_samples:]
        Kmm += (self.white + settings.numerics.jitter_level) * tf.eye(num_samples, dtype=settings.float_type)
        Knn += self.white * tf.eye(num_samples_, dtype=settings.float_type)
        if not full_cov:
            Knn = tf.diag_part(Knn)
        f_mean, f_var = base_conditional(Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=tf.matrix_band_part(self.q_sqrt, -1, 0), white=False)

        return f_mean + self.mean_function(X_new), f_var
