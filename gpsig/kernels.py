
import tensorflow as tf
import numpy as np

from gpflow import utilities
from gpflow import config
from gpflow.base import Parameter
from gpflow.kernels import Kernel

import tensorflow_probability as tfp

from . import lags
from . import low_rank_calculations
from . import signature_algs

class SignatureKernel(Kernel):
    """
    """
    def __init__(self, num_features, len_examples, num_levels=3, variances=1., base_variance=None, lengthscales=1., ARD=True, order=1, normalization=False, difference=True, num_lags=None,
                lags=None, lags_delta=0.1, lagscales=None, low_rank=False, rank_bound=50, sparsity='sqrt', fixed_low_rank_params=True, **kwargs):
        """
        # Inputs:
        ## Args:
        :num_features:      the state-space dimension of the input sequebces,
        :len_examples:      sequence length
        :num_levels:        the degree of cut-off for the truncated signatures
                            (i.e. len_examples = len(active_dims) / num_features)
        ## Kwargs:
        ### Kernel options
        :active_dims:       if specified, should contain a list of the dimensions in the input that should be sliced out and fed to the kernel.
        :variances:         multiplicative scaling applied to the Signature kernel,
                            if ARD is True, there is one parameter for each level, i.e. variances is of size (num_levels + 1)                
		:lengthscales:      lengthscales for scaling the coordinates of the input sequences,
                            if lengthscales is None, no scaling is applied to the paths
                            if ARD is True, there is one lengthscale for each path dimension, i.e. lengthscales is of size (num_features)
        :ARD:               boolean indicating whether the ARD option is used, if None inferred from variances and lengthscales param
        :order:             order of the signature kernel minimum is 1 and maximum is num_levels (set to -1 to set to max)
        :normalization:     False - no normalization, True - normalize signature levels
        :difference:        boolean indicating whether to difference paths
                            False corresponds to the signature of the integrated path
        :num_lags:          Nonnegative integer or None, the number of lags added to each sequence. Usually between 0-5.
        
        ### Low-rank options:
        :low_rank:          boolean indicating whether to use low-rank kernel
        :rank_bound:        size of low-rank factors
        :sparsity:          controls the sparsity of the random projection matrix used in the low-rank algorithm
                            possible values are:
                            - 'sqrt' - approximately O(n * sqrt(n)) non-zero entries;
                            - 'log' - approximately O(n * log(n)) non-zero entries;
                            - 'lin' - approximately O(n) non-zero entries;
        """
        
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.len_examples = len_examples
        self._validate_active_dims(num_features, len_examples)

        self.num_levels = num_levels
        self.order = num_levels if (order <= 0 or order >= num_levels) else order

        if self.order != 1 and low_rank: 
            raise NotImplementedError('Higher-order algorithms not yet compatible with low-rank mode.')

        self.normalization = normalization
        self.difference = difference

        self.ARD = ARD
        self.variances = Parameter(self._validate_signature_param("variances", variances, num_levels + 1, ARD=ARD), transform=utilities.positive(), dtype=config.default_float())
        self.lengthscales = Parameter(self._validate_signature_param("lengthscales", lengthscales, self.num_features, ARD=ARD), transform=utilities.positive(), dtype=config.default_float())

        if not self.ARD and not self.normalization and base_variance is not None:
            self.base_variance = Parameter(base_variance, transform=utilities.positive(), dtype=config.default_float())
        else:
            self.base_variance = None

        if self.ARD:
            self.global_variance = Parameter(1., transform=utilities.positive(), dtype=config.default_float())

        self.num_lags, self.lags, self.lags_delta, self.lagscales = self._validate_lag_args(num_lags, lags, lags_delta, lagscales, ARD=ARD)
        self.lags = Parameter(self.lags, transform=tfp.bijectors.Sigmoid(), dtype=config.default_float()) if self.lags is not None else None
        self.lagscales = Parameter(self.lagscales, transform=utilities.positive(), dtype=config.default_float()) if self.lagscales is not None else None

        self.low_rank, self.rank_bound, self.sparsity, self.fixed_low_rank_params = self._validate_low_rank_args(low_rank, rank_bound, sparsity, fixed_low_rank_params)
        if self.low_rank and self.fixed_low_rank_params:
            self.is_low_rank_fitted = False
            # self.nystrom_samples = Parameter(np.zeros((self.rank_bound, self.num_features * (self.num_lags+1)), dtype=config.default_float()), trainable=False, dtype=config.default_float())
            # self.projection_seeds = Parameter(np.zeros((self.num_levels-1, 2), dtype=config.default_int()), trainable=False, dtype=config.default_int())

	######################
	## Input validators ##
	######################
	
    def _validate_active_dims(self, num_features, len_examples):
        """
        Validates the format of the input samples.
        """

        if self.active_dims is None or isinstance(self.active_dims, slice):
            # Can only validate parameter if active_dims is an array
            return

        if num_features * len_examples != len(self.active_dims):
            raise ValueError("Size of 'active_dims' {self.active_dims} does not match 'num_features*len_examples' {num_features*len_examples}.")
        return len_examples

    def _validate_signature_param(self, name, value, size, ARD=True):
        """
        Validates signature params
        """

        value = value * np.ones(size, dtype=config.default_float()) if ARD else value

        correct_shape = (size,) if ARD else () 
        
        if np.asarray(value).shape != correct_shape:
            raise ValueError("shape of {} does not match the expected shape {}".format(name, correct_shape))
        return value
    
    def _validate_lag_args(self, num_lags, lags, lags_delta, lagscales, ARD=True):
        """
        Validates the lags parameters
        """
        num_lags = num_lags or 0
        
        if not isinstance(num_lags, int) or num_lags < 0:
            raise ValueError('The variable num_lags most be a nonnegative integer or None.')
        else:
            if num_lags > 0:
                if lags is None:
                    if not isinstance(lags_delta, float) or lags_delta < 0. or lags_delta > 1./float(num_lags):
                        raise ValueError('If lags is not specified, then lags_delta most be a float between 0 and 1/num_lags')
                    lags = lags_delta * np.asarray(range(1, num_lags+1))
                else:
                    lags = self._validate_signature_param('lags', lags, num_lags, ARD=True)
                    if not np.all(np.logical_and(lags > 0., lags < 1.)):
                        raise ValueError('If lags is specified, its value should be in the range (0, 1)')
                
                if ARD:
                    lagscales = lagscales or 1./float(num_lags+1)
                    lagscales = self._validate_signature_param('lagscales', lagscales, num_lags+1, ARD=ARD)
                else:
                    lagscales = None
            else:
                lags, lags_delta, lagscales = None, None, None
        return num_lags, lags, lags_delta, lagscales
        


    def _validate_low_rank_args(self, low_rank, rank_bound, sparsity, fixed_low_rank_params):
        """
        Validates the low-rank params
        """
        if low_rank is not None and low_rank == True:
            if not type(low_rank) == bool:
                raise ValueError("Unknown low-rank argument: %s. It should be True of False." % low_rank)
            if sparsity not in ['log', 'sqrt', 'lin']:
                raise ValueError("Unknown sparsity argument %s. Possible values are 'sqrt', 'log', 'lin'" % sparsity)
            if rank_bound is not None and rank_bound <= 0:
                raise ValueError("The rank-bound in the low-rank algorithm must be either None or a positiv integer.")
        else:
            low_rank = False
            rank_bound, sparsity, fixed_low_rank_params = None, None, None
        return low_rank, rank_bound, sparsity, fixed_low_rank_params

    def fit_low_rank_params(self, X=None, nys_samples=None):

        if not self.low_rank:
            raise RuntimeError('Method fit_low_rank_params only accessible when low_rank option is active')
        
        if X is None and nys_samples is None:
            raise RuntimeError('At least one of \'X\' and \'nys_samples\' should be specified.')

        self.projection_seeds = tf.random.uniform((self.num_levels-1, 2), maxval=np.iinfo(np.int32).max, dtype=np.int32)

        if nys_samples is None:
            X = tf.reshape(X, [-1, self.num_features])
            num_samples = tf.shape(X)[0]
            # logits = tf.math.log(1./tf.cast(num_samples, config.default_float()) * tf.ones((num_samples), dtype=config.default_float()))
            # idx = tf.random.categorical(logits[None], self.rank_bound)[0]
            idx = tf.random.shuffle(tf.range(num_samples, dtype=config.default_int()))[:self.rank_bound]
            nys_samples = tf.gather(X, idx, axis=0)
            
        self.nystrom_samples = tf.reshape(tf.tile(nys_samples[:, None, :], [1, self.num_lags+1, 1]), [-1, self.num_features*(self.num_lags+1)])
        self.is_low_rank_fitted = True
        
        return


    def _K_seq_diag(self, X):
        """
        # Input
        :X:             (num_examples, len_examples, num_features) tensor of sequences
        # Output
        :K:             (num_levels+1, num_examples) tensor of (unnormalized) diagonals of signature kernel
        """
        
        len_examples = tf.shape(X)[-2]                    
        
        M = self._base_kern(X)
        
        if self.order == 1:
            K_lvls_diag = signature_algs.signature_kern_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls_diag = signature_algs.signature_kern_higher_order(M, self.num_levels, order=self.order, difference=self.difference)
        
        return K_lvls_diag

    
    def _K_seq(self, X, X2 = None):
        """
        # Input
        :X:             (num_examples, len_examples, num_features) tensor of  sequences 
        :X2:            (num_examples2, len_examples2, num_features) tensor of sequences
        # Output
        :K:             (num_levels+1, num_examples, num_examples2) tensor of (unnormalized) signature kernel matrices     
        """
        
        num_examples, len_examples, num_features = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        num_samples = num_examples * len_examples

        if X2 is not None:
            num_examples2, len_examples2 = tf.shape(X2)[0], tf.shape(X2)[1]
            num_samples2 = num_examples2 * len_examples2

        if X2 is None:
            X = tf.reshape(X, [num_samples, num_features])
            M = tf.reshape(self._base_kern(X), [num_examples, len_examples, num_examples, len_examples])        
        else:
            X = tf.reshape(X, [num_samples, num_features])
            X2 = tf.reshape(X2, [num_samples2, num_features])
            M = tf.reshape(self._base_kern(X, X2), [num_examples, len_examples, num_examples2, len_examples2])    
        
        if self.order == 1:
            K_lvls = signature_algs.signature_kern_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls = signature_algs.signature_kern_higher_order(M, self.num_levels, order=self.order, difference=self.difference)

        return K_lvls

    def _K_seq_lr_feat(self, X, nys_samples=None, seeds=None):
        """
        # Input
        :X:                 (num_examples, len_examples, num_features) tensor of sequences 
        :nys_samples:       (num_samples, num_features) tensor of samples to use in Nystrom approximation
        :seeds:             (num_levels-1, 2) array of ints for seeding randomized projection matrices
        # Output
        :Phi_lvls:          a (num_levels+1,) list of low-rank factors for each signature level     
        """
        
        num_examples, len_examples, num_features = tf.shape(X)[-3], tf.shape(X)[-2], tf.shape(X)[-1]
        num_samples = num_examples * len_examples

        X = tf.reshape(X, [num_samples, num_features])
        X_feat = low_rank_calculations.Nystrom_map(X, self._base_kern, nys_samples, self.rank_bound)
        X_feat = tf.reshape(X_feat, [num_examples, len_examples, self.rank_bound])
        
        if self.order == 1:
            Phi_lvls = signature_algs.signature_kern_first_order_lr_feature(X_feat, self.num_levels, self.rank_bound, self.sparsity, seeds, difference=self.difference)
        else:
            raise NotImplementedError('Low-rank mode not implemented for order higher than 1.')
        
        return Phi_lvls

    def _K_tens_diag(self, Z, increments=False):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) array of inducing tensor components, if not increments 
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        # Output
        :K_lvls:        (num_levels+1,) list of (num_tensors) kernel matrices (for each T.A. level)
        """

        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]

        if increments:
            Z = tf.reshape(Z, [len_tensors * num_tensors, 2, num_features])
            M = tf.reshape(self._base_kern(Z), [len_tensors, num_tensors, 2, 2])
            M = M[:, :, 1, 1] + M[:, :, 0, 0] - M[:, :, 1, 0] - M[:, :, 0, 1]
        else:
            Z = tf.reshape(Z, [len_tensors * num_tensors, 1, num_features])
            M = tf.reshape(self._base_kern(Z), [len_tensors, num_tensors])
        
        K_lvls = signature_algs.tensor_kern(M, self.num_levels)

        return K_lvls
                
    def _K_tens(self, Z, increments=False):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments 
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        # Output
        :K_lvls:        (num_levels+1,) list of (num_tensors, num_tensors) kernel matrices (for each T.A. level)
        """

        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]

        if increments:
            Z = tf.reshape(Z, [len_tensors, 2 * num_tensors, num_features])
            M = tf.reshape(self._base_kern(Z), [len_tensors, num_tensors, 2, num_tensors, 2])
            M = M[:, :, 1, :, 1] + M[:, :, 0, :, 0] - M[:, :, 1, :, 0] - M[:, :, 0, :, 1]
        else:
            M = self._base_kern(Z)
        
        K_lvls = signature_algs.tensor_kern(M, self.num_levels)

        return K_lvls

    def _K_tens_lr_feat(self, Z, increments=False, nys_samples=None, seeds=None):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments 
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        :nys_samples:   (num_samples, num_features) tensor of samples to use in Nystrom approximation
        :seeds:         (num_levels-1, 2) array of ints for seeding randomized projection matrices
        # Output
        :Phi_lvls:      a (num_levels+1,) list of low-rank factors for cov matrices of inducing tensors on each TA level     
        """

        if self.order > 1: raise NotImplementedError('higher order not implemented yet for low-rank mode')

        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]

        if increments:
            Z = tf.reshape(Z, [num_tensors * len_tensors * 2, num_features])
            Z_feat = low_rank_calculations.Nystrom_map(Z, self._base_kern, nys_samples, self.rank_bound)
            Z_feat = tf.reshape(Z_feat, [len_tensors, num_tensors, 2, self.rank_bound])
            Z_feat = Z_feat[:, :, 1, :] - Z_feat[:, :, 0, :]
        else:
            Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])
            Z_feat = low_rank_calculations.Nystrom_map(Z, self._base_kern, nys_samples, self.rank_bound)
            Z_feat = tf.reshape(Z_feat, [len_tensors, num_tensors, self.rank_bound])
        
        Phi_lvls = signature_algs.tensor_kern_lr_feature(Z_feat, self.num_levels, self.rank_bound, self.sparsity, seeds)
        return Phi_lvls

    def _K_tens_vs_seq(self, Z, X, increments=False):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments 
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        :X:             (num_examples, len_examples, num_features) tensor of sequences 
        Output
        :K_lvls:        (num_levels+1,) list of inducing tensors vs input sequences covariance matrices on each T.A. level
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        num_examples, len_examples = tf.shape(X)[-3], tf.shape(X)[-2]

        X = tf.reshape(X, [num_examples * len_examples, num_features])
        if increments:
            Z = tf.reshape(Z, [2 * num_tensors * len_tensors, num_features])
            M = tf.reshape(self._base_kern(Z, X), (len_tensors, num_tensors, 2, num_examples, len_examples))
            M = M[:, :, 1] - M[:, :, 0]
        else:
            Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])
            M = tf.reshape(self._base_kern(Z, X), (len_tensors, num_tensors, num_examples, len_examples))

        if self.order == 1:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_higher_order(M, self.num_levels, order=self.order, difference=self.difference)
        
        return K_lvls

    
    def _apply_scaling_to_samples(self, X):
        """
        Applies scaling to given samples.
        """
        
        num_samples, _ = tf.unstack(tf.shape(X))
        
        num_features = self.num_features * (self.num_lags + 1)

        X = tf.reshape(X, (num_samples, self.num_lags+1, self.num_features))
        
        if self.lengthscales is not None:
            X /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales

        if self.num_lags > 0 and self.lagscales is not None:
            X *= self.lagscales[None, :, None]
        
        X = tf.reshape(X, (num_samples, num_features))
        return X

    
    def _apply_scaling_and_lags_to_sequences(self, X):
        """
        Applies scaling and lags to sequences.
        """
        
        num_examples, len_examples, _ = tf.unstack(tf.shape(X))
        
        num_features = self.num_features * (self.num_lags + 1)
        
        if self.num_lags > 0:
            X = lags.add_lags_to_sequences(X, self.lags)

        X = tf.reshape(X, (num_examples, len_examples, self.num_lags+1, self.num_features))
        
        if self.lengthscales is not None:
            X /= self.lengthscales[None, None, None, :] if self.ARD else self.lengthscales

        if self.num_lags > 0 and self.lagscales is not None:
            X *= self.lagscales[None, None, :, None]
        
        X = tf.reshape(X, (num_examples, len_examples, num_features))
        return X

    
    def _apply_scaling_to_tensors(self, Z):
        """
        Applies scaling to simple tensors of shape (num_levels*(num_levels+1)/2, num_tensors, num_features*(num_lags+1))
        """
        
        len_tensors, num_tensors = tf.shape(Z)[0], tf.shape(Z)[1]
        
        if self.lengthscales is not None:
            Z = tf.reshape(Z, (len_tensors, num_tensors, self.num_lags+1, self.num_features))
            Z /= self.lengthscales[None, None, None, :] if self.ARD else self.lengthscales
        
        if self.num_lags > 0 and self.lagscales is not None:
            Z *= self.lagscales[None, None, :, None]

        Z = tf.reshape(Z, (len_tensors, num_tensors, -1)) 
        return Z
    
    
    def _apply_scaling_to_incremental_tensors(self, Z):
        """
        Applies scaling to incremental tensors of shape (num_levels*(num_levels+1)/2, num_tensors, 2, num_features*(num_lags+1))
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        
        if self.lengthscales is not None:
            Z = tf.reshape(Z, (len_tensors, num_tensors, 2, self.num_lags+1, self.num_features))
            Z /= self.lengthscales[None, None, None, None, :] if self.ARD else self.lengthscales
        
        if self.num_lags > 0 and self.lagscales is not None:
            Z *= self.lagscales[None, None, None, :, None]

        Z = tf.reshape(Z, (len_tensors, num_tensors, 2, num_features))
        return Z
            
    
    def K(self, X, X2 = None):
        """
        Computes signature kernel between sequences
        """

        if self.low_rank and not self.is_low_rank_fitted:
            raise RuntimeError('The fit_low_rank_params method must be called before using the kernel.')

        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, [num_examples, -1, self.num_features])
        len_examples = tf.shape(X)[1]

        X = self._apply_scaling_and_lags_to_sequences(X)

        if X2 is None:
            if self.low_rank:
                if self.fixed_low_rank_params:
                    nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
                    Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=self.projection_seeds)
                else:
                    Phi_lvls = self._K_seq_lr_feat(X)

                K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
            else:
                K_lvls = self._K_seq(X)
            
            if self.normalization:
                K_lvls += config.default_jitter()
                K_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(K_lvls))
                K_lvls /= K_lvls_diag_sqrt[:, :, None] * K_lvls_diag_sqrt[:, None, :]

        else:
            num_examples2 = tf.shape(X2)[0]
            X2 = tf.reshape(X2, [num_examples2, -1, self.num_features])
            len_examples2 = tf.shape(X2)[1]
            
            X2 = self._apply_scaling_and_lags_to_sequences(X2)

            if self.low_rank:
                if self.fixed_low_rank_params:
                    seeds = self.projection_seeds
                    nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
                else:
                    seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(config.default_int()).max, dtype=config.default_int())
                    idx, _ = low_rank_calculations._draw_indices(num_examples*len_examples + num_examples2*len_examples2, self.rank_bound)
                    nys_samples = tf.gather(tf.concat((tf.reshape(X, [num_examples*len_examples, -1]), tf.reshape(X2, [num_examples2*len_examples2, -1])), axis=0), idx, axis=0)
                
                Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)
                Phi2_lvls = self._K_seq_lr_feat(X2, nys_samples=nys_samples, seeds=seeds)

                K_lvls = tf.stack([tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
            else:
                K_lvls = self._K_seq(X, X2)

            if self.normalization:
                if self.low_rank:
                    K1_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
                    K2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
                else:
                    K1_lvls_diag = self._K_seq_diag(X)
                    K2_lvls_diag = self._K_seq_diag(X2)

                K_lvls += config.default_jitter()
                K1_lvls_diag += config.default_jitter()
                K2_lvls_diag += config.default_jitter()

                K1_lvls_diag_sqrt = tf.sqrt(K1_lvls_diag)
                K2_lvls_diag_sqrt = tf.sqrt(K2_lvls_diag)
                
                K_lvls /= K1_lvls_diag_sqrt[:, :, None] * K2_lvls_diag_sqrt[:, None, :]
        
        K_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
        
        return tf.reduce_sum(K_lvls, axis=0)

    
    def K_diag(self, X):
        """
        Computes the diagonals of a square signature kernel matrix.
        """

        if self.low_rank and not self.is_low_rank_fitted:
            raise RuntimeError('The fit_low_rank_params method must be called before using the kernel.')

        num_examples = tf.shape(X)[0]
        
        if self.normalization:
            return tf.fill((num_examples,), self.global_variance * tf.reduce_sum(self.variances)) if self.ARD else tf.fill((num_examples,), self.variances)

        X = tf.reshape(X, (num_examples, -1, self.num_features))

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            if self.fixed_low_rank_params:
                nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
                Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=self.projection_seeds)
            else:
                Phi_lvls = self._K_seq_lr_feat(X)

            K_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
        else:
            K_lvls_diag = self._K_seq_diag(X)

        K_lvls_diag *= self.global_variance * self.variances[:, None] if self.ARD else self.variances         

        return tf.reduce_sum(K_lvls_diag, axis=0)

    def feat(self, X):
        """
        Computes feature map for low-rank signature kernel
        """

        if not self.low_rank:
            raise RuntimeError('The feat method is only available when low-rank option is active..')
        elif not self.is_low_rank_fitted:
            raise RuntimeError('The fit_low_rank_params method must be called before using the kernel.')

        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, [num_examples, -1, self.num_features])
        len_examples = tf.shape(X)[1]

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.fixed_low_rank_params:
            nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
            Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=self.projection_seeds)
        else:
            Phi_lvls = self._K_seq_lr_feat(X)

        if self.normalization:
            jitter_sqrt = tf.cast(tf.sqrt(config.default_jitter()), config.default_float())
            Phi_lvls = [tf.concat((P,  jitter_sqrt * tf.ones((num_examples, 1), dtype=config.default_float())), axis=-1) for P in Phi_lvls]
            Phi_lvls = [P / tf.linalg.norm(P, axis=-1)[..., None] for P in Phi_lvls]
        
        Phi_lvls = [tf.sqrt(self.global_variance * self.variances[i]) * P for i, P in enumerate(Phi_lvls)] if self.ARD else [tf.sqrt(self.variances) * P for P in Phi_lvls]
        
        Phi = tf.concat(Phi_lvls, axis=-1)
        
        return Phi
        
    
    def K_tens(self, Z, increments=False):
        """
        Computes a square covariance matrix of inducing tensors Z.
        """

        if self.low_rank and not self.is_low_rank_fitted:
            raise RuntimeError('The fit_low_rank_params method must be called before using the kernel.')
        
        num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        if self.low_rank:
            if self.fixed_low_rank_params:
                nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
                Phi_lvls = self._K_tens_lr_feat(Z, increments=increments, nys_samples=nys_samples, seeds=self.projection_seeds)
            else:
                Phi_lvls = self._K_tens_lr_feat(Z, increments=increments)

            K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
        else:
            K_lvls = self._K_tens(Z, increments=increments)

        if self.normalization:
            K_lvls += config.default_jitter()
            K_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(K_lvls))
            K_lvls /= K_lvls_diag_sqrt[:, :, None] * K_lvls_diag_sqrt[:, None, :] 

        K_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
        
        return tf.reduce_sum(K_lvls, axis=0)

    
    def K_tens_vs_seq(self, Z, X, increments=False):
        """
        Computes a cross-covariance matrix between inducing tensors and sequences.
        """

        if self.low_rank and not self.is_low_rank_fitted:
            raise RuntimeError('The fit_low_rank_params method must be called before using the kernel.')
        # if self.low_rank and self.fixed_low_rank_params:
        #     tf.assert_equal(self.is_low_rank_fitted, 1)
        
        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, (num_examples, -1, self.num_features))
        len_examples = tf.shape(X)[1]
        
        num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            if self.fixed_low_rank_params:
                seeds = self.projection_seeds
                nys_samples = self._apply_scaling_to_samples(self.nystrom_samples)
            else:
                seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(config.default_int()).max, dtype=config.default_int())
                idx, _ = low_rank_calculations._draw_indices(num_tensors*len_tensors*(int(increments)+1) + num_examples*len_examples, self.rank_bound)
                nys_samples = tf.gather(tf.concat((tf.reshape(Z, [num_tensors*len_tensors*(int(increments)+1), -1]), tf.reshape(X, [num_examples*len_examples, -1])), axis=0), idx, axis=0)
            
            Phi_Z_lvls = self._K_tens_lr_feat(Z, increments=increments, nys_samples=nys_samples, seeds=seeds)
            Phi_X_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)

            Kzx_lvls = tf.stack([tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0) 
        else:
            Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

        if self.normalization:
            if self.low_rank:
                Kxx_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0)
                Kzz_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_Z_lvls], axis=0)
            else:
                Kxx_lvls_diag = self._K_seq_diag(X)
                Kzz_lvls_diag = self._K_tens_diag(Z, increments=increments)

            Kzx_lvls += config.default_jitter()
            Kzz_lvls_diag += config.default_jitter()
            Kxx_lvls_diag += config.default_jitter()

            Kzz_lvls_diag_sqrt = tf.sqrt(Kzz_lvls_diag)
            Kxx_lvls_diag_sqrt = tf.sqrt(Kxx_lvls_diag)
            Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :] * Kzz_lvls_diag_sqrt[:, :, None]

        Kzx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances

        return tf.reduce_sum(Kzx_lvls, axis=0)

    # 
    # def K_tens_n_seq_covs(self, Z, X, full_X_cov = False, return_levels=False, increments=False, presliced=False):
    #     """
    #     Computes and returns all three relevant matrices between inducing tensors tensors and input sequences, Kzz, Kzx, Kxx. Kxx is only diagonal if not full_X_cov
    #     """

    #     if not presliced:
    #         X, _ = self._slice(X, None)
        
    #     num_examples = tf.shape(X)[0]
    #     X = tf.reshape(X, (num_examples, -1, self.num_features))
    #     len_examples = tf.shape(X)[1]
        
    #     num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]

    #     if increments:
    #         Z = self._apply_scaling_to_incremental_tensors(Z)
    #     else:
    #         Z = self._apply_scaling_to_tensors(Z)

    #     X = self._apply_scaling_and_lags_to_sequences(X)

    #     if self.low_rank:
    #         seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(config.default_int()).max, dtype=config.default_int())
    #         idx, _ = low_rank_calculations._draw_indices(num_tensors*len_tensors*(int(increments)+1) + num_examples*len_examples, self.rank_bound)
    #         nys_samples = tf.gather(tf.concat((tf.reshape(Z, [num_tensors*len_tensors*(int(increments)+1), -1]), tf.reshape(X, [num_examples*len_examples, -1])), axis=0), idx, axis=0)
            
    #         Phi_Z_lvls = self._K_tens_lr_feat(Z, increments=increments, nys_samples=nys_samples, seeds=seeds)
    #         Phi_X_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)

    #         Kzz_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_Z_lvls], axis=0)
    #         Kzx_lvls = tf.stack([tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
    #     else:
    #         Kzz_lvls = self._K_tens(Z, increments=increments)
    #         Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

    #     if full_X_cov:
    #         if self.low_rank:
    #             Kxx_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_X_lvls], axis=0)
    #         else:
    #             Kxx_lvls = self._K_seq(X)

    #         if self.normalization:
    #             Kxx_lvls += config.default_jitter() * tf.eye(num_examples, dtype=config.default_float())[None]
                
    #             Kxx_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(Kxx_lvls))

    #             Kxx_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kxx_lvls_diag_sqrt[:, None, :]
    #             Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]
            
    #         Kxx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kzz_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kzx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances

    #         if return_levels:
    #             return Kzz_lvls, Kzx_lvls, Kxx_lvls
    #         else:
    #             return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls, axis=0)

    #     else:
    #         if self.low_rank:
    #             Kxx_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0)
    #         else:
    #             Kxx_lvls_diag = self._K_seq_diag(X)

    #         if self.normalization:
    #             Kxx_lvls_diag += config.default_jitter()

    #             Kxx_lvls_diag_sqrt = tf.sqrt(Kxx_lvls_diag)

    #             Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]
    #             Kxx_lvls_diag = tf.tile(self.global_variance * self.variances[:, None], [1, num_examples]) if self.ARD else tf.fill((num_levels, num_examples), self.variances)
    #         else:
    #             Kxx_lvls_diag *= self.global_variance * self.variances[:, None] if self.ARD else self.variances
            
    #         Kzz_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kzx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
            
    #         if return_levels:
    #             return Kzz_lvls, Kzx_lvls, Kxx_lvls_diag
    #         else:
    #             return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls_diag, axis=0)

    # 
    # def K_seq_n_seq_covs(self, X, X2, full_X2_cov = False, return_levels=False, presliced=False):
    #     """
    #     Computes and returns all three relevant matrices between inducing sequences and input sequences, Kxx, Kxx2, Kx2x2. Kx2x2 is only diagonal if not full_X2_cov
    #     """

    #     if not presliced:
    #         X2, _ = self._slice(X2, None)

    #     num_examples = tf.shape(X)[0]
    #     X = tf.reshape(X, (num_examples, -1, self.num_features))
    #     len_examples = tf.shape(X)[1]

    #     num_examples2 = tf.shape(X2)[0]
    #     X2 = tf.reshape(X2, (num_examples2, -1, self.num_features))
    #     len_examples2 = tf.shape(X2)[1]

    #     X = self._apply_scaling_and_lags_to_sequences(X)
    #     X2 = self._apply_scaling_and_lags_to_sequences(X2)

    #     if self.low_rank:
    #         seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(config.default_int()).max, dtype=config.default_int())
    #         idx, _ = low_rank_calculations._draw_indices(num_examples*len_examples + num_examples2*len_examples2, self.rank_bound)
    #         nys_samples = tf.gather(tf.concat((tf.reshape(X, [num_examples*len_examples, -1]), tf.reshape(X2, [num_examples2*len_examples2, -1])), axis=0), idx, axis=0)
            
    #         Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)
    #         Phi2_lvls = self._K_seq_lr_feat(X2, nys_samples=nys_samples, seeds=seeds)

    #         Kxx_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
    #         Kxx2_lvls = tf.stack([tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
    #     else:
    #         Kxx_lvls = self._K_seq(X)
    #         Kxx2_lvls = self._K_seq(X, X2)

    #     if self.normalization:
            
    #         Kxx_lvls += config.default_jitter() * tf.eye(num_examples, dtype=config.default_float())[None]

    #         Kxx_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(Kxx_lvls))
    #         Kxx_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kxx_lvls_diag_sqrt[:, None, :]
    #         Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None]
        
    #     if full_X2_cov:
    #         if self.low_rank:
    #             Kx2x2_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi2_lvls], axis=0)
    #         else:
    #             Kx2x2_lvls = self._K_seq(X2)

    #         if self.normalization:

    #             K_x2x2_lvls += config.default_jitter() * tf.eye(num_examples2, dtype=config.default_float())[None]

    #             Kx2x2_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(K_x2x2_lvls)) 

    #             Kxx2_lvls /= Kx2x2_lvls_diags_sqrt[:, None, :]
    #             Kx2x2_lvls /= Kx2x2_lvls_diags_sqrt[:, :, None] * Kx2x2_lvls_diags_sqrt[:, None, :]
            
    #         Kxx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kxx2_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kx2x2_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances

    #         if return_levels:
    #             return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls
    #         else:
    #             return tf.reduce_sum(Kxx_lvls, axis=0), tf.reduce_sum(Kxx2_lvls, axis=0), tf.reduce_sum(Kx2x2_lvls, axis=0)

    #     else:
    #         if self.low_rank:
    #             Kx2x2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
    #         else:
    #             Kx2x2_lvls_diag = self._K_seq_diag(X2)

    #         if self.normalization:
    #             Kx2x2_lvls_diag += config.default_jitter()

    #             Kx2x2_lvls_diag_sqrt = tf.sqrt(Kx2x2_lvls_diag)
                
    #             Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kx2x2_lvls_diag_sqrt[:, None, :]
    #             Kx2x2_lvls_diag = tf.tile(self.global_variance * self.variances[:, None], [1, num_examples2]) if self.ARD else tf.fill((num_levels, num_examples2), self.variances)
    #         else:
    #             Kx2x2_lvls_diag *= self.global_variance * self.variances[:, None] if self.ARD else self.variances

    #         Kxx_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
    #         Kxx2_lvls *= self.global_variance * self.variances[:, None, None] if self.ARD else self.variances
        
    #         if return_levels:
    #             return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls_diag
    #         else:
    #             return tf.reduce_sum(Kxx_lvls, axis=0), tf.reduce_sum(Kxx2_lvls, axis=0), tf.reduce_sum(Kx2x2_lvls_diag, axis=0)

    ##### Helper functions for base kernels

    def _square_dist(self, X, X2=None):
        batch = tf.shape(X)[:-2]
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, tf.concat((batch, [-1, 1]), axis=0))  + tf.reshape(Xs, tf.concat((batch, [1, -1]), axis=0))
            return dist

        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, tf.concat((batch, [-1, 1]), axis=0)) + tf.reshape(X2s, tf.concat((batch, [1, -1]), axis=0))
        return dist


    def _euclid_dist(self, X, X2 = None):
        r2 = self._square_dist(X, X2)
        return tf.sqrt(tf.maximum(r2, 1e-40))
    
    ##### Base kernel implementations:
    # To-do: use GPflow kernels for vector valued-data as base kernels 

class SignatureLinear(SignatureKernel):
    """
    The signature kernel, which uses the identity as state-space embedding 
    """

    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        # self.gamma = Parameter(1.0/float(self.num_features), transform=utilities.positive(), dtype=config.default_float())
        # self.offsets = Parameter(np.zeros(self.num_features), dtype=config.default_float())
        self._base_kern = self._lin
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    def _lin(self, X, X2=None):
        if X2 is None:
            # K = self.gamma * tf.matmul(X, X, transpose_b = True)
            K = tf.matmul(X, X, transpose_b = True)
            return  K
        else:
            # return self.gamma * tf.matmul(X, X2, transpose_b = True)
            return tf.matmul(X, X2, transpose_b = True)

class SignatureCosine(SignatureKernel):
    """
    The signature kernel, which uses the cos-similarity as state-space embedding
    """

    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        # self.gamma = Parameter(1.0, transform=utilities.positive(), dtype=config.default_float())
        self._base_kern = self._cos
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    def _cos(self, X, X2=None):
        X_norm =  tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1))
        if X2 is None:
            # return self.gamma * tf.matmul(X, X, transpose_b = True) / (X_norm[..., :, None] * X_norm[..., None, :])
            return tf.matmul(X, X, transpose_b = True) / (X_norm[..., :, None] * X_norm[..., None, :])
        else:
            X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))
            # return self.gamma * tf.matmul(X, X2, transpose_b = True) / (X_norm[..., :, None] * X2_norm[..., None, :])
            return tf.matmul(X, X2, transpose_b = True) / (X_norm[..., :, None] * X2_norm[..., None, :])


class SignaturePoly(SignatureKernel):
    """
    The signature kernel, which uses a (finite number of) monomials of vectors - i.e. polynomial kernel - as state-space embedding
    """
    def __init__(self, num_features, len_examples, num_levels, gamma = 1, degree = 3, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        self.gamma = Parameter(gamma, transform=utilities.positive(), dtype=config.default_float())
        self.degree = Parameter(degree, dtype=config.default_float(), trainable=False)
        self._base_kern = self._poly
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    
    def _poly(self, X, X2=None):
        if X2 is None:
            return (tf.matmul(X, X, transpose_b = True) + self.gamma) ** self.degree
        else:
            return (tf.matmul(X, X2, transpose_b = True) + self.gamma) ** self.degree

class SignatureRBF(SignatureKernel):
    """
    The signature kernel, which uses an (infinite number of) monomials of vectors - i.e. Gauss/RBF/SquaredExponential kernel - as state-space embedding
    """
    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        self._base_kern = self._rbf
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    def _rbf(self, X, X2=None):
        K = tf.exp(-self._square_dist(X, X2) / 2)
        return K 

    

SignatureGauss = SignatureRBF
        
class SignatureMatern12(SignatureKernel):
    """
    The signature kernel, which uses the MA12 kernel as state-space embedding.
    """
    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        # self.gamma = Parameter(1.0, transform=utilities.positive(), dtype=config.default_float())
        self._base_kern = self._Matern12

    __init__.__doc__ = SignatureKernel.__init__.__doc__
    
    def _Matern12(self, X, X2 = None):
        r = self._euclid_dist(X, X2)
        # return tf.exp(-self.gamma * r)
        return tf.exp(-r)


SignatureLaplace = SignatureMatern12
SignatureExponential = SignatureMatern12

class SignatureMatern32(SignatureKernel):
    """
    The signature kernel, which uses the MA32 kernel as state-space embedding.
    """
    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        self._base_kern = self._Matern32

    __init__.__doc__ = SignatureKernel.__init__.__doc__
    
    def _Matern32(self, X, X2=None):
        r = self._euclid_dist(X, X2)

        return (1. + np.sqrt(3.)*r) * tf.exp(-np.sqrt(3.)*r)

    
    
class SignatureMatern52(SignatureKernel):
    """
    The signature kernel, which uses the MA52 kernel as state-space embedding.
    """
    def __init__(self, num_features, len_examples, num_levels, **kwargs):
        SignatureKernel.__init__(self, num_features, len_examples, num_levels, **kwargs)
        self._base_kern = self._Matern52

    __init__.__doc__ = SignatureKernel.__init__.__doc__
    
    def _Matern52(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        return (1.0 + np.sqrt(5.)*r + 5./3.*tf.square(r)) * tf.exp(-np.sqrt(5.)*r)
    
    