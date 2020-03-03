
import tensorflow as tf
import numpy as np

from gpflow.params import Parameter
from gpflow.decors import params_as_tensors, params_as_tensors_for, autoflow
from gpflow import transforms
from gpflow import settings
from gpflow.kernels import Kernel

from . import lags
from . import low_rank_calculations
from . import signature_algs

class SignatureKernel(Kernel):
    """
    """
    def __init__(self, input_dim, num_features, num_levels, active_dims=None, variances=1, lengthscales=1, order=1, normalization=True,
                 difference=True, num_lags=None, low_rank=False, num_components=50, rank_bound=None, sparsity='sqrt', name=None):
        """
        # Inputs:
        ## Args:
        :input_dim:         the total size of an input sample to the kernel
        :num_features:      the state-space dimension of the input sequebces,
        :num_levels:        the degree of cut-off for the truncated signatures
                            (i.e. len_examples = len(active_dims) / num_features)
        ## Kwargs:
        ### Kernel options
        :active_dims:       if specified, should contain a list of the dimensions in the input that should be sliced out and fed to the kernel.
                            if not specified, defaults to range(input_dim)
        :variances:          multiplicative scaling applied to the Signature kernel,
                            if ARD is True, there is one parameter for each level, i.e. variances is of size (num_levels + 1)                
		:lengthscales:      lengthscales for scaling the coordinates of the input sequences,
                            if lengthscales is None, no scaling is applied to the paths
                            if ARD is True, there is one lengthscale for each path dimension, i.e. lengthscales is of size (num_features)
        :order:             order of the signature kernel minimum is 1 and maximum is num_levels (set to -1 to set to max)
        :normalization:     False - no normalization, True - normalize signature levels
        :difference:        boolean indicating whether to difference paths
                            False corresponds to the signature of the integrated path
        :num_lags:          Nonnegative integer or None, the number of lags added to each sequence. Usually between 0-5.
        
        ### Low-rank options:
        :low_rank:          boolean indicating whether to use low-rank kernel
        :num_components:    number of components used in Nystrom approximation
        :rank_bound:        max rank of low-rank factor in signature algs, if None, defaults to num_components.
        :sparsity:          controls the sparsity of the random projection matrix used in the low-rank algorithm
                            possible values are:
                            - 'sqrt' - approximately O(n * sqrt(n)) non-zero entries;
                            - 'log' - approximately O(n * log(n)) non-zero entries;
                            - 'lin' - approximately O(n) non-zero entries;
        """
        
        super().__init__(input_dim, active_dims, name=name)
        self.num_features = num_features
        self.num_levels = num_levels
        self.len_examples = self._validate_number_of_features(input_dim, num_features)
        self.order = num_levels if (order <= 0 or order >= num_levels) else order

        if self.order != 1 and low_rank: 
            raise NotImplementedError('Higher-order algorithms not compatible with low-rank mode (yet).')

        self.normalization = normalization
        self.difference = difference

        self.variances = Parameter(self._validate_signature_param("variances", variances, num_levels + 1), transform=transforms.positive, dtype=settings.float_type)
        self.sigma = Parameter(1., transform=transforms.positive, dtype=settings.float_type)

        self.low_rank, self.num_components, self.rank_bound, self.sparsity = self._validate_low_rank_params(low_rank, num_components, rank_bound, sparsity)

        if num_lags is None:
            self.num_lags = 0	
        else:
            # check if right value
            if not isinstance(num_lags, int) or num_lags < 0:
                raise ValueError('The variable num_lags most be a nonnegative integer or None.')
            else:
                self.num_lags = int(num_lags)
                if num_lags > 0:
                    self.lags = Parameter(0.1 * np.asarray(range(1, num_lags+1)), transform=transforms.Logistic(), dtype=settings.float_type)
                    gamma = 1. / np.asarray(range(1, self.num_lags+2))
                    gamma /= np.sum(gamma)                   
                    self.gamma = Parameter(gamma, transform=transforms.positive, dtype=settings.float_type)

        if lengthscales is not None:
            lengthscales = self._validate_signature_param("lengthscales", lengthscales, self.num_features)
            self.lengthscales = Parameter(lengthscales, transform=transforms.positive, dtype=settings.float_type)
        else:
            self.lengthscales = None

	######################
	## Input validators ##
	######################
	
    def _validate_number_of_features(self, input_dim, num_features):
        """
        Validates the format of the input samples.
        """
        if input_dim % num_features == 0:
            len_examples = int(input_dim / num_features)
        else:
            raise ValueError("The arguments num_features and input_dim are not consistent.")
        return len_examples


    def _validate_low_rank_params(self, low_rank, num_components, rank_bound, sparsity):
        """
        Validates the low-rank options
        """
        if low_rank is not None and low_rank == True:
            if not type(low_rank) == bool:
                raise ValueError("Unknown low-rank argument: %s. It should be True of False." % low_rank)
            if sparsity not in ['log', 'sqrt', 'lin']:
                raise ValueError("Unknown sparsity argument %s. Possible values are 'sqrt', 'log', 'lin'" % sparsity)
            if rank_bound is not None and rank_bound <= 0:
                raise ValueError("The rank-bound in the low-rank algorithm must be either None or a positiv integer.")
            if num_components is None or num_components <= 0:
                raise ValueError("The number of components in the kernel approximation must be a positive integer.")
            if rank_bound is None:
                rank_bound = num_components
        else:
            low_rank = False
        return low_rank, num_components, rank_bound, sparsity

          
    def _validate_signature_param(self, name, value, length):
        """
        Validates signature params
        """
        value = value * np.ones(length, dtype=settings.float_type)
        correct_shape = () if length==1 else (length,)
        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError("shape of parameter {} is not what is expected ({})".format(name, length))
        return value
	

    ########################################
    ## Autoflow functions for interfacing ##
    ########################################


    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K(self, X, Y):
        return self.K(X, Y)

    @autoflow((settings.float_type, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X)

    @autoflow((settings.float_type, [None, None]))
    def compute_base_kern_symm(self, X):
        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, (num_examples, -1, self.num_features))
        len_examples = tf.shape(X)[1]
        X = tf.reshape(self._apply_scaling_and_lags_to_sequences(X), (-1, self.num_features))
        M = tf.transpose(tf.reshape(self._base_kern(X), [num_examples, len_examples, num_examples, len_examples]), [0, 2, 1, 3])
        return M

    @autoflow((settings.float_type, [None, None]))
    def compute_K_level_diags(self, X):
        return self.Kdiag(X, return_levels=True)

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K_levels(self, X, X2):
        return self.K(X, X2, return_levels=True)
    
    @autoflow((settings.float_type, [None, None]))
    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    @autoflow((settings.float_type, ))
    def compute_K_tens(self, Z):
        return self.K_tens(Z, return_levels=False)

    @autoflow((settings.float_type,), (settings.float_type, [None, None]))
    def compute_K_tens_vs_seq(self, Z, X): 
        return self.K_tens_vs_seq(Z, X, return_levels=False)
    
    @autoflow((settings.float_type, ))
    def compute_K_incr_tens(self, Z):
        return self.K_tens(Z, increments=True, return_levels=False)

    @autoflow((settings.float_type,), (settings.float_type, [None, None]))
    def compute_K_incr_tens_vs_seq(self, Z, X): 
        return self.K_tens_vs_seq(Z, X, increments=True, return_levels=False)

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
        X_feat = low_rank_calculations.Nystrom_map(X, self._base_kern, nys_samples, self.num_components)
        X_feat = tf.reshape(X_feat, [num_examples, len_examples, self.num_components])    
        
        if self.order == 1:
            Phi_lvls = signature_algs.signature_kern_first_order_lr_feature(X_feat, self.num_levels, self.rank_bound, self.sparsity, seeds, difference=self.difference)
        else:
            raise NotImplementedError('Low-rank mode not implemented for order higher than 1.')
        
        return Phi_lvls
                
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
            Z_feat = low_rank_calculations.Nystrom_map(Z, self._base_kern, nys_samples, self.num_components)
            Z_feat = tf.reshape(Z_feat, [len_tensors, num_tensors, 2, self.num_components])
            Z_feat = Z_feat[:, :, 1, :] - Z_feat[:, :, 0, :]
        else:
            Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])
            Z_feat = low_rank_calculations.Nystrom_map(Z, self._base_kern, nys_samples, self.num_components)
            Z_feat = tf.reshape(Z_feat, [len_tensors, num_tensors, self.num_components])
        
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

    @params_as_tensors
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
            X /= self.lengthscales[None, None, None, :]

        if self.num_lags > 0:
            X *= self.gamma[None, None, :, None]
        
        X = tf.reshape(X, (num_examples, len_examples, num_features))
        return X

    @params_as_tensors
    def _apply_scaling_to_tensors(self, Z):
        """
        Applies scaling to simple tensors of shape (num_levels*(num_levels+1)/2, num_tensors, num_features*(num_lags+1))
        """
        
        len_tensors, num_tensors = tf.shape(Z)[0], tf.shape(Z)[1]
        
        if self.lengthscales is not None:
            Z = tf.reshape(Z, (len_tensors, num_tensors, self.num_lags+1, self.num_features))
            Z /= self.lengthscales[None, None, None, :]
            if self.num_lags > 0:
                Z *= self.gamma[None, None, :, None]
            Z = tf.reshape(Z, (len_tensors, num_tensors, -1)) 

        return Z
    
    @params_as_tensors
    def _apply_scaling_to_incremental_tensors(self, Z):
        """
        Applies scaling to incremental tensors of shape (num_levels*(num_levels+1)/2, num_tensors, 2, num_features*(num_lags+1))
        """
        
        len_tensors, num_tensors, num_features = tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[-1]
        
        if self.lengthscales is not None:
            Z = tf.reshape(Z, (len_tensors, num_tensors, 2, self.num_lags+1, self.num_features))
            Z /= self.lengthscales[None, None, None, None, :]
            if self.num_lags > 0:
                Z *= self.gamma[None, None, None, :, None]

        Z = tf.reshape(Z, (len_tensors, num_tensors, 2, num_features))
        return Z
            
    @params_as_tensors
    def K(self, X, X2 = None, presliced = False, return_levels = False, presliced_X = False, presliced_X2 = False):
        """
        Computes signature kernel between sequences
        """

        if presliced:
            presliced_X = True
            presliced_X2 = True

        if not presliced_X and not presliced_X2:
            X, X2 = self._slice(X, X2)
        elif not presliced_X:
            X, _ = self._slice(X, None)
        elif not presliced_X2 and X2 is not None:
            X2, _ = self._slice(X2, None)

        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, [num_examples, -1, self.num_features])
        len_examples = tf.shape(X)[1]

        X_scaled = self._apply_scaling_and_lags_to_sequences(X)

        if X2 is None:
            if self.low_rank:
                Phi_lvls = self._K_seq_lr_feat(X)
                K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
            else:
                K_lvls = self._K_seq(X_scaled)
            
            if self.normalization:
                K_lvls += settings.jitter * tf.eye(num_examples, dtype=settings.float_type)[None]
                K_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(K_lvls))
                K_lvls /= K_lvls_diag_sqrt[:, :, None] * K_lvls_diag_sqrt[:, None, :]

        else:
            num_examples2 = tf.shape(X2)[0]
            X2 = tf.reshape(X2, [num_examples2, -1, self.num_features])
            len_examples2 = tf.shape(X2)[1]
            
            X2_scaled = self._apply_scaling_and_lags_to_sequences(X2)

            if self.low_rank:
                seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
                idx, _ = low_rank_calculations._draw_indices(num_examples*len_examples + num_examples2*len_examples2, self.num_components)
                
                nys_samples = tf.gather(tf.concat((tf.reshape(X, [num_examples*len_examples, -1]), tf.reshape(X2, [num_examples2*len_examples2, -1])), axis=0), idx, axis=0)
                
                Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)
                Phi2_lvls = self._K_seq_lr_feat(X2, nys_samples=nys_samples, seeds=seeds)

                K_lvls = tf.stack([tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
            else:
                K_lvls = self._K_seq(X_scaled, X2_scaled)

            if self.normalization:
                if self.low_rank:
                    K1_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
                    K2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
                else:
                    K1_lvls_diag = self._K_seq_diag(X_scaled)
                    K2_lvls_diag = self._K_seq_diag(X2_scaled)
                
                K1_lvls_diag += settings.jitter
                K2_lvls_diag += settings.jitter

                K1_lvls_diag_sqrt = tf.sqrt(K1_lvls_diag)
                K2_lvls_diag_sqrt = tf.sqrt(K2_lvls_diag)
                
                K_lvls /= K1_lvls_diag_sqrt[:, :, None] * K2_lvls_diag_sqrt[:, None, :]
        
        K_lvls *= self.sigma * self.variances[:, None, None]
        
        if return_levels:
            return K_lvls
        else:
            return tf.reduce_sum(K_lvls, axis=0)

    @params_as_tensors
    def Kdiag(self, X, presliced=False, return_levels=False):
        """
        Computes the diagonal of a square signature kernel matrix.
        """

        num_examples = tf.shape(X)[0]
        
        if self.normalization:
            if return_levels:
                return tf.tile(self.sigma * self.variances[:, None], [1, num_examples])
            else:
                return tf.fill((num_examples,), self.sigma * tf.reduce_sum(self.variances))
                
        if not presliced:
            X, _ = self._slice(X, None)

        X = tf.reshape(X, (num_examples, -1, self.num_features))

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            Phi_lvls = self._K_seq_lr_feat(X)
            K_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
        else:
            K_lvls_diag = self._K_seq_diag(X)

        K_lvls_diag *= self.sigma * self.variances[:, None]         

        if return_levels:
            return K_lvls_diag
        else:
            return tf.reduce_sum(K_lvls_diag, axis=0)
        
    @params_as_tensors
    def K_tens(self, Z, return_levels=False, increments=False):
        """
        Computes a square covariance matrix of inducing tensors Z.
        """
        
        num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        if self.low_rank:
            Phi_lvls = self._K_tens_lr_feat(Z, increments=increments)
            K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
        else:
            K_lvls = self._K_tens(Z, increments=increments)

        K_lvls *= self.sigma * self.variances[:, None, None]
        
        if return_levels:
            return K_lvls
        else:
            return tf.reduce_sum(K_lvls, axis=0)

    @params_as_tensors
    def K_tens_vs_seq(self, Z, X, return_levels=False, increments=False, presliced=False):
        """
        Computes a cross-covariance matrix between inducing tensors and sequences.
        """

        if not presliced:
            X, _ = self._slice(X, None)
        
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
            seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
            idx, _ = low_rank_calculations._draw_indices(num_tensors*len_tensors*(int(increments)+1) + num_examples*len_examples, self.num_components)
            nys_samples = tf.gather(tf.concat((tf.reshape(Z, [num_tensors*len_tensors*(int(increments)+1), -1]), tf.reshape(X, [num_examples*len_examples, -1])), axis=0), idx, axis=0)
            
            Phi_Z_lvls = self._K_tens_lr_feat(Z, increments=increments, nys_samples=nys_samples, seeds=seeds)
            Phi_X_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)

            Kzx_lvls = tf.stack([tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0) 
        else:
            Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

        if self.normalization:
            if self.low_rank:
                Kxx_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0)
            else:
                Kxx_lvls_diag = self._K_seq_diag(X)

            Kxx_lvls_diag += settings.jitter

            Kxx_lvls_diag_sqrt = tf.sqrt(Kxx_lvls_diag)
            Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]

        Kzx_lvls *= self.sigma * self.variances[:, None, None]
        
        if return_levels:
            return Kzx_lvls
        else:
            return tf.reduce_sum(Kzx_lvls, axis=0)

    @params_as_tensors
    def K_tens_n_seq_covs(self, Z, X, full_X_cov = False, return_levels=False, increments=False, presliced=False):
        """
        Computes and returns all three relevant matrices between inducing tensors tensors and input sequences, Kzz, Kzx, Kxx. Kxx is only diagonal if not full_X_cov
        """

        if not presliced:
            X, _ = self._slice(X, None)
        
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
            seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
            idx, _ = low_rank_calculations._draw_indices(num_tensors*len_tensors*(int(increments)+1) + num_examples*len_examples, self.num_components)
            nys_samples = tf.gather(tf.concat((tf.reshape(Z, [num_tensors*len_tensors*(int(increments)+1), -1]), tf.reshape(X, [num_examples*len_examples, -1])), axis=0), idx, axis=0)
            
            Phi_Z_lvls = self._K_tens_lr_feat(Z, increments=increments, nys_samples=nys_samples, seeds=seeds)
            Phi_X_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)

            Kzz_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_Z_lvls], axis=0)
            Kzx_lvls = tf.stack([tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
        else:
            Kzz_lvls = self._K_tens(Z, increments=increments)
            Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

        if full_X_cov:
            if self.low_rank:
                Kxx_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_X_lvls], axis=0)
            else:
                Kxx_lvls = self._K_seq(X)

            if self.normalization:
                Kxx_lvls += settings.jitter * tf.eye(num_examples, dtype=settings.float_type)[None]
                
                Kxx_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(Kxx_lvls))

                Kxx_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kxx_lvls_diag_sqrt[:, None, :]
                Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]
            
            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kzz_lvls *= self.sigma * self.variances[:, None, None]
            Kzx_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls
            else:
                return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls, axis=0)

        else:
            if self.low_rank:
                Kxx_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0)
            else:
                Kxx_lvls_diag = self._K_seq_diag(X)

            if self.normalization:
                Kxx_lvls_diag += settings.jitter

                Kxx_lvls_diag_sqrt = tf.sqrt(Kxx_lvls_diag)

                Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]
                Kxx_lvls_diag = tf.tile(self.sigma * self.variances[:, None], [1, num_examples])
            else:
                Kxx_lvls_diag *= self.sigma * self.variances[:, None]
            
            Kzz_lvls *= self.sigma * self.variances[:, None, None]
            Kzx_lvls *= self.sigma * self.variances[:, None, None]
            
            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls_diag
            else:
                return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls_diag, axis=0)

    @params_as_tensors
    def K_seq_n_seq_covs(self, X, X2, full_X2_cov = False, return_levels=False, presliced=False):
        """
        Computes and returns all three relevant matrices between inducing sequences and input sequences, Kxx, Kxx2, Kx2x2. Kx2x2 is only diagonal if not full_X2_cov
        """

        if not presliced:
            X2, _ = self._slice(X2, None)

        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, (num_examples, -1, self.num_features))
        len_examples = tf.shape(X)[1]

        num_examples2 = tf.shape(X2)[0]
        X2 = tf.reshape(X2, (num_examples2, -1, self.num_features))
        len_examples2 = tf.shape(X2)[1]

        X = self._apply_scaling_and_lags_to_sequences(X)
        X2 = self._apply_scaling_and_lags_to_sequences(X2)

        if self.low_rank:
            seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
            idx, _ = low_rank_calculations._draw_indices(num_examples*len_examples + num_examples2*len_examples2, self.num_components)
            nys_samples = tf.gather(tf.concat((tf.reshape(X, [num_examples*len_examples, -1]), tf.reshape(X2, [num_examples2*len_examples2, -1])), axis=0), idx, axis=0)
            
            Phi_lvls = self._K_seq_lr_feat(X, nys_samples=nys_samples, seeds=seeds)
            Phi2_lvls = self._K_seq_lr_feat(X2, nys_samples=nys_samples, seeds=seeds)

            Kxx_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
            Kxx2_lvls = tf.stack([tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True) for i in range(self.num_levels+1)], axis=0)
        else:
            Kxx_lvls = self._K_seq(X)
            Kxx2_lvls = self._K_seq(X, X2)

        if self.normalization:
            
            Kxx_lvls += settings.jitter * tf.eye(num_examples, dtype=settings.float_type)[None]

            Kxx_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(Kxx_lvls))
            Kxx_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kxx_lvls_diag_sqrt[:, None, :]
            Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None]
        
        if full_X2_cov:
            if self.low_rank:
                Kx2x2_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi2_lvls], axis=0)
            else:
                Kx2x2_lvls = self._K_seq(X2)

            if self.normalization:

                K_x2x2_lvls += settings.jitter * tf.eye(num_examples2, dtype=settings.float_type)[None]

                Kx2x2_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(K_x2x2_lvls)) 

                Kxx2_lvls /= Kx2x2_lvls_diags_sqrt[:, None, :]
                Kx2x2_lvls /= Kx2x2_lvls_diags_sqrt[:, :, None] * Kx2x2_lvls_diags_sqrt[:, None, :]
            
            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kxx2_lvls *= self.sigma * self.variances[:, None, None]
            Kx2x2_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls
            else:
                return tf.reduce_sum(Kxx_lvls, axis=0), tf.reduce_sum(Kxx2_lvls, axis=0), tf.reduce_sum(Kx2x2_lvls, axis=0)

        else:
            if self.low_rank:
                Kx2x2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
            else:
                Kx2x2_lvls_diag = self._K_seq_diag(X2)

            if self.normalization:
                Kx2x2_lvls_diag += settings.jitter

                Kx2x2_lvls_diag_sqrt = tf.sqrt(Kx2x2_lvls_diag)
                
                Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None] * Kx2x2_lvls_diag_sqrt[:, None, :]
                Kx2x2_lvls_diag = tf.tile(self.sigma * self.variances[:, None], [1, num_examples2])
            else:
                Kx2x2_lvls_diag *= self.sigma * self.variances[:, None]

            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kxx2_lvls *= self.sigma * self.variances[:, None, None]
        
            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls_diag
            else:
                return tf.reduce_sum(Kxx_lvls, axis=0), tf.reduce_sum(Kxx2_lvls, axis=0), tf.reduce_sum(Kx2x2_lvls_diag, axis=0)

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

    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        # self.gamma = Parameter(1.0/float(self.num_features), transform=transforms.positive, dtype=settings.float_type)
        # self.offsets = Parameter(np.zeros(self.num_features), dtype=settings.float_type)
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

    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        # self.gamma = Parameter(1.0, transform=transforms.positive, dtype=settings.float_type)
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
    def __init__(self, input_dim, num_features, num_levels, gamma = 1, degree = 3, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        self.gamma = Parameter(gamma, transform=transforms.positive, dtype=settings.float_type)
        self.degree = Parameter(degree, dtype=settings.float_type, trainable=False)
        self._base_kern = self._poly
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    @params_as_tensors
    def _poly(self, X, X2=None):
        if X2 is None:
            return (tf.matmul(X, X, transpose_b = True) + self.gamma) ** self.degree
        else:
            return (tf.matmul(X, X2, transpose_b = True) + self.gamma) ** self.degree

class SignatureRBF(SignatureKernel):
    """
    The signature kernel, which uses an (infinite number of) monomials of vectors - i.e. Gauss/RBF/SquaredExponential kernel - as state-space embedding
    """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        # self.sigma = Parameter([1.0], transform=transforms.positive, dtype=settings.float_type)
        # self.gamma = Parameter(1.0/float(self.num_features), transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._rbf
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    def _rbf(self, X, X2=None):
        K = tf.exp(-self._square_dist(X, X2) / 2)
        return K 

    

SignatureGauss = SignatureRBF

class SignatureMix(SignatureKernel):
    """
    The signature kernel, which uses a convex combination of identity and RBF as state-space embedding.
    """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        self.mixing = Parameter(0.5, transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._mix
    
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    def _mix(self, X, X2=None):
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            inner = tf.matmul(X, X, transpose_b=True)
            ds = Xs[..., :, None] + Xs[..., None, :] - 2 * inner
        else:
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            inner = tf.matmul(X, X2, transpose_b=True)
            ds = Xs[..., :, None] + X2s[..., None, :] - 2 * inner

        K = self.mixing * tf.exp(-ds/2) + (1. - self.mixing) * inner
        return K

class SignatureSpectral(SignatureKernel):
    """
    The signature kernel, which uses the spectral kernels as state-space embedding.
    """

    def __init__(self, input_dim, num_features, num_levels, family = 'gauss', Q = 5, **kwargs):

        SignatureKernel.__init__(self, input_dim, num_features, num_levels, lengthscales = None, **kwargs)

        if family == 'exp' or family == 'exponential':
            self.family = 'exp'
        elif family == 'gauss' or family == 'gaussian' or family == 'rbf':
            self.family = 'rbf'
        elif family =='mixed' or family == 'mix':
            self.family = 'mixed'
        else:
            raise ValueError("Unrecognized spectral family name.")

        self.Q = Q
        self.alpha = Parameter(np.exp(np.random.randn(Q)), transform=transforms.positive, dtype=settings.float_type)
        self.omega = Parameter(np.exp(np.random.randn(Q, self.num_features)), transform=transforms.positive, dtype=settings.float_type)
        self.gamma = Parameter(np.exp(np.random.randn(Q, self.num_features)), transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._spectral
        
    __init__.__doc__ = SignatureKernel.__init__.__doc__

    @params_as_tensors
    def _spectral(self, X, X2 = None):
        if X2 is None:
            diff_tiled = tf.tile((X[None, :, None, :] - X[None, None, :, :]), [self.Q, 1, 1, 1])
        else:
            diff_tiled = tf.tile((X[None, :, None, :] - X2[None, None, :, :]), [self.Q, 1, 1, 1])

        if self.family == 'exp':
            kern_term = tf.exp(- tf.sqrt(tf.reduce_sum(tf.square(diff_tiled * self.gamma[:, None, None, :]), axis=-1)) / 2)
        elif self.family == 'rbf':
            kern_term = tf.exp(- tf.reduce_sum(tf.square(diff_tiled * self.gamma[:, None, None, :]), axis=-1) / 2)
        elif self.family == 'mixed':
            Q1 = tf.cast(tf.floor(tf.cast(Q, settings.float_type) / 2.), settings.int_type)
            Q2 = tf.cast(tf.ceil(tf.cast(Q, settings.float_type) / 2.), settings.int_type)
            square_dist = - tf.reduce_sum(tf.square(diff_tiled * self.gamma[:, None, None, :]), axis=-1)
            rbf_term = tf.exp(-1.*square_dist[:Q1] / 2)
            exp_term = tf.exp( -1.*tf.sqrt(square_dist[Q1:]) / 2)
        spectral_term = tf.cos(2. * np.pi * tf.reduce_sum(diff_tiled * self.omega[:, None, None, :], axis=-1))
        if self.family == 'mixed':
            return tf.reduce_sum(rbf_term * spectral_term[:Q1] * self.alpha[:Q1, None, None], axis=0) \
                + tf.reduce_sum(exp_term * spectral_term[Q1:] * self.alpha[Q1:, None, None], axis=0)
        else:
            return tf.reduce_sum(kern_term * spectral_term * self.alpha[:, None, None], axis=0)
        
class SignatureMatern12(SignatureKernel):
    """
    The signature kernel, which uses the MA12 kernel as state-space embedding.
    """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        # self.gamma = Parameter(1.0, transform=transforms.positive, dtype=settings.float_type)
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
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._Matern32

    __init__.__doc__ = SignatureKernel.__init__.__doc__
    
    def _Matern32(self, X, X2=None):
        r = self._euclid_dist(X, X2)

        return (1. + np.sqrt(3.)*r) * tf.exp(-np.sqrt(3.)*r)

    
    
class SignatureMatern52(SignatureKernel):
    """
    The signature kernel, which uses the MA52 kernel as state-space embedding.
    """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        SignatureKernel.__init__(self, input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._Matern52

    __init__.__doc__ = SignatureKernel.__init__.__doc__
    
    def _Matern52(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        return (1.0 + np.sqrt(5.)*r + 5./3.*tf.square(r)) * tf.exp(-np.sqrt(5.)*r)
    
    