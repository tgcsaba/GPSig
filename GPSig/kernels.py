
import tensorflow as tf
import numpy as np

from gpflow.params import Parameter
from gpflow.decors import params_as_tensors, autoflow
from gpflow import transforms
from gpflow import settings
from gpflow.kernels import Kernel

from . import helpers
from . import low_rank_calculations
from . import sequential_algs

class Sequential(Kernel):
    """
    Base class for sequentialized kernels.
    """
    def __init__(self, input_dim, active_dims=None, num_features=1, num_levels=2, variances=1.0, lengthscales=1.0, ARD=None, normalize_levels=True,
                 include_white=False, difference=True, num_lags=None, low_rank=False, num_components=None, rank_bound=None, sparsity='sqrt', name=None):
        """
        Inputs
        :input_dim:         the total size of an input sample to the kernel
        :active_dims:       if specified, should contain a list of the dimensions in the input that should be sliced out and fed to the kernel.
                            if not specified, defaults to range(input_dim)
        :num_features:      the number of dimensions in the input streams,
                            (i.e. stream_length = len(active_dims) / num_features)
        :num_levels:        the degree of cut-off for the truncated signatures
        :variances:          multiplicative scaling applied to the sequential kernel,
                            if ARD is True, there is one parameter for each level, i.e. variances is of size (num_levels + 1)                
		:lengthscales:      lengthscales for scaling the input streams,
                            if lengthscales is None, no scaling is applied to the paths
                            if ARD is True, there is one lengthscale for each path dimension, i.e. lengthscales is of size (num_features)
        :ARD:               boolean indicating whether the ARD option is used
        :normalize_levels:  boolean indicating whether the signature levels are normalized independently
        :include_white:     boolean indicating whether to include a white kernel in the static kernel -> only implemented for SequentialLinear
        :difference:        boolean indicating whether to difference base kernel matrix in sequentializer algorithm.
                            False corresponds to the signature of the integrated path
        :num_lags:          Nonnegative integer or None, the number of lags added to each streams. Usually between 0-5.
        :low_rank:          boolean indicating whether low-rank calculations are used
        :num_components:    number of components used in Nystrom approximation and the rank-bound in the low-rank sequential alg.
        :rank_bound:        max rank of low-rank factor in sequential alg, if None, defaults to num_components.
        :sparsity:          controls the sparsity of the random projection matrix,
                            possible values are: 'sqrt' - accurate, costly; 'log' - less accurate, cheaper; 'subsample' - least accurate, cheapest
        
        """
        
        super().__init__(input_dim, active_dims, name=name)
        self.num_features = num_features
        self.num_levels = num_levels
        self.stream_length = self._validate_number_of_features(input_dim, num_features)
        self.normalize_levels = normalize_levels
        self.difference = difference
        
        self.low_rank, self.num_components, self.rank_bound, self.sparsity = self._validate_low_rank_params(low_rank, num_components, rank_bound, sparsity)

        
        variances, self.ARD = self._validate_sequential_ard_params("variances", variances, num_levels + 1, ARD)
        # variances = np.ones(num_levels+1, dtype=settings.float_type)
        self.variances = Parameter(variances, transform=transforms.positive, dtype=settings.float_type)
            

        if num_lags is None:
            self.num_lags = 0
        else:
            # check if right value
            if not isinstance(num_lags, int) or num_lags < 0:
                raise ValueError('The variable num_lags most be a nonnegative integer or None.')
            else:
                self.num_lags = int(num_lags)
                if num_lags > 0:
                    self.lags = Parameter(1.0 / (float(num_lags) + 1) * np.array(range(1, num_lags+1), dtype=settings.float_type), transform=transforms.Logistic(), dtype=settings.float_type)

        if lengthscales is not None:
            lengthscales, _ = self._validate_sequential_ard_params("lengthscales", lengthscales, self.num_features, ARD)
            self.lengthscales = Parameter(lengthscales, transform=transforms.positive, dtype=settings.float_type)
        else:
            self.lengthscales = None
        
        # self.offsets = None

        self.include_white = include_white
        if include_white:
            self.white = Parameter(0.05, transform=transforms.positive, dtype=settings.float_type)

	######################
	## Input validators ##
	######################
	
    def _validate_number_of_features(self, input_dim, num_features):
        """
        Validates the format of the input samples.
        """
        if input_dim % num_features == 0:
            stream_length = int(input_dim / num_features)
        else:
            raise ValueError("The arguments num_features and input_dim are not consistent.")
        return stream_length
    
    def _validate_low_rank_params(self, low_rank, num_components, rank_bound, sparsity):
        """
        Validates the low-rank related arguments.
        """
        if low_rank is not None and low_rank == True:
            if not type(low_rank) == bool:
                raise ValueError("Unknown low-rank argument: %s. It should be True of False." % low_rank)
            if sparsity not in ['log', 'sqrt', 'subsample', 'subsample_gauss']:
                raise ValueError("Unknown sparsity argument %s. Possible values are 'sqrt', 'log', 'subsample', 'subsample_gauss'." % sparsity)
            if rank_bound is not None and rank_bound <= 0:
                raise ValueError("The rank-bound in the low-rank algorithm must be either None or a positiv integer.")
            if num_components is None or num_components <= 0:
                raise ValueError("The number of components in the kernel approximation must be a positive integer.")
        else:
            low_rank = False
        return low_rank, num_components, rank_bound, sparsity
          
    def _validate_sequential_ard_params(self, name, value, length, ARD=None):
        """
        Validates a potentially ARD hyperparameter.
        """
        if ARD is None:
            ARD = np.asarray(value).squeeze().shape != ()
        if ARD:
            value = value * np.ones(length, dtype=settings.float_type)
        if length == 1 or not ARD:
            correct_shape = ()
        else:
            correct_shape = (length,)
        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError("shape of {} does not match input_dim".format(name))
        return value, ARD
	

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
    
    @autoflow((settings.float_type, [None, None, None]))
    def compute_K_inducing(self, Z):
        return self.Kinter(Z)

    @autoflow((settings.float_type, [None, None]))
    def compute_K_symm_full(self, X):
        return self.K(X, override_full = True)

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K_full(self, X, Y):
        return self.K(X, Y, override_full = True)

    @autoflow((settings.float_type, [None, None]))
    def compute_phi(self, X):
        return self.K(X, feature_map = True)

    @autoflow((settings.float_type, [None, None]))
    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None, None]))
    def compute_K_inter(self, X, Z):
        return self.Kinter(X, Z)
    
    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_K_presliced(self, X, X2 = None):
        return self.K(X, X2, presliced=True)

    
    @params_as_tensors
    def _sequentializer(self, base_kern, X, X2 = None, feature_map = False, override_full = False):
        """
        Input
        :X:             (num_streams1, num_features * stream_length1) array of flattened streams 
        :X2:            (num_streams2, num_features * stream_length2) array of flattened streams
        :base_kern:     function handle to a kernel function that takes two arrays as input,
                        say, z1 of size (num_samples1, num_features) and z2 of size (num_samples2, num_features),
                        and computes k(z1,z2) of size (num_samples1, num_samples2)
        :feature_map:   boolean indicating whether output should be a feature map if in low_rank mode, else an error is raised
        :override_full: boolean flag indicating that full rank calculations should be used for computing the sequentialized kernel
                        not compatible with feature_map, if not in low_rank mode, an error is raised 
        Output
        :K:             (num_streams1, num_streams2) tensor of the sequentialized kernel matrix     
        """


        # Handle error messages for input flags
        if not self.low_rank and feature_map:
            raise ValueError("Feature map representation is only valid in low-rank mode.")
        if override_full and feature_map:
            raise ValueError("Feature map output is not compatible with full-rank mode.")
        
        num_streams = tf.shape(X)[0]
        X = tf.reshape(X, [num_streams, -1, self.num_features])
        stream_length = tf.shape(X)[1]
        if X2 is not None:
            num_streams2 = tf.shape(X2)[0]
            X2 = tf.reshape(X2, [num_streams2, -1, self.num_features])
            stream_length2 = tf.shape(X2)[1]
        
        if self.lengthscales is not None:
             X /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales
             if X2 is not None:
                 X2 /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales

        # if self.offsets is not None:
        #      X -= self.offsets[None, None, :]
        #      if X2 is not None:
        #          X2 -= self.offsets[None, None, :]
        
        num_features = self.num_features
        if self.num_lags > 0:
            num_features *= self.num_lags + 1
            X = helpers.add_lags_to_streams(X, self.lags)
            if X2 is not None: 
                X2 = helpers.add_lags_to_streams(X2, self.lags)

        num_samples = num_streams * stream_length
        X = tf.reshape(X, [num_samples, num_features])
        if X2 is not None:
            num_samples2 = num_streams2 * stream_length2
            X2 = tf.reshape(X2, [num_samples2, num_features])

        # Calls appropriate sequential algorithm
        if override_full or not self.low_rank:
            if X2 is None:
                M = base_kern(X)
                M = tf.reshape(M, [num_streams, stream_length, num_streams, stream_length])
                K = sequential_algs.sequentialize_kern_symm(M, self.variances, self.num_levels, self.normalize_levels, difference=self.difference) 
            else:
                M = tf.reshape(base_kern(X, X2), [num_streams, stream_length, num_streams2, stream_length2])
                if self.normalize_levels:
                    Mxx = base_kern(tf.reshape(X, [num_streams, stream_length, num_features]))
                    Myy = base_kern(tf.reshape(X2, [num_streams2, stream_length2, num_features]))
                    _, K_i_norms1 = sequential_algs.sequentialize_kern_diag(Mxx, self.variances, self.num_levels, normalize_levels=True,
                                                                                difference=self.difference, return_level_norms=True)
                    _, K_i_norms2 = sequential_algs.sequentialize_kern_diag(Myy, self.variances, self.num_levels, normalize_levels=True, 
                                                                                difference=self.difference, return_level_norms=True)
                    K = sequential_algs.sequentialize_kern_rect(M, self.variances, self.num_levels, normalize_levels=True,
                                                                difference=self.difference, K_i_norms1=K_i_norms1, K_i_norms2=K_i_norms2)
                else:
                    K = sequential_algs.sequentialize_kern_rect(M, self.variances, self.num_levels, normalize_levels=False, difference=self.difference)  
            return K
        else:
            num_samples = num_streams * stream_length
            if X2 is not None:
                num_samples2 = num_streams2 * stream_length2
                X = tf.concat((X, X2), axis=0)
            X_feat = low_rank_calculations.Nystrom_map(X, base_kern, num_components=self.num_components)
            if X2 is not None:
                X_feat, X_feat2 = tf.split(X_feat, [num_samples, num_samples2], axis=0)
                X_feat2 = tf.reshape(X_feat2, [num_streams2, stream_length2, -1], name='X_feat2')
            X_feat = tf.reshape(X_feat, [num_streams, stream_length, -1], name='X_feat')
            rank_bound = self.rank_bound or self.num_components
            if X2 is not None:
                seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
                Phi = sequential_algs.sequentialize_kern_low_rank_feature(X_feat, self.variances, self.num_levels, rank_bound, self.sparsity, self.normalize_levels, difference=self.difference, seeds=seeds)
                Phi2 = sequential_algs.sequentialize_kern_low_rank_feature(X_feat2, self.variances, self.num_levels, rank_bound, self.sparsity, self.normalize_levels, difference=self.difference, seeds=seeds)
            else:
                Phi = sequential_algs.sequentialize_kern_low_rank_feature(X_feat, self.variances, self.num_levels, rank_bound, self.sparsity, self.normalize_levels, difference=self.difference)
            if feature_map:
                if X2 is not None:
                    return Phi, Phi2
                else:
                    return Phi
            else:
                if X2 is None:
                    return tf.matmul(Phi, Phi, transpose_b=True)
                if X2 is not None:
                    K = tf.matmul(Phi, Phi2, transpose_b=True)
                    return K

    @params_as_tensors
    def _sequentializer_inducing(self, base_kern, Z, X, full_cov = False, override_full = False):
        """
        Input
        :Z:             (num_inducing, num_features * stream_length1) array of flattened streams 
        :X:             (num_streams, num_features * stream_length2) array of flattened streams
        :base_kern:     function handle to a kernel function that takes two arrays as input,
                        say, z1 of size (num_samples1, num_features) and z2 of size (num_samples2, num_features),
                        and computes k(z1,z2) of size (num_samples1, num_samples2)
        :override_full: boolean flag indicating that full rank calculations should be used for computing the sequentialized kernel     
        :full_cov:      whether to compute the off-diagonal terms of Kxx
        Output
        :Kzz:           (num_inducing, num_inducing) tensor
        :Kzx:           (num_inducing, num_streams) tensor
        :Kxx:           (num_streams,) or (num_streams, num_streams) tensor
        """
        
        num_inducing = tf.shape(Z)[0]
        Z = tf.reshape(Z, [num_inducing, -1, self.num_features])
        inducing_length = tf.shape(Z)[1]
        
        num_streams = tf.shape(X)[0]
        X = tf.reshape(X, [num_streams, -1, self.num_features])
        stream_length = tf.shape(X)[1]
        
        if self.lengthscales is not None:
             X /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales
             Z /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales

        # if self.offsets is not None:
        #      X -= self.offsets[None, None, :]
        #      Z -= self.offsets[None, None, :]

        num_features = self.num_features
        if self.num_lags > 0:
            num_features *= self.num_lags + 1
            X = helpers.add_lags_to_streams(X, self.lags) 
            Z = helpers.add_lags_to_streams(Z, self.lags) 
        
        X = tf.reshape(X, [-1, num_features])
        Z = tf.reshape(Z, [-1, num_features])

        if override_full or not self.low_rank:
                Mzz = tf.reshape(base_kern(Z), [num_inducing, inducing_length, num_inducing, inducing_length], name='Mzz')
                Mzx = tf.reshape(base_kern(Z, X), [num_inducing, inducing_length, num_streams, stream_length],  name='Mzx')
                if full_cov:
                    Mxx = tf.reshape(base_kern(X), [num_streams, stream_length, num_streams, stream_length])
                else:
                    Mxx = tf.reshape(base_kern(tf.reshape(X, [num_streams, stream_length, num_features])), [num_streams, stream_length, stream_length], name='Mxx')
                if self.normalize_levels:    
                    Kzz, Kzz_i_norms = sequential_algs.sequentialize_kern_symm(Mzz, self.variances, self.num_levels, difference=self.difference, normalize_levels=True, return_level_norms=True) 
                    if full_cov:
                        Kxx, Kxx_i_norms = sequential_algs.sequentialize_kern_symm(Mxx, self.variances, self.num_levels, difference=self.difference, normalize_levels=True, return_level_norms=True)
                    else:
                        Kxx, Kxx_i_norms = sequential_algs.sequentialize_kern_diag(Mxx, self.variances, self.num_levels, difference=self.difference, normalize_levels=True, return_level_norms=True)
                    Kzx = sequential_algs.sequentialize_kern_rect(Mzx, self.variances, self.num_levels, difference=self.difference,
                                                                  normalize_levels=True, K_i_norms1=Kzz_i_norms, K_i_norms2=Kxx_i_norms)
                else:
                    Kzz = sequential_algs.sequentialize_kern_symm(Mzz, self.variances, self.num_levels, difference=self.difference, normalize_levels=False)
                    if full_cov:
                        Kxx = sequential_algs.sequentialize_kern_symm(Mxx, self.variances, self.num_levels, difference=self.difference, normalize_levels=False)
                    else:
                        Kxx = sequential_algs.sequentialize_kern_diag(Mxx, self.variances, self.num_levels, difference=self.difference, normalize_levels=False)
                    Kzx = sequential_algs.sequentialize_kern_rect(Mzx, self.variances, self.num_levels, difference=self.difference, normalize_levels=False)
        else:
            features = low_rank_calculations.Nystrom_map(tf.concat((Z, X), axis=0), base_kern, num_components=self.num_components)
            Z_feat, X_feat = tf.split(features, [num_inducing * inducing_length, num_streams * stream_length], axis=0)
            Z_feat = tf.reshape(Z_feat, [num_inducing, inducing_length, num_features])
            X_feat = tf.reshape(X_feat, [num_streams, stream_length, num_features])
            
            rank_bound = self.rank_bound if self.rank_bound is not None else self.num_components
            seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
            Phi_z = sequential_algs.sequentialize_kern_low_rank_feature(Z_feat, self.variances, self.num_levels, rank_bound, self.sparsity, self.normalize_levels, difference=self.difference, seeds=seeds)
            Phi_x = sequential_algs.sequentialize_kern_low_rank_feature(X_feat, self.variances, self.num_levels, rank_bound, self.sparsity, self.normalize_levels, difference=self.difference, seeds=seeds) 
            
            Kzz = tf.matmul(Phi_z, Phi_z, transpose_b=True)
            Kzx = tf.matmul(Phi_z, Phi_x, transpose_b=True)
            if full_cov:
                Kxx = tf.matmul(Phi_x, Phi_x, transpose_b=True)
            else:
                Kxx = tf.reduce_sum(tf.square(Phi_x), axis=-1)
        
        return Kzz, Kzx, Kxx
                

    @params_as_tensors
    def _sequentializer_inter(self, base_kern, Z, X = None, full_cov = False, override_full = False): #, feature_map = False):
        """
        Input
        :base_kern:     function handle to a kernel function that takes two arrays as input,
                        say, z1 of size (num_samples1, num_features) and z2 of size (num_samples2, num_features),
                        and computes k(z1,z2) of size (num_samples1, num_samples2)
        :Z:             (num_levels*(num_levels+1)/2, num_inducing, num_features*(num_lags+1)) tensor of inter-domain inducing-points
        :X:             (num_streams1, num_features * stream_length) array of flattened streams 
        Output
        :Kzz:           (num_inducing, num_inducing)
        And if X is not None:
        :Kzz:           (num_inducing, num_inducing) inter-domain covariance matrix of inducing-points
        :Kzx:           (num_inducing, num_streams) tensor of inter-domain cross-covariances between inducing-points and sample-paths
        :Kxx:           (num_streams, num_streams) or (num_streams,) sequentialized kernel entries
        """


        inducing_length, num_inducing = tf.shape(Z)[0], tf.shape(Z)[1]
        
        if self.lengthscales is not None:
            Z /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales

        if X is not None:
            num_streams = tf.shape(X)[0]
            X = tf.reshape(X, [num_streams, -1, self.num_features])
            stream_length, num_features = tf.shape(X)[1], tf.shape(X)[2]
            if self.lengthscales is not None:
                X /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales
            # if self.offsets is not None:
            #     X -= self.offsets[None, None, :]
            if self.num_lags > 0:
                num_features *= self.num_lags + 1
                X = helpers.add_lags_to_streams(X, self.lags)

            X = tf.reshape(X, [-1, self.num_features * (self.num_lags + 1)])
        
        if not self.low_rank or override_full:
            Mzz = base_kern(tf.reshape(Z, [inducing_length, num_inducing, -1]))
            Kzz = sequential_algs.sequentialize_kern_inducing(Mzz, self.variances, self.num_levels)
            if X is None:
                return Kzz
            
            Z = tf.reshape(Z, [num_inducing * inducing_length, -1])
            Mzx = tf.reshape(base_kern(Z, X), [inducing_length, num_inducing, num_streams, stream_length])
            if full_cov:
                Mxx = tf.reshape(base_kern(X), [num_streams, stream_length, num_streams, stream_length])
            else:
                Mxx = tf.reshape(base_kern(tf.reshape(X, [num_streams, stream_length, num_features])), [num_streams, stream_length, stream_length])

            if self.normalize_levels:
                if full_cov:
                    Kxx, K_i_norms = sequential_algs.sequentialize_kern_symm(Mxx, self.variances, self.num_levels, normalize_levels=True,
                                                                                difference=self.difference, return_level_norms=True)
                else:
                    Kxx, K_i_norms = sequential_algs.sequentialize_kern_diag(Mxx, self.variances, self.num_levels, normalize_levels=True,
                                                                                difference=self.difference, return_level_norms=True)
                Kzx = sequential_algs.sequentialize_kern_inter(Mzx, self.variances, self.num_levels, normalize_levels=True,
                                                               difference=self.difference, K_i_norms=K_i_norms)
            else:
                if full_cov:
                    Kxx = sequential_algs.sequentialize_kern_symm(Mxx, self.variances, self.num_levels, normalize_levels=False, difference=self.difference)
                else:
                    Kxx = sequential_algs.sequentialize_kern_diag(Mxx, self.variances, self.num_levels, normalize_levels=True, difference=self.difference)
                Kzx = sequential_algs.sequentialize_kern_inter(Mzx, self.variances, self.num_levels, normalize_levels=True, difference=self.difference)
        else:
            Z = tf.reshape(Z, [num_inducing * inducing_length, -1])
            rank_bound = self.rank_bound if self.rank_bound is not None else self.num_components
            if X is None:    
                Z_feat = low_rank_calculations.Nystrom_map(Z, base_kern, num_components=self.num_components)
                Phi_z = sequential_algs.sequentialize_kern_inducing_low_rank_feature(Z_feat, self.variances, self.num_levels, rank_bound, self.sparsity)
                Kzz = tf.matmul(Phi_z, Phi_z, transpose_b=True)
                return Kzz

            features  = low_rank_calculations.Nystrom_map(tf.concat((Z, X), axis=0), base_kern, num_components=self.num_components)
            Z_feat, X_feat = tf.split(features, [num_inducing * inducing_length, num_streams * stream_length], axis=0)
            Z_feat = tf.reshape(Z_feat, [inducing_length, num_inducing, -1])
            X_feat = tf.reshape(X_feat, [num_streams, stream_length, -1])
            
            seeds = tf.random_uniform((self.num_levels-1, 2), minval=0, maxval=np.iinfo(settings.int_type).max, dtype=settings.int_type)
            Phi_x = sequential_algs.sequentialize_kern_low_rank_feature(X_feat, self.variances, self.num_levels, rank_bound, self.sparsity,
                                                                            self.normalize_levels, difference=self.difference, seeds=seeds)
            Phi_z = sequential_algs.sequentialize_kern_inducing_low_rank_feature(Z_feat, self.variances, self.num_levels, rank_bound, self.sparsity, seeds)
            
            Kzz = tf.matmul(Phi_z, Phi_z, transpose_b=True)
            Kzx = tf.matmul(Phi_z, Phi_x, transpose_b=True)
            if full_cov:
                Kxx = tf.matmul(Phi_x, Phi_x, transpose_b=True)
            else:
                Kxx = tf.reduce_sum(tf.square(Phi_x), axis=-1)
        
        return Kzz, Kzx, Kxx


    @params_as_tensors
    def _sequentializer_diag(self, base_kern, X, override_full=False):
        """
        Input
        :X:         (num_streams1, num_features * stream_length) array of flattened streams 
        :base_kern: function handle to a kernel function that takes two arrays as input,
                    say, z1 of size (num_samples1, num_features) and z2 of size (num_samples2, num_features),
                    and computes k(z1,z2) of size (num_samples1, num_samples2)
        Output
        :K:         (num_streams1) array of (full-rank) sequentialized kernel entries corresponding to the diagonal entries.
        """

        if self.low_rank:
            raise NotImplementedError("Diagonal mode not implemented for low-rank mode. Use feature-map output with sequentializer instead.")

        
        num_streams = tf.shape(X)[0]
        X = tf.reshape(X, [num_streams, -1, self.num_features])
        stream_length = tf.shape(X)[1]

        if self.lengthscales is not None:
             X /= self.lengthscales[None, None, :] if self.ARD else self.lengthscales
        # if self.offsets is not None:
        #      X -= self.offsets[None, None, :]
        
        num_features = self.num_features
        if self.num_lags > 0:
            num_features *= self.num_lags + 1
            X = helpers.add_lags_to_streams(X, self.lags) 


        M = base_kern(X)
        M = tf.reshape(M, [num_streams, stream_length, stream_length])
        K = sequential_algs.sequentialize_kern_diag(M, self.variances, self.num_levels, self.normalize_levels, difference=self.difference)
        return K 
    

	###############################
	## Kernel related functions  ##
	###############################
    
    @params_as_tensors
    def _square_dist(self, X, X2):
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

    def K(self, X, X2 = None, presliced = False, feature_map = False, override_full = False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self._sequentializer(self._base_kern, X, X2, feature_map, override_full)

    def Kdiag(self, X, presliced=False, override_full = False):
        if not presliced:
            X, _ = self._slice(X, None)
        return self._sequentializer_diag(self._base_kern, X, override_full=override_full)

    def Kinducing(self, Z, X, presliced=False, override_full=False, full_cov=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return self._sequentializer_inducing(self._base_kern, Z, X, override_full=override_full, full_cov=full_cov)
        

    def Kinter(self, Z, X=None, presliced=False, override_full=False, full_cov=False):
        if X is not None and not presliced:
            X, _ = self._slice(X, None)
        return self._sequentializer_inter(self._base_kern, Z, X, full_cov=full_cov)

class SequentialLinear(Sequential):
    """
    The sequentialized linear kernel
    """

    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        # self.gamma = Parameter(1.0/float(self.num_features), transform=transforms.positive, dtype=settings.float_type)
        # self.offsets = Parameter(np.zeros(self.num_features), dtype=settings.float_type)
        self._base_kern = self._lin
    
    __init__.__doc__ = Sequential.__init__.__doc__

    def _lin(self, X, X2=None):
        if X2 is None:
            # K = self.gamma * tf.matmul(X, X, transpose_b = True)
            K = tf.matmul(X, X, transpose_b = True)
            if self.include_white:
                num_data = tf.shape(K)[-1]
                diag = self.white * tf.eye(num_data, dtype=settings.float_type)
                if K.get_shape().ndims > 2:
                    diag = diag[None, :, :]
                K += diag
            return  K
        else:
            # return self.gamma * tf.matmul(X, X2, transpose_b = True)
            return tf.matmul(X, X2, transpose_b = True)

class SequentialCosine(Sequential):
    """
    The sequentialized cosine similarity kernel
    """

    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        # self.gamma = Parameter(1.0, transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._cos
    
    __init__.__doc__ = Sequential.__init__.__doc__

    def _cos(self, X, X2=None):
        X_norm =  tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1))
        if X2 is None:
            # return self.gamma * tf.matmul(X, X, transpose_b = True) / (X_norm[..., :, None] * X_norm[..., None, :])
            return tf.matmul(X, X, transpose_b = True) / (X_norm[..., :, None] * X_norm[..., None, :])
        else:
            X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))
            # return self.gamma * tf.matmul(X, X2, transpose_b = True) / (X_norm[..., :, None] * X2_norm[..., None, :])
            return tf.matmul(X, X2, transpose_b = True) / (X_norm[..., :, None] * X2_norm[..., None, :])


class SequentialPoly(Sequential):
    """
    The sequentialized polynomial kernel
    """
    def __init__(self, input_dim, gamma = 1, degree = 3, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        self.gamma = Parameter(gamma, transform=transforms.positive, dtype=settings.float_type)
        self.degree = Parameter(degree, dtype=settings.float_type, trainable=False)
        self._base_kern = self._poly
    
    __init__.__doc__ = Sequential.__init__.__doc__

    @params_as_tensors
    def _poly(self, X, X2=None):
        if X2 is None:
            return (tf.matmul(X, X, transpose_b = True) + self.gamma) ** self.degree
        else:
            return (tf.matmul(X, X2, transpose_b = True) + self.gamma) ** self.degree

class SequentialRBF(Sequential):
    """
    The sequentialized RBF kernel
    """
    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        # self.sigma = Parameter([1.0], transform=transforms.positive, dtype=settings.float_type)
        # self.gamma = Parameter(1.0/float(self.num_features), transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._rbf
    
    __init__.__doc__ = Sequential.__init__.__doc__

    def _rbf(self, X, X2=None):
        # K = tf.exp(-self.gamma * self._square_dist(X, X2) / 2)
        K = tf.exp(-self._square_dist(X, X2) / 2)
        if X2 is None and self.include_white:
            num_data = tf.shape(K)[-1]
            diag = self.white * tf.eye(num_data, dtype=settings.float_type)
            if K.get_shape().ndims > 2:
                diag = diag[None, :, :]
            K += diag
        return K 

    

SequentialGaussian = SequentialRBF

class SequentialMix(Sequential):
    """
    The sequentialized RBF kernel
    """
    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        self.mixing = Parameter([0.5], transform=transforms.Logistic())
        self._base_kern = self._mix
    
    __init__.__doc__ = Sequential.__init__.__doc__

    def _mix(self, X, X2=None):
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            inner = tf.matmul(X, X, transpose_b=True)
            square_dist = Xs[..., :, None] + Xs[..., None, :] - 2 * inner
        else: 
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            inner = tf.matmul(X, X2, transpose_b=True)
            square_dist = Xs[..., :, None] + X2s[..., None, :] - 2 * inner
        
        return self.mixing[0] * tf.exp(-1.0/2.0 * square_dist) + (1.0 - self.mixing[0]) * inner
        
class SequentialMatern12(Sequential):
    """
    The sequentialized Matern 1/2 kernel
    """
    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        # self.gamma = Parameter(1.0, transform=transforms.positive, dtype=settings.float_type)
        self._base_kern = self._Matern12

    __init__.__doc__ = Sequential.__init__.__doc__
    
    def _Matern12(self, X, X2 = None):
        r = self._euclid_dist(X, X2)
        # return tf.exp(-self.gamma * r)
        return tf.exp(-r)


SequentialLaplace = SequentialMatern12
SequentialExponential = SequentialMatern12

class SequentialMatern32(Sequential):
    """
    The sequentialized Matern 3/2 kernel
    """
    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        self._base_kern = self._Matern32

    __init__.__doc__ = Sequential.__init__.__doc__
    
    def _Matern32(self, X, X2=None):
        r = self._euclid_dist(X, X2)

        return (1. + np.sqrt(3.)*r) * tf.exp(-np.sqrt(3.)*r)

    
    
class SequentialMatern52(Sequential):
    """
    The sequentialized Matern 5/2 kernel
    """
    def __init__(self, input_dim, **kwargs):
        Sequential.__init__(self, input_dim, **kwargs)
        self._base_kern = self._Matern52

    __init__.__doc__ = Sequential.__init__.__doc__
    
    def _Matern52(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        return (1.0 + np.sqrt(5.)*r + 5./3.*tf.square(r)) * tf.exp(-np.sqrt(5.)*r)
    
    