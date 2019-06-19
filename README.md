# GPSig
Variational Gaussian processes with signature covariances for classifying multivariate streams based on GPflow.

## Overview
The repository contains the source code for VGP models using signature covariances for supervised learning.

The main algorithms, which turn observations-vs-observations kernel matrices into a streams-vs-streams kernel matrix, are implemented in *sequential_algs.py*.

The *kernels.py* Python file contains the main class for kernels, and gives several options to use signature kernels, which are derived from different base kernels, such as the Euclidean inner product (SequentialLinear), the Gaussian RBF (SequentialRBF), or the exponential kernel (SequentialExponential).

The GP models, that can be considered re-implementations of the models in GPflow to be compatible with the sequentialised kernel, *SeqVGP* and *SeqSVGP* are found in *models.py.*. This file also implements the *SeqIDSVGP* model (Sequentialised Inter-Domain Variational Gaussian process), which uses inducing tensors to variationally summarise the input streams of data. For an example on using this model, see the notebook *notebooks/mts_example1.ipynb*.

The other files, *preprocessing.py*, *training.py*, *optimizers.py* contain convenience functins that are used for preprocessing the data, training models and importing optimizers from tf.contrib. See *notebooks/mts_exampl1.ipynb* for their usage.

Stay tuned for more examples and notebooks.

## Dependencies

The package dependencies that the library was tested with are available in the *requirements.txt* text file. 
