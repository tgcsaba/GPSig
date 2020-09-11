# GPSig
A Gaussian process library for Bayesian learning from sequential data, such as time series, using signature kernels as covariance functions based on GPflow and TensorFlow. This repository contains supplementary code to the paper https://arxiv.org/abs/1906.08215.
***
## Installing
To get started, you should first clone the repository using git, e.g. with the command
```
git clone https://github.com/tgcsaba/GPSig.git
```
and then create and activate virtual environment with Python <= 3.7
```
conda create -n env_name python=3.7
conda activate env_name
```
Then, install the requirements using pip by
```
pip install -r requirements.txt
```
If you would like to use a GPU to run computations (which we heavily recommend, if you have one available), you most likely need to install a GPU compatible version of TensorFlow instead.
Depending on your OS and CUDA compute capability of your GPU, you might be able to acquire a pre-built version of Tensorflow for your system (from pip, conda or other sources). In some cases, you might have to build it on your system yourself (https://www.tensorflow.org/install/source), which is generally recommended so that you end up with a version that is able to make full use of your hardware.
***
## Getting started
To get started, we suggest to first look at the notebook `signature_kernel.ipynb`, which gives a simple worked out example of how to use the signature kernel as a standalone object. In this notebook, we validate the implementation of the signature kernel by comparing our results to an alternative way of computing signature features using the `esig` package.
The difference between the two ways of computing the signature kernel is a 'kernel trick', which makes it possible to compute the signature kernel using only inner product evaluation on the underlying state-space.

In the other notebook, `ts_classification.ipynb`, a worked out example is given on how to use signature kernels for time series classification using inter-domain sparse variational inference with inducing tensors to make computations tractable and efficient. To make the most of these examples, we also recommend to look into the [GPflow](https://github.com/GPflow/GPflow) syntax of defining kernels and GP models, a Gaussian process library that we build on.
***

## Download datasets
The benchmarks directory contains the appropriate scripts used to run the benchmarking experiments in the paper. The datasets can be downloaded from our dropbox folder using the `download_data.sh` script in the `./benchmarks/datasets` folder by running
```
cd benchmarks
bash ./datasets/download_data.sh
```
or manually by copy-pasting the dropbox url containd within the aforementioned script.

## Support
We encourage the use of this code for applications, and we aim to provide support in as many cases as possible. For further assistance or to tell us about your project, please send an email to

`csaba.toth@maths.ox.ac.uk` or `harald.oberhauser@maths.ox.ac.uk`.
