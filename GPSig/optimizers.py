from sys import modules
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer, _REGISTERED_TENSORFLOW_OPTIMIZERS
import tensorflow as tf

"""
Imports optimizers from TF contrib
"""

def _register_optimizer(name, optimizer_type):
    #if optimizer_type.__base__ is not tf.train.Optimizer and optimizer_type.__base__.__base__ is not tf.train.Optimizer \
    #    and optimizer_type.__base__.__base__.__base__ is not tf.train.Optimizer:
    #    raise ValueError('Wrong TensorFlow optimizer type passed: "{0}".'
    #                     .format(optimizer_type))
    gp_optimizer = type(name, (_TensorFlowOptimizer, ), {})
    module = modules[__name__]
    _REGISTERED_TENSORFLOW_OPTIMIZERS[name] = optimizer_type
    setattr(module, name, gp_optimizer)

for key, train_type in tf.contrib.opt.__dict__.items():
    suffix = 'Optimizer'
    if key != suffix and key.endswith(suffix):
        _register_optimizer(key, train_type)