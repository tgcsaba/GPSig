from sys import modules
from copy import deepcopy

import numpy as np
import tensorflow as tf

from gpflow import actions, settings
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer, _REGISTERED_TENSORFLOW_OPTIMIZERS

### Imports optimizers from TF contrib, uses same code as in GPflow

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


### GPflow actions based helpers for training models

class print_iter(actions.Action):
    def __init__(self, model):
        self.model = model
    def run(self, ctx):
        print('\rIteration {}'.format(ctx.iteration), end='')
        
class print_timing(actions.Action):
    def __init__(self, model, start_time):
        self.model = model
        self.start_time = start_time
        
    def run(self, ctx):
        print('\t|\tTime: {:.2f}'.format(ctx.time_spent + self.start_time), end='')

class print_elbo(actions.Action):
    
    def __init__(self, model):
        self.model = model
        
    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('\t|\tELBO: {:.2f}'.format(likelihood), end='')

class save_snapshot(actions.Action):
    def __init__(self, model, start_time, history, val_scorer = None, save_params = False, callbacks = None, save_best_params = False, var_list = None, lower_is_better = False, patience = None):
        self.model = model
        self.history = history
        self.val_scorer = val_scorer
        self.save_params = save_params
        self.start_time = start_time
        self.callbacks = callbacks
        self.save_best_params = save_best_params
        self.var_list = var_list
        self.lower_is_better = lower_is_better
        self.patience = patience
        
    def run(self, ctx):
        current_iter = ctx.iteration
        current_time = ctx.time_spent + self.start_time
        self.history[current_iter] = {}
        self.history[current_iter]['time'] = current_time
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        self.history[current_iter]['elbo'] = likelihood
        print('\t|\tELBO: {:.2f}'.format(likelihood), end = '')
        if self.save_params:
            save_trainables = {}
            for param in self.model.parameters:
                save_trainables[param.pathname] = ctx.session.run(param.constrained_tensor)
            self.history[current_iter]['params'] = save_trainables

        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                self.history[current_iter]['saved'] = []
                for i, callback in enumerate(self.callbacks):
                    self.history[current_iter]['saved'].append(callback(self.model))
            else:
                self.history[current_iter]['saved'] = [self.callbacks(self.model)]

        if self.val_scorer is not None:
            if isinstance(self.val_scorer, list):
                scores = []
                for i, scorer in enumerate(self.val_scorer):
                    score = scorer(self.model)
                    print('\t|\tVal. {}: {:.4f}'.format(i, score), end = '')
                    scores.append(score)
                self.history[current_iter]['val'] = scores
                score = scores[-1]
            else:
                score = self.val_scorer(self.model)
                print('\t|\tVal. : {:.4f}'.format(score), end = '')
                self.history[current_iter]['val'] = score
                scores = score
            
            if self.save_best_params:
                if 'best' in self.history:
                    if isinstance(self.history['best']['val'], list):
                        best_so_far = self.history['best']['val'][-1]
                    else:
                        best_so_far = self.history['best']['val']
                
                save_current_params = False
                if 'best' not in self.history:
                    self.history['best'] = {}
                    save_current_params = True
                elif (self.lower_is_better and score <= best_so_far) or (not self.lower_is_better and score >= best_so_far):
                    save_current_params = True 

                if save_current_params:
                    self.history['best']['iter'] = current_iter 
                    self.history['best']['time'] = current_time
                    self.history['best']['elbo'] = likelihood
                    self.history['best']['val'] = scores
                    
                    if self.var_list is None:
                        save_trainables = {}
                        for param in self.model.parameters:
                            save_trainables[param.pathname] = ctx.session.run(param.constrained_tensor)
                        self.history['best']['params'] = save_trainables
                    else:
                        self.history['best']['params'] = ctx.session.run(self.var_list)

            if self.patience is not None:
                best_iter = self.history['best']['iter']
                if current_iter - best_iter > self.patience:
                    print('\nNo improvement over validation loss has occured for {} iterations: stopping early...'.format(self.patience))
                    raise actions.Loop.Break
                        
        print()
    

def optimize(model, opt, max_iter=1000, print_freq=1, save_freq=50, val_scorer=None, history=None, callbacks=None,
                save_params=False, start_iter=0, global_step=None, var_list=None, save_best_params=False, lower_is_better=False, patience=None):
    # try:
    if isinstance(opt, list):
        assert isinstance(var_list, list)
        assert len(var_list) == len(opt) or len(var_list) + 1 == len(opt)
        action_list = []
        if len(var_list) == len(opt):
            for i, vars in enumerate(var_list):
                action_list.append(opt[i].make_optimize_action(model, global_step=global_step, var_list=vars))
        elif len(var_list) + 1 == len(opt):
            considered_vars = []
            for i, vars in enumerate(var_list):
                considered_vars = considered_vars + vars
                action_list.append(opt[i].make_optimize_action(model, global_step=global_step, var_list=vars))
            remaining_vars = []
            for var in model.trainable_tensors:
                if var not in considered_vars:
                    remaining_vars.append(var)
            action_list.append(opt[-1].make_optimize_action(model, global_step=global_step, var_list=remaining_vars))
    else:
        if var_list is None:
            action_list = [opt.make_optimize_action(model, global_step=global_step)]
        else:
            action_list = [opt.make_optimize_action(model, global_step=global_step, var_list=var_list)]

    if history is None or len([x for x in history.keys() if str(x).isnumeric()]) == 0:
        history = {}
        start_iter = 0
        start_time = 0.0
    else:
        start_iter = max([x for x in history.keys() if str(x).isnumeric()])
        start_time = history[start_iter]['time']


    if 'best' in history:
        history['best']['iter'] = start_iter
        history['best']['time'] = start_time
        if var_list is None:
            history['best']['params'] = {}
            for param in model.parameters:
                history['best']['params'][param.pathname] = model.enquire_session().run(param.constrained_tensor)
        else:
            history['best']['params'] = model.enquire_session().run(var_list)


    print_it = actions.Condition(lambda ctx: ctx.iteration % print_freq == 0 or ctx.iteration == max_iter + 1, print_iter(model))
    print_tm = actions.Condition(lambda ctx: ctx.iteration % print_freq == 0 or ctx.iteration == max_iter + 1, print_timing(model, start_time))
    
    
    save = actions.Condition(lambda ctx: ctx.iteration % save_freq == 0 or ctx.iteration == max_iter + 1,
                                    save_snapshot(model, start_time, history, val_scorer, save_params, callbacks, save_best_params, var_list, lower_is_better, patience))

    action_list += [print_it, print_tm, save]

    if start_iter == 0:
        print('-------------------------')
        print('  Starting optimization  ')
        print('-------------------------')
    else:
        print('---------------------------')
        print('  Continuing optimization  ')
        print('---------------------------')
    actions.Loop(action_list, stop = start_iter + max_iter + 1, start = start_iter + 1)()

    model.anchor(model.enquire_session())
    # except Exception as e:
    #     print('Error code: {}'.format(str(e)))
    #     print('Error occured. Halting optimisation.')

    print('\nOptimization session finished...')
        
    return history