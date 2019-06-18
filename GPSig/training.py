import tensorflow as tf
from gpflow import actions, settings
import numpy as np
from copy import deepcopy

class print_iter(actions.Action):
    def __init__(self, model):
        self.model = model
    def run(self, ctx):
        print('\rIteration {}'.format(ctx.iteration), end='\t|\t')
        
class print_timing(actions.Action):
    def __init__(self, model, start_time):
        self.model = model
        self.start_time = start_time
        
    def run(self, ctx):
        print('Time: {:.2f}'.format(ctx.time_spent + self.start_time), end='\t|\t')

class print_elbo(actions.Action):
    
    def __init__(self, model):
        self.model = model
        
    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('ELBO: {:.2f}'.format(likelihood), end='\t|\t')

class save_snapshot(actions.Action):
    def __init__(self, model, start_time, save_dict, val_scorer = None, save_params = False):
        self.model = model
        self.save_dict = save_dict
        self.val_scorer = val_scorer
        self.save_params = save_params
        self.start_time = start_time
        
    def run(self, ctx):
        current_iter = ctx.iteration
        self.save_dict[current_iter] = {}
        self.save_dict[current_iter]['time'] = ctx.time_spent + self.start_time
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        self.save_dict[current_iter]['elbo'] = likelihood
        print('ELBO: {:.2f}'.format(likelihood), end = '\t|\t')
        if self.save_params:
            save_trainables = {}
            for param in self.model.trainable_parameters:
                save_trainables[param.pathname] = ctx.session.run(param.constrained_tensor)
            self.save_dict[current_iter]['params'] = save_trainables
        if self.val_scorer is not None:
            val_score = self.val_scorer(self.model)
            print('Val. accuracy: {:.3f}'.format(val_score), end = '\t|\t')
            self.save_dict[current_iter]['val'] = val_score
        print()

class early_stop_action(actions.Action):
    def __init__(self, model, save_dict, start_iter = 0, stale_period = 1000, improve_margin = 10):
        self.model = model
        self.best_iter = start_iter - 1
        self.save_dict = save_dict
        self.best_likelihood = - np.inf
        self.stale_period = stale_period
        self.improve_margin = improve_margin

    def run(self, ctx):
        current_iter = ctx.iteration
        likelihood = self.save_dict[current_iter]['elbo']
        # Update stored values if better
        if likelihood >= self.best_likelihood + self.improve_margin:
            self.best_likelihood = likelihood
            self.best_iter = current_iter
        elif current_iter - self.best_iter > self.stale_period:
            raise actions.Loop.Break

    

def optimize(model, opt, max_iter=1000, print_freq=10, save_freq=50, val_scorer=None, save_dict=None, 
                save_params=False, start_iter=0, stale_period=None, improve_margin=10, global_step=None):
    # try:
    opt_step = opt.make_optimize_action(model, global_step=global_step)
    if save_dict is None:
        save_dict = {}
        start_iter = 0
        start_time = 0.0
    else:
        start_iter = max([x for x in save_dict.keys() if str(x).isnumeric()])
        start_time = save_dict[start_iter]['time']

    save = actions.Condition(lambda ctx: ctx.iteration % save_freq == 0 or ctx.iteration == max_iter + 1,
                                    save_snapshot(model, start_time, save_dict, val_scorer, save_params))

    if not stale_period is None:
        check_early_stop = actions.Condition(lambda ctx: ctx.iteration % save_freq ==  0,
                        early_stop_action(model, save_dict, start_iter, stale_period, improve_margin))
        action_list = [opt_step, print_iter(model), print_timing(model, start_time), save, check_early_stop]
    else:
        action_list = [opt_step, print_iter(model), print_timing(model, start_time), save]

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
        
    
    return save_dict

def optimize_with_natgrads(model, opt1, opt2, max_iter = 1000, save_freq = 10, start_iter = 0,
                           val_scorer = None, save_params = False, stale_period = None, improve_margin = 10):
    
    hyperparams = []
    variationals = [(model.q_mu, model.q_sqrt)]
    
    for var in model.trainable_tensors:
        if not 'q_mu' in var.name and not 'q_sqrt' in var.name:
            hyperparams.append(var)
    
                      
    variational_step = opt2.make_optimize_action(model, var_list = variationals)
    hyperparameter_step = opt1.make_optimize_action(model, var_list = hyperparams)
    
    saved = {}

    save = actions.Condition(lambda ctx: ctx.iteration % save_freq == 0 or ctx.iteration == max_iter + 1,
                                  save_snapshot(model, saved, val_scorer, save_params))
    
    if not stale_period is None:
        check_early_stop = actions.Condition(lambda ctx: ctx.iteration % save_freq ==  0 or ctx.iteration == max_iter + 1,
                                early_stop_action(model, saved, start_iter, stale_period, improve_margin))
        action_list = [variational_step, hyperparameter_step, print_iter(model), print_timing(model), save, check_early_stop]
    else:
        action_list = [variational_step, hyperparameter_step, print_iter(model), print_timing(model), save]

    actions.Loop(action_list, stop = max_iter + 1)()
    
    model.anchor(model.enquire_session())
    
    return saved