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
    def __init__(self, model, start_time, save_dict, val_scorer = None, save_params = False, callbacks = None):
        self.model = model
        self.save_dict = save_dict
        self.val_scorer = val_scorer
        self.save_params = save_params
        self.start_time = start_time
        self.callbacks = callbacks
        
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

        if self.callbacks is not None:
            if isinstance(self.callbacks, list):
                self.save_dict[current_iter]['saved'] = []
                for i, callback in enumerate(self.callbacks):
                    self.save_dict[current_iter]['saved'].append(callback(self.model))
            else:
                self.save_dict[current_iter]['saved'] = [self.callbacks(self.model)]

        if self.val_scorer is not None:
            if isinstance(self.val_scorer, list):
                scores = []
                for i, scorer in enumerate(self.val_scorer):
                    score = scorer(self.model)
                    print('Val. {}: {:.3f}'.format(i, score), end = '\t|\t')
                    scores.append(score)
                self.save_dict[current_iter]['val'] = scores
            else:
                score = self.val_scorer(self.model)
                print('Val. : {:.3f}'.format(score), end = '\t|\t')
                self.save_dict[current_iter]['val'] = score
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

    

def optimize(model, opt, max_iter=1000, print_freq=10, save_freq=50, val_scorer=None, save_dict=None, callbacks=None,
                save_params=False, start_iter=0, stale_period=None, improve_margin=10, global_step=None, var_list=None):
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

    if save_dict is None:
        save_dict = {}
        start_iter = 0
        start_time = 0.0
    else:
        start_iter = max([x for x in save_dict.keys() if str(x).isnumeric()])
        start_time = save_dict[start_iter]['time']

    save = actions.Condition(lambda ctx: ctx.iteration % save_freq == 0 or ctx.iteration == max_iter + 1,
                                    save_snapshot(model, start_time, save_dict, val_scorer, save_params, callbacks))

    if not stale_period is None:
        check_early_stop = actions.Condition(lambda ctx: ctx.iteration % save_freq ==  0,
                        early_stop_action(model, save_dict, start_iter, stale_period, improve_margin))
        action_list += [print_iter(model), print_timing(model, start_time), save, check_early_stop]
    else:
        action_list += [print_iter(model), print_timing(model, start_time), save]

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

def optimize_with_natgrads(model, opt, natgrad_opt, max_iter=1000, print_freq=10, save_freq=50, val_scorer=None, save_dict=None, 
                save_params=False, start_iter=0, stale_period=None, improve_margin=10, global_step=None, callback=None):
    # try:
    hyperparams = []
    variationals = [(model.q_mu, model.q_sqrt)]
    
    for var in model.trainable_tensors:
        if not 'q_mu' in var.name and not 'q_sqrt' in var.name:
            hyperparams.append(var)
    
                      
    variational_step = natgrad_opt.make_optimize_action(model, var_list = variationals)
    hyperparameter_step = opt.make_optimize_action(model, var_list = hyperparams, global_step=global_step)

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
        action_list = [variational_step, hyperparameter_step, print_iter(model), print_timing(model, start_time), save, check_early_stop]
    else:
        action_list = [variational_step, hyperparameter_step, print_iter(model), print_timing(model, start_time), save]

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