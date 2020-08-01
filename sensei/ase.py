from __future__ import division

import os
import collections
from copy import deepcopy

import matplotlib.pyplot as plt

from . import utils

class InteractiveGuideOptimizer(object):

  def __init__(
    self,
    sess,
    env,
    guide_env
    ):

    self.sess = sess
    self.env = env
    self.guide_env = guide_env

  def format_train_rollouts(self, train_rollouts):
    rollouts = [utils.strip_guide_noops(rollout) for rollout in train_rollouts]
    rollouts = [rollout for rollout in rollouts if rollout != []]
    assert rollouts != []
    max_ep_len = max(len(rollout) for rollout in rollouts)
    rollouts = utils.vectorize_rollouts(rollouts, max_ep_len, preserve_trajs=(self.env.name != 'lander'))
    rollouts = utils.split_rollouts(rollouts, train_frac=None)
    return rollouts

  def eval_guide(
    self,
    guide_model,
    guide_perf_evals,
    train_rollouts,
    verbose=False,
    n_train_rollouts=100,
    n_eval_rollouts=100
    ):
    self.guide_env.set_val_mode(False)
    guide_eval = utils.evaluate_policy(
      self.sess,
      self.guide_env,
      guide_model,
      n_eval_rollouts=n_train_rollouts
    )
    new_train_rollouts = train_rollouts + guide_eval['rollouts']

    if n_eval_rollouts is not None:
      self.guide_env.set_val_mode(True)
      guide_eval = utils.evaluate_policy(
        self.sess,
        self.guide_env,
        guide_model,
        n_eval_rollouts=n_eval_rollouts
      )

    guide_perf = guide_eval['perf']
    guide_perf['n_train_rollouts'] = len(train_rollouts)
    if verbose:
      print('-----')
      print_guide_perf(guide_perf)
      print('-----\n')
    new_guide_perf_evals = deepcopy(guide_perf_evals)
    for k, v in guide_perf.items():
      new_guide_perf_evals[k].append(v)

    return new_guide_perf_evals, new_train_rollouts

  def run(
    self,
    guide_model,
    n_train_batches=1,
    n_rollouts_per_batch=1000,
    guide_train_kwargs={},
    verbose=False,
    init_train_rollouts=[],
    n_eval_rollouts=100
    ):

    guide_perf_evals = collections.defaultdict(list)
    train_rollouts = init_train_rollouts

    train_data = self.format_train_rollouts(train_rollouts)
    guide_model.train(train_data, **guide_train_kwargs)

    for _ in range(n_train_batches):
      guide_perf_evals, train_rollouts = self.eval_guide(
        guide_model,
        guide_perf_evals,
        train_rollouts,
        verbose=verbose,
        n_train_rollouts=n_rollouts_per_batch,
        n_eval_rollouts=n_eval_rollouts
      )
      train_data = self.format_train_rollouts(train_rollouts)
      guide_model.train(train_data, **guide_train_kwargs)

    train_log = {
      'train_rollouts': train_rollouts,
      'guide_perf_evals': guide_perf_evals
    }

    return train_log


def evaluate_baseline_guides(
  sess,
  guide_env,
  guides,
  n_eval_rollouts=1000,
  ):
  guide_evals = {}
  for guide_name, guide_policy in guides.items():
    guide_env.set_val_mode(True)
    guide_env.reset_init_order()
    guide_eval = utils.evaluate_policy(
      sess,
      guide_env,
      guide_policy,
      n_eval_rollouts=n_eval_rollouts
    )
    guide_evals[guide_name] = guide_eval
  return guide_evals


def print_guide_perf(guide_perf):
  print('\n'.join(['%s: %s' % (k, str(v)) for k, v in guide_perf.items() if not k.endswith('_t')]))
