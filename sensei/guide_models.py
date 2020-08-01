from __future__ import division

import collections
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from . import utils
from .obs_models import TabularObsModel, EncoderUserObsModel

class GridWorldGuide(object):

  def __init__(
    self,
    sess,
    env,
    obs_model,
    dynamics_model,
    q_func,
    n_obs_per_act=1,
    tabular_obs_model_kwargs={},
    prior_internal_obs_model=None,
    internal_dynamics_model=None,
    learn_internal_obs_model=False,
    init_belief_conf=(1-1e-9),
    user_init_belief_conf=(1-1e-9)
    ):

    assert env.name in ['gridworldnav', 'habitatnav']
    assert type(obs_model) == np.ndarray
    assert type(dynamics_model) == np.ndarray

    self.env = env
    self.obs_model = obs_model
    self.dynamics_model = dynamics_model
    self.n_obs_per_act = n_obs_per_act
    self.learn_internal_obs_model = learn_internal_obs_model
    self.init_belief_conf = init_belief_conf
    self.user_init_belief_conf = user_init_belief_conf

    if prior_internal_obs_model is None:
      prior_internal_obs_model = self.obs_model

    if internal_dynamics_model is None:
      internal_dynamics_model = self.dynamics_model
    self.internal_dynamics_model = internal_dynamics_model

    self.belief_logits = None
    self.user_belief_logits = None
    self.action_info = None

    if self.learn_internal_obs_model:
      self.internal_obs_model = TabularObsModel(
        internal_dynamics_model,
        prior_internal_obs_model,
        q_func,
        sess,
        self.env,
        **tabular_obs_model_kwargs
        )
    else:
      self.internal_obs_model = prior_internal_obs_model

    self.curr_step = None

  def get_action_info(self):
    return self.action_info

  def reset(self, obs, info):
    init_state = info['state']
    self.belief_logits = utils.spike_and_slab(self.init_belief_conf, self.env.n_states, init_state)
    try:
      n_ens_mem = self.internal_obs_model.n_ens_mem
    except AttributeError:
      n_ens_mem = 1
    self.user_belief_logits = np.stack([utils.spike_and_slab(self.user_init_belief_conf, self.env.n_states, init_state) for _ in range(n_ens_mem)], axis=0)
    self.curr_step = 0
    self.action_info = {'agent_obs': obs}

  def __call__(self, obs, info={}):
    prev_user_act = info.get('user_action', None)

    update_beliefs = lambda logits, obs, obs_model, dynamics_model: utils.update_beliefs(self.env, logits, obs, prev_user_act, obs_model, dynamics_model)

    self.belief_logits = utils.spike_and_slab(self.init_belief_conf, self.env.n_states, info['state'])

    if self.curr_step % self.n_obs_per_act == 0:
      if self.learn_internal_obs_model:
        internal_obs_models = self.internal_obs_model.obs_logits_eval
      else:
        internal_obs_models = self.internal_obs_model[np.newaxis, :, :]
      log_loss = lambda updated_user_belief_logits: -np.sum(np.exp(self.belief_logits[np.newaxis, :]) * updated_user_belief_logits, axis=1)
      cost = lambda x: min(1e9, np.mean(log_loss(x)))

      # all possible observations on the table
      #feasible_actions = list(range(self.env.n_obses))

      # only consider showing observations with >uniform p(o|s)
      unif_prob = 1 / self.env.n_obses
      actions = np.arange(0, self.env.n_obses, 1)
      state = info['state']
      feasible = np.mean(np.exp(internal_obs_models[:, :, state]), axis=0) >= unif_prob
      feasible_actions = actions[feasible]

      candidate_actions = [(action, update_beliefs(self.user_belief_logits, action, internal_obs_models, self.internal_dynamics_model)) for action in feasible_actions]
      costs = [(a, cost(u)) for a, u in candidate_actions]
      action, self.user_belief_logits = utils.rand_tiebreak_min(candidate_actions, lambda x: cost(x[1]))
    else:
      action = None # noop

    self.curr_step += 1
    self.action_info['agent_obs'] = obs

    return action

  def train(self, *args, **kwargs):
    assert self.learn_internal_obs_model
    self.internal_obs_model.train(*args, **kwargs)

  def save(self, *args, **kwargs):
    assert self.learn_internal_obs_model
    self.internal_obs_model.save(*args, **kwargs)

  def load(self, *args, **kwargs):
    assert self.learn_internal_obs_model
    self.internal_obs_model.load(*args, **kwargs)


class CLFGuide(object):

  def __init__(
    self,
    sess,
    env,
    encoder
    ):

    assert env.name == 'clf'

    self.env = env
    self.encoder = encoder

    self.unused_obs_idxes = None
    self.true_state = None
    self.action_info = None
    self.actions = []

  def reset(self, obs, info):
    self.encoder.reset(obs)
    self.actions = [obs]
    for obs_idx in self.env.obs_idxes[1:]:
      action = self.env.format_obs(obs_idx)
      self.actions.append(action)
      self.encoder.update(action)
    self.true_state = self.encoder.get_state()

    self.encoder.reset(obs)
    self.unused_obs_idxes = set(self.env.obs_idxes)
    self.action_info = None

  def __call__(self, obs, info={}):
    cost = lambda belief: min(1e9, np.mean(-belief))
    true_state = self.true_state
    candidate_actions = [(action_idx, self.encoder.update(self.actions[action_idx], true_state=true_state)) for action_idx in self.unused_obs_idxes]
    costs = [(a_idx, cost(u)) for a_idx, u in candidate_actions]
    action_idx, _ = utils.rand_tiebreak_min(candidate_actions, lambda x: cost(x[1]))
    action = self.actions[action_idx]
    self.unused_obs_idxes.remove(action_idx)
    self.action_info = {'action_idx': action_idx}
    self.encoder.update(action)
    return action

  def get_action_info(self):
    return self.action_info

  def train(self, *args, **kwargs):
    self.encoder.train(*args, **kwargs)

  def save(self, *args, **kwargs):
    self.encoder.save(*args, **kwargs)

  def load(self, *args, **kwargs):
    self.encoder.load(*args, **kwargs)


class LanderGuide(object):

  def __init__(
    self,
    env,
    distortion_model
    ):
    self.env = env
    self.distortion_model = distortion_model

    self.precompute_perceived_obses()

  def precompute_perceived_obses(self):
    # not as efficient as inverting the user's logistic model,
    # but simpler to implement and debug,
    # and still runs fast enough for real-time assisted control
    n_angs = 100 # discretization
    angs = np.arange(-np.pi, np.pi, 2 * np.pi / n_angs)
    self.candidate_actions = [np.zeros(self.env.n_obs_dim) for _ in angs]
    for i, ang in enumerate(angs):
      self.candidate_actions[i][4] = ang
    try:
      self.perceived_obses = self.distortion_model.compute_perceived_obses(self.candidate_actions)
    except:
      pass

  def reset(self, obs, info):
    self.distortion_model.reset(obs, info=info)

  def __call__(self, obs, info={}):
    self.distortion_model.update(obs, info=info)

    true_state = deepcopy(obs)
    beliefs_in_true_state = self.distortion_model.hypothetical_beliefs_in_true_state(self.perceived_obses, true_state)
    action_idxes = list(range(len(beliefs_in_true_state)))
    best_action_idx = max(action_idxes, key=lambda idx: beliefs_in_true_state[idx])
    best_action = self.candidate_actions[best_action_idx]
    return best_action

  def train(self, *args, **kwargs):
    self.distortion_model.train(*args, **kwargs)
    self.precompute_perceived_obses()

  def save(self, *args, **kwargs):
    self.distortion_model.save(*args, **kwargs)

  def load(self, *args, **kwargs):
    self.distortion_model.load(*args, **kwargs)
    self.precompute_perceived_obses()
