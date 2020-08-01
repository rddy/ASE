from __future__ import division

from copy import deepcopy

import numpy as np
import scipy

from . import utils

class User(object):

  def set_practice_mode(self, *args, **kwargs):
    pass

class GridWorldNavUser(User):

  def __init__(
    self,
    env,
    obs_model,
    dynamics_model,
    q_func=None,
    init_belief_conf=(1-1e-9)
    ):

    assert env.name in ['gridworldnav', 'habitatnav']
    assert type(obs_model) == np.ndarray
    assert type(dynamics_model) == np.ndarray
    assert q_func is None or type(q_func) == np.ndarray

    self.env = env
    self.obs_model = obs_model
    self.dynamics_model = dynamics_model
    self.init_belief_conf = init_belief_conf

    self.q_func = deepcopy(q_func)

    self.belief_logits = None
    self.prev_act = None
    self.q_vals = None # conditioned on goal

  def reset(self, obs, info):
    init_state = info['state']
    self.belief_logits = utils.spike_and_slab(self.init_belief_conf, self.env.n_states, init_state)
    self.prev_act = None
    goal = info['goal']
    self.q_vals = self.q_func[:, :, goal]

  def __call__(self, obs, info={}):
    self.belief_logits = utils.update_beliefs(self.env, self.belief_logits, obs, self.prev_act, self.obs_model, self.dynamics_model)
    act_logits = scipy.special.logsumexp(self.q_vals + self.belief_logits[:, np.newaxis], axis=0)
    self.prev_act = utils.sample_from_categorical(act_logits)
    return self.prev_act

  def set_prev_act(self, prev_act):
    self.prev_act = prev_act

  def belief(self, state):
    return self.belief_logits[state]


class CLFUser(User):

  def __init__(
    self,
    encoder
    ):

    self.encoder = encoder

    self.state = None
    self.logits = None

  def reset(self, obs, info):
    self.encoder.reset(obs)

  def __call__(self, obs, info={}):
    self.logits = self.encoder.update(obs)
    ens_mem_idx = np.random.choice(self.encoder.n_ens_mem)
    logits = self.logits[ens_mem_idx, :]
    action = utils.sample_from_categorical(logits)
    return action

  def belief(self, state):
    return np.mean(self.logits[:, state])


class CarUser(User):

  def __init__(self, env):
    self.env = env

    self.state = None

  def reset(self, obs, info):
    self.state = None

  def __call__(self, obs, info={}):
    self.state = self.env.extract_state(obs)
    return self.env.oracle_policy(obs)

  def belief(self, state):
    return -np.mean((state - self.state)**2)


class LanderUser(User):

  def __init__(self, env, distortion_factor=2):
    self.env = env
    self.distortion_factor = distortion_factor

    self.state = None

  def distort(self, obs):
    return utils.distort_lander_obs(obs, self.distortion_factor)

  def reset(self, obs, info):
    self.state = self.distort(obs)

  def __call__(self, obs, info={}):
    self.state = self.distort(obs)
    return self.env.oracle_policy(self.state)

  def belief(self, state):
    return -np.mean((state - self.state)**2)

