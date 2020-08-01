from __future__ import division

import os
import time

import numpy as np
import gym
from pyglet.window import key as pygkey

from . import utils
from .obs_models import LanderObsModel
from .user_models import LanderUser, User
from .guide_models import LanderGuide
from .envs import LanderEnv, GuideEnv

n_prac_rollouts = 5
n_rollouts_per_batch = 1
n_train_batches = 5
n_eval_rollouts = 10

instructions = [
"""Please practice landing using the left and right arrow keys.
The left key fires the right thruster, which tends to rotate the lander counter-clockwise.
The right key fires the left thruster, which tends to rotate the lander clockwise.
Your objective is to prevent the lander from tilting as it descends.
Try to keep the lander level throughout the episode, not just immediately before it lands.
It does not matter where you land.
The red bar indicates the lander's tilt.
You have %d practice episodes.
They will not count toward the experiment, and are intended to familiarize you with the task.""" % n_prac_rollouts,
"""In this phase, the lander will be difficult to see.
The red bar will still indicate the lander's tilt.
You will play through %d episodes in this phase.""" % n_eval_rollouts,
"",
"""In this phase, the red bar will be adjusted
to more accurately indicate the lander's tilt.
The red bar's behavior may change from episode to episode.
You will play through %d episodes in this condition.""" % (n_rollouts_per_batch * n_train_batches),
"""In this phase, you will play through an additional %d episodes.""" % n_eval_rollouts
]

class HumanLanderUser(User):

  def __init__(self):
    self.is_human = True

    self.action = 0

  def reset(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    return self.action

  def belief(self, *args, **kwargs):
    return np.nan

  def key_press(self, key, mod):
    a = int(key)
    if a == pygkey.LEFT:
      self.action = 1
    elif a == pygkey.RIGHT:
      self.action = 3

  def key_release(self, key, mod):
    a = int(key)
    if a == pygkey.LEFT or a == pygkey.RIGHT:
      self.action = 0

def build_exp(sess, user_data_dir):
  base_env = gym.make('LunarLander-v2')
  env = LanderEnv(base_env)

  user_model = HumanLanderUser()

  env.render()
  env.base_env.unwrapped.viewer.window.on_key_press = user_model.key_press
  env.base_env.unwrapped.viewer.window.on_key_release = user_model.key_release

  guide_env = GuideEnv(env, user_model)

  iden_guide_policy = lambda obs, info: obs

  guide_model_kwargs = {
    'scope_file': os.path.join(user_data_dir, 'obs_model_scope.pkl'),
    'tf_file': os.path.join(user_data_dir, 'obs_model.tf'),
    'n_hidden': 32,
    'n_layers': 0,
    'policy': env.tf_policy
  }

  guide_train_kwargs = {
    'iterations': 5000,
    'ftol': 1e-6,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'val_update_freq': 100,
    'verbose': True,
    'show_plots': False
  }

  obs_model_for_guide = LanderObsModel(sess, env, **guide_model_kwargs)

  guide_model = LanderGuide(env, obs_model_for_guide)

  return {
    'iden_guide_policy': iden_guide_policy,
    'guide_env': guide_env,
    'env': env,
    'guide_model': guide_model,
    'guide_train_kwargs': guide_train_kwargs
  }
