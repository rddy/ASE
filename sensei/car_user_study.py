from __future__ import division

import os
import time

import numpy as np
import gym
from pyglet.window import key as pygkey

from . import utils
from .user_models import User
from .envs import CarEnv, GuideEnv
from . import dynamics_models
from . import encoder_models

n_prac_rollouts = 2
n_eval_rollouts = 3

instructions = [
"""
Please practice driving the car using the left/right arrow keys to steer.
Once you start steering, you will not be able to stop: you will always be either turning left or right.
In order to drive straight, you will need to switch between turning left and right.
Your objective is to drive on the road (gray) while avoiding the grass (green).
You have %d practice episodes.
They will not count toward the experiment, and are intended to familiarize you with the task.""" % n_prac_rollouts,
"""In this phase, the picture on the screen will ocassionally be delayed.
You will play through %d episodes.""" % n_eval_rollouts,
"""In this phase, you will be assisted by the system.
Instead of seeing a delayed picture of the car, you will see a prediction of where the car currently is.
You will play through %d episodes.""" % n_eval_rollouts,
"",
""
]

class HumanCarUser(User):

  def __init__(self):
    self.is_human = True

    self.acc_mag = 1
    self.steer_mag = 0.25
    self.init_acc_period = 100

    self.action = None
    self.curr_step = None

  def reset(self, *args, **kwargs):
    self.curr_step = 0
    self.action = np.zeros(3)

  def __call__(self, *args, **kwargs):
    action = self.action
    if (self.curr_step % (2 * self.init_acc_period)) < self.init_acc_period:
      action[1] = self.acc_mag
    else:
      action[1] = 0.
    self.curr_step += 1
    time.sleep(0.05)
    return action

  def belief(self, *args, **kwargs):
    return np.nan

  def key_press(self, key, mod):
    a = int(key)
    if a == pygkey.LEFT:
      self.action = np.array([-self.steer_mag, 0., 0.])
    elif a == pygkey.RIGHT:
      self.action = np.array([self.steer_mag, 0., 0.])

  def key_release(self, key, mod):
    a = int(key)
    if a in [pygkey.LEFT, pygkey.RIGHT]:
      self.action[1:] = np.zeros(2)

def build_exp(sess, user_data_dir):
  encoder_model = encoder_models.load_wm_pretrained_vae(sess)
  dynamics_model = dynamics_models.load_wm_pretrained_rnn(encoder_model, sess)
  env = CarEnv(encoder_model, dynamics_model, delay=5)

  user_model = HumanCarUser()

  env.reset()
  env.render()
  env.viewer.window.on_key_press = user_model.key_press
  env.viewer.window.on_key_release = user_model.key_release

  guide_env = GuideEnv(env, user_model)

  iden_guide_policy = utils.CarGuidePolicy('iden')
  naive_guide_model = utils.CarGuidePolicy('naive')

  return {
    'iden_guide_policy': iden_guide_policy,
    'naive_guide_model': naive_guide_model,
    'guide_env': guide_env,
    'env': env
  }
