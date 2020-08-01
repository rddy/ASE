from __future__ import division

import os
import types

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from . import gw_viz
from . import utils
from .envs import GridWorldNavEnv, GuideEnv
from .user_models import GridWorldNavUser, User
from .guide_models import GridWorldGuide

n_prac_rollouts = 3
n_rollouts_per_batch = 1
n_train_batches = 0
n_eval_rollouts = 5

instructions = [
"""Your objective is to navigate to the goal position and orientation (highlighted in green) as fast as possible.
Each circle corresponds to a position in the 5x5 grid.
Each triangle corresponds to one of four orientations for each position.
There are no obstacles, so you can move freely.
Walls at the four edges of the map prevent you from going out of bounds.
You will not be able to directly see your current position and orientation.
At each timestep, you will be informed that a particular object is visible.
This "object guidance" information is intended to help you infer your current position and orientation.
Some objects have known locations, which are highlighted in orange circles.
Other objects have unknown locations, in which case no circles are highlighted.
An object is only visible from neighboring positions with orientations facing the object.
Hence, if you are informed that an object is visible and its locations are known,
you may infer that you are currently in one of the positions and orientations facing a highlighted orange circle.
Note that the wall surrounding the map is also an object.
When you are informed that the wall is visible, the boundary will be highlighted.
When prompted by 'What is the current position and orientation?: ', please type a space-delimited list of all positions and orientations you believe you might be in, and hit enter.
For example, if you think it might be 18, 25, or 42, then type '18 25 42' and hit enter.
If you have no idea what the current state might be, then hit enter without typing anything.
This information is meant to be a scratchpad to help you remember where you might have been in the past.
It will not be used by the system in any way.
When prompted by 'What action would you like to take (a/w/s/d)?: ', please type a single action and hit enter.
Action 'a' rotates you 90 degrees counter-clockwise.
Action 'w' moves you forward by one position. If you are at the boundary and facing outward, you will stay in place.
Action 'd' rotates you 90 degrees clockwise.
Action 's' does nothing, keeping you in your current position and orientation.
You have %d practice episodes.
They will not count toward the experiment, and are intended to familiarize you with the task.
During this practice phase, we will highlight all possible current positions and orientations in gray/black,
in order to illustrate how you can infer your current position and orientation
from object guidance and your recent movements.
The episode ends when you reach the goal or run out of time (25 actions).""" % n_prac_rollouts,
"""In this phase, you will play through %d episodes.
You will no longer see all possible current positions and orientations highlighted in gray.
Instead, you will need to infer your current position and orientation on your own,
using object guidance and your recent movements.""" % n_eval_rollouts,
"""In this phase, the system will attempt to provide more informative object guidance.
You will play through %d episodes.""" % n_eval_rollouts,
"""In this phase, the system will attempt to learn to provide more informative object guidance.
You will play through %d episodes.""" % (n_rollouts_per_batch * n_train_batches),
"""In this phase, you will play through an additional %d episodes.""" % n_eval_rollouts
]

class HumanGridWorldUser(User):

  def __init__(self, env, *args, **kwargs):
    self.env = env

    self.user_model = GridWorldNavUser(env, *args, **kwargs)

    self.is_human = True
    self.practice_mode = True

    self.action_of_raw = {
      'a': 0,
      'd': 1,
      'w': 2,
      's': 3
    }

    self.belief_logits = None
    self.curr_timestep = None

  def set_practice_mode(self, mode):
    self.practice_mode = mode

  def reset(self, obs, info):
    self.curr_timestep = 0
    if self.practice_mode:
      self.user_model.reset(obs, info)
    self.belief_logits = np.log(np.ones(self.env.n_states) / self.env.n_states)

  def sim_belief(self, state):
    assert self.practice_mode
    return self.user_model.belief(state)

  def belief(self, state):
    return self.belief_logits[state]

  def __call__(self, obs):
    print('-----')
    print('Time: %d of %d steps' % (self.curr_timestep, self.env.max_ep_len))

    if self.practice_mode:
      self.user_model(obs)
      self.env.render()

    # the name of the object is just a small comfort for the human participant,
    # and does not have any connection to the actual environment.
    # the orange circles in the display contain all the object guidance information
    obs_is_wall = ((obs % (self.env.n_objes_per_set+1)) == self.env.n_objes_per_set)
    obs_is_inf = self.env.is_obs_informative(obs)
    if obs_is_wall and obs_is_inf:
      obs_name = 'wall'
    else:
      if obs_is_inf:
        obs_names = ['plant', 'sofa', 'table', 'desk', 'lamp', 'painting', 'computer', 'fireplace']
      else:
        obs_names = ['painting', 'mirror', 'door', 'carpet', 'litterbox', 'coffee machine', 'microwave', 'refrigerator']
      obs_name = np.random.choice(obs_names)

    print('Object guidance: there is a %s directly in front of you.' % obs_name)
    if obs_is_inf:
      print('There are %ss located at the highlighted orange circles.' % obs_name)
    else:
      print('Unfortunately, you do not know where the %s is located.' % obs_name)

    state_prompt = "What is the current position and orientation (e.g., '18 25 42' or '')?: "
    raw_states = input(state_prompt)
    raw_states = raw_states.strip()
    valid_input = False
    while not valid_input:
      try:
        if raw_states != '':
          current_states = [int(x) for x in raw_states.split(' ')]
          assert all(0 <= x and x < self.env.n_states for x in current_states)
        else:
          current_states = []
        valid_input = True
      except:
        valid_input = False
        raw_states = input('Invalid position and orientation. %s' % state_prompt)
    self.belief_logits[:] = -1e6
    if current_states != []:
      self.belief_logits[current_states] = np.log(1. / len(current_states))
    self.belief_logits -= scipy.special.logsumexp(self.belief_logits)

    action_prompt = 'What action would you like to take (a/w/s/d)?: '
    raw_act = input(action_prompt)
    raw_act = raw_act.strip()
    while raw_act not in self.action_of_raw:
      raw_act = input('Invalid action. %s' % action_prompt)
    print('-----\n')

    self.curr_timestep += 1
    action = self.action_of_raw[raw_act]
    if self.practice_mode:
      self.user_model.set_prev_act(action)
    return action

def build_exp(
  sess,
  user_data_dir,
  gw_size=5
  ):

  n_goals = gw_size**2
  n_states = 4*gw_size**2
  n_objes_per_set = gw_size**2
  n_obj_instances_of_set = [1, 2, 1]
  n_obj_sets = len(n_obj_instances_of_set)
  n_objes = n_objes_per_set*n_obj_sets
  n_obses = n_objes + n_obj_sets
  ground_truth = np.zeros((n_obses, n_states))
  ticks = np.arange(0, gw_size, 1)
  poses = utils.enumerate_gw_poses(ticks, ticks)
  poses_of_obs = [[] for _ in range(n_obses)]
  for obj_set in range(n_obj_sets):
    for obj in range(n_objes_per_set):
      obs = obj_set*(n_objes_per_set+1)+obj
      obj_poses = [poses[obj*4]]
      for i in range(1, n_obj_instances_of_set[obj_set]):
        obj_poses.append(poses[np.random.choice(list(range(n_objes_per_set)))*4])
      poses_of_obs[obs] = obj_poses
      for obj_pos in obj_poses:
        for state, user_pos in enumerate(poses):
          conds = []
          conds.append(obj_pos[0] == user_pos[0] and obj_pos[1] == user_pos[1] + 1 and user_pos[2] == 2)
          conds.append(obj_pos[0] == user_pos[0] and obj_pos[1] == user_pos[1] - 1 and user_pos[2] == 0)
          conds.append(obj_pos[1] == user_pos[1] and obj_pos[0] == user_pos[0] + 1 and user_pos[2] == 3)
          conds.append(obj_pos[1] == user_pos[1] and obj_pos[0] == user_pos[0] - 1 and user_pos[2] == 1)
          if any(conds):
            ground_truth[obs, state] = 1

  for obj_set in range(n_obj_sets):
    obs = obj_set*(n_objes_per_set+1)+n_objes_per_set
    for state, user_pos in enumerate(poses):
      conds = []
      conds.append(user_pos[0] == 0 and user_pos[2] == 1)
      conds.append(user_pos[0] == gw_size - 1 and user_pos[2] == 3)
      conds.append(user_pos[1] == 0 and user_pos[2] == 0)
      conds.append(user_pos[1] == gw_size - 1 and user_pos[2] == 2)
      if any(conds):
        ground_truth[obs, state] = 1

  ground_truth = utils.smooth_matrix(ground_truth, n_states, eps=1e-6)
  ground_truth_obs_model = np.log(ground_truth)

  max_ep_len = gw_size**2
  env = GridWorldNavEnv(
    gw_size=gw_size,
    n_goals=n_goals,
    max_ep_len=max_ep_len,
    ground_truth_obs_model=ground_truth_obs_model
  )

  env.n_objes_per_set = n_objes_per_set
  env.n_obj_sets = n_obj_sets
  def is_obs_informative(self, obs):
    n_uninf_obses = self.n_obses // self.n_obj_sets
    return obs >= n_uninf_obses
  env.is_obs_informative = types.MethodType(is_obs_informative, env)

  env.practice = False
  def set_practice_mode(self, mode):
    self.practice = mode
  env.set_practice_mode = types.MethodType(set_practice_mode, env)

  masked_obses = np.arange(0, env.n_obses // env.n_obj_sets, 1)
  internal = np.exp(env.ground_truth_obs_model)
  obs_weights = np.ones(env.n_obses)
  for obs in masked_obses:
    obs_weights[obs] = 1e-6
  internal = utils.smooth_matrix(internal, env.n_obses, eps=(1-obs_weights[:, np.newaxis]))
  internal = np.log(internal)
  internal_obs_model = internal

  user_init_belief_conf = 1e-9

  user_model = HumanGridWorldUser(
    env,
    internal_obs_model,
    env.make_dynamics_model(eps=1e-6),
    q_func=env.Q,
    init_belief_conf=user_init_belief_conf
  )
  guide_env = GuideEnv(env, user_model, n_obs_per_act=1)

  def visualize(self, state_colors=None):
    action_of_ang = lambda ang: (ang + 3) % 4 + 1
    gw_size = int(np.sqrt(self.n_states // 4))
    R = np.zeros((gw_size, gw_size, 5))
    highlight_wall = False
    if self.prev_shown_obs is not None:
      if user_model.practice_mode:
        for state in range(self.n_states):
          x, y, ang = self.pos_from_state(state)
          action = action_of_ang(ang)
          R[x, y, action] = np.exp(user_model.sim_belief(state))
      if self.is_obs_informative(self.prev_shown_obs):
        if self.prev_shown_obs % (n_objes_per_set+1) == n_objes_per_set:
          highlight_wall = True
        else:
          for pos in poses_of_obs[self.prev_shown_obs]:
            x, y = pos[:2]
            R[x, y, 0] = 2
    x, y, ang = self.goal
    action = action_of_ang(ang)
    R[x, y, action] = 3
    reward_arrays = {'': R}
    fig = gw_viz.plot_gridworld_rewards(self, highlight_wall, reward_arrays, ncols=1)

    canvas = FigureCanvas(fig)
    agg = canvas.switch_backends(FigureCanvas)
    agg.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img
  env.visualize = types.MethodType(visualize, env)

  iden_guide_policy = lambda obs, info: obs
  iden_guide_policy = utils.StutteredPolicy(iden_guide_policy, guide_env.n_obs_per_act)

  init_belief_conf = 1-1e-9
  dynamics_model = env.make_dynamics_model(eps=1e-9)
  internal_dynamics_model = env.make_dynamics_model(eps=0.1)

  naive_guide_model = GridWorldGuide(
    sess,
    env,
    ground_truth_obs_model,
    dynamics_model,
    env.Q,
    n_obs_per_act=guide_env.n_obs_per_act,
    internal_dynamics_model=internal_dynamics_model,
    prior_internal_obs_model=ground_truth_obs_model,
    learn_internal_obs_model=False,
    init_belief_conf=init_belief_conf,
    user_init_belief_conf=user_init_belief_conf
  )

  tabular_obs_model_kwargs = {
    'scope_file': os.path.join(user_data_dir, 'guide_scope.pkl'),
    'tf_file': os.path.join(user_data_dir, 'guide.tf'),
    'user_init_belief_conf': user_init_belief_conf,
    'obs_params_only': True,
    'prior_coeff': 0.,
    'warm_start': False
  }

  guide_train_kwargs = {
    'iterations': 1000,
    'ftol': 1e-6,
    'batch_size': 32,
    'learning_rate': 1e-2,
    'val_update_freq': 100,
    'verbose': True,
    'show_plots': False
  }

  guide_model = GridWorldGuide(
    sess,
    env,
    env.ground_truth_obs_model,
    dynamics_model,
    env.Q,
    n_obs_per_act=guide_env.n_obs_per_act,
    prior_internal_obs_model=env.ground_truth_obs_model,
    internal_dynamics_model=internal_dynamics_model,
    tabular_obs_model_kwargs=tabular_obs_model_kwargs,
    learn_internal_obs_model=True,
    init_belief_conf=init_belief_conf,
    user_init_belief_conf=user_init_belief_conf
  )

  return {
    'iden_guide_policy': iden_guide_policy,
    'naive_guide_model': naive_guide_model,
    'guide_env': guide_env,
    'env': env,
    'guide_model': guide_model,
    'guide_train_kwargs': guide_train_kwargs
  }
