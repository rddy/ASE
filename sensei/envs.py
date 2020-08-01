from __future__ import division

from copy import deepcopy
import os
import pickle
import collections
import sys
import types
import queue
import time

import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
import sklearn.neighbors
import habitat
import scipy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import tensorflow as tf

from . import utils
from . import dynamics_models
from . import obs_models
from . import qlearn

sys.path.append(utils.wm_dir)
import model as carracing_model

class Env(gym.Env):
  metadata = {
    'render.modes': ['human']
  }

  def render(self, mode='human', close=False, **kwargs):
    if close:
      if self.viewer is not None:
        self.viewer.close()
        self.viewer = None
      return

    if self.viewer is None:
      self.viewer = rendering.SimpleImageViewer(maxwidth=(10 if self.name == 'carracing' else 1))

    img = self.visualize(**kwargs)
    self.viewer.imshow(img)

    #plt.imshow(img)
    #plt.savefig(os.path.join(utils.gw_human_data_dir, 'figures', 'gw-screenshot.pdf'), bbox_inches='tight')
    #plt.close()

  def show(self, obs, info={}):
    pass

  def reset_init_order(self):
    pass

  def set_val_mode(self, val_mode):
    pass

  def set_practice_mode(self, mode):
    pass


class GridWorldNavEnv(Env):

  def __init__(self, navigable_poses=None, **init_kwargs):
    self.init_kwargs = init_kwargs
    if navigable_poses is None:
      ticks = np.arange(0, self.init_kwargs['gw_size'], 1)
      navigable_poses = utils.enumerate_gw_poses(ticks, ticks)
    self.init_with_navigable_poses(navigable_poses, **self.init_kwargs)

  def init_with_navigable_poses(
    self,
    navigable_poses,
    goals=None,
    n_goals=None,
    gw_size=7,
    succ_rew_bonus=10,
    time_penalty=0,
    max_ep_len=50,
    max_hop_dist=0,
    max_lat_dist=0,
    n_obses=None,
    ground_truth_obs_model=None
    ):

    if goals is not None and n_goals is not None:
      assert n_goals == goals.shape[0]

    if n_obses is None and ground_truth_obs_model is not None:
      n_obses = ground_truth_obs_model.shape[0]

    if ground_truth_obs_model is not None:
      assert n_obses == ground_truth_obs_model.shape[0]

    if ground_truth_obs_model is None:
      raise ValueError

    self.name = 'gridworldnav'

    self.gw_size = gw_size
    self.n_actions = 4
    self.succ_rew_bonus = succ_rew_bonus
    self.time_penalty = time_penalty
    self.max_ep_len = max_ep_len
    self.navigable_poses = navigable_poses
    self.max_hop_dist = max_hop_dist
    self.max_lat_dist = max_lat_dist
    self.noop_action = 3

    self.n_states = self.navigable_poses.shape[0]
    if n_obses is None and ground_truth_obs_model is None:
      n_obses = self.n_states
    self.n_obses = n_obses

    self.ground_truth_obs_model = ground_truth_obs_model

    self.pos_of_state = {i: self.navigable_poses[i, :] for i in range(self.navigable_poses.shape[0])}
    self.state_of_pos = {tuple(self.navigable_poses[i, :]): i for i in range(self.navigable_poses.shape[0])}

    if goals is None:
      if n_goals is None:
        goals = self.navigable_poses
      else:
        idxes = np.random.choice(self.navigable_poses.shape[0], size=n_goals, replace=False)
        goals = self.navigable_poses[idxes, :]
    self.goals = goals
    self.n_goals = self.goals.shape[0]
    self.goal_states = [self.state_from_pos(self.goals[g, :]) for g in range(self.n_goals)]

    self.pos = None
    self.curr_step = None
    self.viewer = None
    self.prev_state = None
    self.goal_idx_idx = None
    self.init_state_idx_idx = None
    self.goal = None
    self.init_state = None
    self.prev_shown_obs = None
    self.prev_agent_obs = None
    self.fast_next_state = {}

    self.omit_vars = {}

    self.T = self.make_dynamics_model()
    self.R = self.make_reward_model()
    self.Q = self.make_q_func()
    self.oracle_dists, self.oracle_preds = self.initialize_oracle()

    self.goal_idxes = list(range(len(self.goal_states)))
    np.random.shuffle(self.goal_idxes)
    self.init_state_idxes = list(range(self.n_states))
    np.random.shuffle(self.init_state_idxes)
    self.reset_init_order()

  def show(self, obs, info={}):
    self.prev_shown_obs = obs
    self.prev_agent_obs = info.get('agent_obs', None)

  def load_from_cache(self, cache_path=None):
    if cache_path is None:
      cache_path = self.cache_path
    with open(cache_path, 'rb') as f:
      self.__dict__.update(pickle.load(f))

  def save_to_cache(self, cache_path=None):
    if cache_path is None:
      cache_path = self.cache_path
    out = {k: v for k, v in self.__dict__.items() if k not in self.omit_vars}
    with open(cache_path, 'wb') as f:
      pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

  def initialize_oracle(self):
    graph = -self.T.max(axis=2)
    graph = csr_matrix(graph)
    return dijkstra(csgraph=graph, directed=True, indices=self.goal_states, return_predecessors=True)

  def make_reward_model(self):
    R = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_goals))
    states = np.tile(np.arange(0, self.n_states, 1), self.n_goals).reshape((self.n_goals, self.n_states)).T
    goals = np.tile(self.goal_states, self.n_states).reshape((self.n_states, self.n_goals))
    mask = (states == goals).astype(float)
    R[:, :, :, :] = mask * self.succ_rew_bonus + (1 - mask) * self.time_penalty
    return R

  def make_dynamics_model(self, eps=1e-9):
    T = np.zeros((self.n_states, self.n_states, self.n_actions)) # p(s'|s,a)
    for s in range(self.n_states):
      for a in range(self.n_actions):
        neighb = self.next_state(s, a)
        T[neighb if neighb is not None else s, s, a] = 1
    T = utils.smooth_matrix(T, self.n_states, eps=eps)
    T = np.log(T)
    return T

  def make_q_func(self, dynamics_model=None, verbose=False):
    if dynamics_model is None:
      dynamics_model = self.T
    Q = np.zeros((self.n_states, self.n_actions, self.n_goals)) # s,a,g
    for g in range(self.n_goals):
      Q[:, :, g] = qlearn.tabular_soft_q_iter(
        self.R[:, :, :, g],
        dynamics_model,
        gamma=1.,
        verbose=verbose,
        maxiter=self.max_ep_len
      )
    return Q

  def _state(self):
    return self.state_from_pos(self.pos)

  def _obs(self):
    state = self._state()
    return utils.sample_from_categorical(self.ground_truth_obs_model[:, state])

  def next_state(self, state, action):
    if (state, action) not in self.fast_next_state:
      pos = self.pos_from_state(state)
      next_pos = self.next_pos(pos, action)
      next_state = self.state_from_pos(next_pos)

      if next_state is None and action == 2:
        ang = pos[2]
        if ang == 0: # left
          mins = np.array([-self.max_lat_dist, -self.max_hop_dist])
          maxs = np.array([self.max_lat_dist, 0])
        elif ang == 1: # up
          mins = np.array([-self.max_hop_dist, -self.max_lat_dist])
          maxs = np.array([0, self.max_lat_dist])
        elif ang == 2: # right
          mins = np.array([-self.max_lat_dist, 0])
          maxs = np.array([self.max_lat_dist, self.max_hop_dist])
        elif ang == 3: # down
          mins = np.array([0, -self.max_lat_dist])
          maxs = np.array([self.max_hop_dist, self.max_lat_dist])
        else:
          raise ValueError
        mins = mins + next_pos[:2]
        maxs = maxs + next_pos[:2]
        min_dist = None
        min_state = None
        for s in range(self.n_states):
          pos = self.navigable_poses[s, :2]
          next_ang = self.navigable_poses[s, 2]
          if next_ang != ang or (pos < mins).any() or (pos >= maxs).any():
            continue
          dist = np.linalg.norm(pos - next_pos[:2])
          if min_dist is None or dist < min_dist:
            min_dist = dist
            min_state = s
          next_state = min_state

      self.fast_next_state[(state, action)] = next_state

    return self.fast_next_state[(state, action)]

  def delta_of_ang(self, ang):
    delta = np.zeros(2)
    if ang == 0: # left
      delta[1] = -1
    elif ang == 1: # up
      delta[0] = -1
    elif ang == 2: # right
      delta[1] = +1
    elif ang == 3: # down
      delta[0] = +1
    else:
      raise ValueError
    return delta

  def next_pos(self, pos, action):
    next_pos = deepcopy(pos)
    if action == 0: # turn left
      next_pos[2] -= 1
      if next_pos[2] < 0:
        next_pos[2] += 4
    elif action == 1: # turn right
      next_pos[2] = (next_pos[2] + 1) % 4
    elif action == 2: # move forward
      ang = pos[2]
      delta = self.delta_of_ang(ang)
      next_pos[:2] = next_pos[:2] + delta
    elif action == 3: # noop
      pass
    else:
      raise ValueError('invalid action')
    return next_pos

  def pos_from_state(self, state):
    return self.pos_of_state.get(state, None)

  def state_from_pos(self, pos):
    return self.state_of_pos.get(tuple(pos), None)

  def reward(self, prev_state, action, state):
    pos = self.pos_from_state(state)
    if (pos == self.goal).all():
      return self.succ_rew_bonus
    else:
      return self.time_penalty

  def dist_to_goal(self, state):
    return self.oracle_dists[self.goal_idx, state]

  def oracle_policy(self, obs):
    state = self.prev_state
    opt_next_state = self.oracle_preds[self.goal_idx, state]
    for action in range(self.n_actions):
      next_state = self.next_state(state, action)
      if next_state == opt_next_state:
        return action
    raise ValueError

  def is_navigable(self, pos):
    return tuple(pos) in self.state_of_pos

  def step(self, action, action_info={}):
    next_pos = self.pos_from_state(self.next_state(self.prev_state, action))
    if next_pos is not None and self.is_navigable(next_pos):
      self.pos = next_pos

    self.curr_step += 1
    succ = (self.pos == self.goal).all()
    oot = self.curr_step >= self.max_ep_len

    obs = self._obs()
    state = self._state()
    r = self.reward(self.prev_state, action, state)
    done = oot or succ
    info = {
      'state': state
    }
    if done:
      info['succ'] = succ
    info['dist_to_goal'] = self.dist_to_goal(state) / self.dist_to_goal(self.init_state)
    if self.curr_step == 1:
      info['goal'] = self.goal_idx
    self.prev_state = state
    return obs, r, done, info

  def reset_init_order(self):
    self.goal_idx_idx = 0
    self.init_state_idx_idx = 0

  def reset(self):
    self.goal_idx = self.goal_idxes[self.goal_idx_idx]
    goal_state = self.goal_states[self.goal_idx]
    self.goal = self.goals[self.goal_idx, :]

    states = np.arange(0, self.n_states, 1)
    states = states[np.not_equal(states, goal_state) & np.less_equal(self.oracle_dists[self.goal_idx, :], 1e-6)]
    init_state_idx = self.init_state_idxes[self.init_state_idx_idx]
    state_idx = init_state_idx % len(states)
    self.init_state = states[state_idx]
    self.pos = self.pos_from_state(self.init_state)

    self.goal_idx_idx = (self.goal_idx_idx + 1) % len(self.goal_idxes)
    self.init_state_idx_idx = (self.init_state_idx_idx + 1) % len(self.init_state_idxes)

    self.prev_shown_obs = None
    self.prev_agent_obs = None
    self.curr_step = 0
    self.prev_state = self._state()
    obs = self._obs()
    info = {'state': self.prev_state, 'goal': self.goal_idx}
    return obs, info

  def visualize(self, state_colors=None):
    fig = plt.figure(figsize=(8,6), dpi=80)
    canvas = FigureCanvas(fig)

    color_kwargs = {}
    if state_colors is None:
      color_kwargs['color'] = 'gray'
    else:
      state_colors = np.maximum(0.1, state_colors)
      color_kwargs['c'] = state_colors
    plt.scatter(self.navigable_poses[:, 0], self.navigable_poses[:, 1], linewidth=0, alpha=1, **color_kwargs, cmap=mpl.cm.binary, norm=mpl.colors.Normalize(vmin=0.,vmax=1.))
    plt.scatter([self.goal[0]], [self.goal[1]], color='green', linewidth=0, alpha=1, marker='*', s=1000)

    pos = self.pos_from_state(np.argmax(state_colors))
    delta = self.delta_of_ang(pos[2])
    plt.arrow(pos[0], pos[1], delta[0], delta[1], width=1, color='black')

    delta = self.delta_of_ang(self.pos[2])
    plt.arrow(self.pos[0], self.pos[1], delta[0], delta[1], width=1, color='teal')

    plt.xlim([-1, self.gw_size+1])
    plt.ylim([-1, self.gw_size+1])
    plt.axis('off')

    agg = canvas.switch_backends(FigureCanvas)
    agg.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return img

class HabitatNavEnv(GridWorldNavEnv):

  def __init__(
    self,
    *args,
    env_id='uNb9QFRL6hY',
    dataset='mp3d',
    bbox_xy=None,
    bbox_z=None,
    n_navigable_poses=100000,
    radius=1e-1,
    verbose=True,
    cache_path=None,
    use_cache=True,
    render_mode=True,
    **kwargs
    ):

    self.init_kwargs = kwargs
    self.gw_size = kwargs['gw_size']

    self.env_id = env_id
    self.dataset = dataset
    self.base_env = self.make_base_env(self.env_id, dataset)
    self.render_mode = render_mode

    self.base_env._dataset = None
    self.base_env._episodes = []

    if cache_path is None:
      cache_path = os.path.join(utils.hab_data_dir, 'cache', '%s.pkl' % self.env_id)
    self.cache_path = cache_path

    if os.path.exists(self.cache_path) and use_cache:
      self.load_from_cache(self.cache_path)
    else:
      self.navigable_samples = np.array([self.base_env._sim.sample_navigable_point() for _ in range(n_navigable_poses)])
      self.navigable_samples = self.navigable_samples[:, [0, 2, 1]]
      navigable_mask = np.ones(self.navigable_samples.shape[0]).astype(bool)
      if bbox_z is not None:
        min_z, max_z = bbox_z
        navigable_mask &= np.greater(self.navigable_samples[:, 2], min_z) & np.less(self.navigable_samples[:, 2], max_z)
      if bbox_xy is not None:
        mins, maxs = bbox_xy
        navigable_mask &= np.greater(self.navigable_samples[:, :2], mins).all(axis=1) & np.less(self.navigable_samples[:, :2], maxs).all(axis=1)
      self.navigable_samples = self.navigable_samples[navigable_mask, :]
      tree = sklearn.neighbors.KDTree(self.navigable_samples[:, :2])

      mins = self.navigable_samples[:, :2].min(axis=0)
      maxs = self.navigable_samples[:, :2].max(axis=0)
      self.gw_ticks, self.gw_tick_size = utils.gw_ticks(mins, maxs, self.gw_size)
      poses = utils.enumerate_gw_poses(self.gw_ticks[:, 0], self.gw_ticks[:, 1])
      counts = tree.query_radius(poses[:, :2], r=radius, count_only=True)
      navigable_mask = np.greater(counts, 0)
      navigable_poses = poses[navigable_mask, :]
      _, idxes = tree.query(poses[:, :2])
      idxes = np.squeeze(np.array(idxes))
      self.nearest_navigable_samples = self.navigable_samples[idxes, :]
      self.nearest_navigable_samples = self.nearest_navigable_samples[navigable_mask, :]

      self.idx_of_unscaled_pos = {}
      for i in range(navigable_poses.shape[0]):
        unscaled_pos = self.unscale_pos(navigable_poses[i, :])
        self.idx_of_unscaled_pos[tuple(unscaled_pos)] = i

      if verbose:
        plt.scatter(self.navigable_samples[:, 0], self.navigable_samples[:, 1], color='gray', alpha=0.5, label='Navigable', s=1)
        plt.scatter(navigable_poses[:, 0], navigable_poses[:, 1], color='orange', alpha=0.5, label='Grid', s=1)
        plt.legend(loc='best')
        plt.show()

      scene = self.base_env.sim.semantic_annotations()
      cat_of_obj_idx = {idx: (obj.category.name() if obj is not None else '') for idx, obj in enumerate(scene.objects)}
      cats = set(cat_of_obj_idx.values())
      cats = [cat for cat in cats if cat != '']
      cats = sorted(cats)
      idx_of_cat = {x: i for i, x in enumerate(cats)}

      self.obj_idxes_of_cat = collections.defaultdict(list)
      for obj_idx, cat in cat_of_obj_idx.items():
        self.obj_idxes_of_cat[cat].append(obj_idx)

      n_objects = len(idx_of_cat)
      n_states = navigable_poses.shape[0]
      ground_truth_obs_model = np.zeros((n_objects, n_states))
      for s in range(n_states):
        agent_position = self.nearest_navigable_samples[s, :]
        agent_position = agent_position[[0, 2, 1]]
        ang = int(navigable_poses[s, 2])
        rotation = utils.quat_of_ang(ang)
        agent_obs = self.base_env._sim.get_observations_at(agent_position, rotation)['semantic']
        counter = collections.Counter(agent_obs.ravel())
        for obj_idx, pix_count in counter.items():
          if obj_idx not in cat_of_obj_idx:
            continue
          cat = cat_of_obj_idx[obj_idx]
          if cat not in idx_of_cat:
            continue
          cat_idx = idx_of_cat[cat]
          ground_truth_obs_model[cat_idx, s] += pix_count
      mask = (ground_truth_obs_model > 0).any(axis=1)
      idxes = np.arange(0, n_objects, 1)[mask]
      ground_truth_obs_model = ground_truth_obs_model[idxes, :]
      self.str_of_obs = {i: cats[idx] for i, idx in enumerate(idxes)}
      ground_truth_obs_model /= ground_truth_obs_model.sum(axis=0)
      n_objects = ground_truth_obs_model.shape[0]
      ground_truth_obs_model = utils.smooth_matrix(ground_truth_obs_model, n_objects)
      ground_truth_obs_model = np.log(ground_truth_obs_model)

      self.init_kwargs['n_obses'] = n_objects
      self.init_kwargs['ground_truth_obs_model'] = ground_truth_obs_model
      self.navigable_poses = self.unscale_pos(navigable_poses)

      self.init_with_navigable_poses(self.navigable_poses, **self.init_kwargs)

      self.name = 'habitatnav'

      self.omit_vars = {'base_env'}
      self.save_to_cache(self.cache_path)

    if not self.render_mode:
      self.base_env = None
      self.navigable_samples = None

    with open(utils.hab_layout_path, 'rb') as f:
      self.samples, self.poses_of_cat = pickle.load(f)

  def viz_cat_locs(self, cats, cat_colors):
    fig = plt.figure(figsize=(8,6), dpi=80)
    canvas = FigureCanvas(fig)

    plt.axis('off')
    plt.scatter(self.samples[:, 0], self.samples[:, 1], color='lightgray', alpha=0.25)
    for cat, cat_color in zip(cats, cat_colors):
      poses = self.poses_of_cat[cat]
      plt.scatter(poses[:, 0], poses[:, 1], color=cat_color, s=1)
    min_x = np.min(self.samples[:, 0])
    max_x = np.max(self.samples[:, 0])
    min_y = np.min(self.samples[:, 1])
    max_y = np.max(self.samples[:, 1])
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    if self.prev_shown_obs is not None:
      text = self.str_of_obs[self.prev_shown_obs]
      delta = (max_x-min_x)/2 if self.prev_agent_obs is not None else 0
      y = max_y+1 if self.prev_agent_obs is None else min_y-4
      plt.text(min_x+delta, y, text, fontdict={'size': 40, 'color': 'orange' if self.prev_agent_obs is not None else 'gray'})
    if self.prev_agent_obs is not None:
      text = self.str_of_obs[self.prev_agent_obs]
      plt.text(min_x, min_y-4, text, fontdict={'size': 40, 'color': 'gray'})

    agg = canvas.switch_backends(FigureCanvas)
    agg.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(agg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return img

  def set_render_mode(self, render_mode):
    if self.base_env is None and render_mode:
      raise ValueError
    if self.render_mode and not render_mode:
      self.base_env = None
      self.navigable_samples = None
    self.render_mode = render_mode

  def make_base_env(self, env_id, dataset):
    habitat_dir = os.path.join('/Users', 'reddy', 'habitat-api')
    scenes_dir = os.path.join(habitat_dir, 'data', 'scene_datasets')
    path_to_cfg = os.path.join(habitat_dir, 'configs', 'datasets', 'pointnav', '%s.yaml' % dataset)
    data_path = os.path.join(habitat_dir, 'data', 'datasets', 'pointnav', dataset, 'v1', '{split}', '{split}.json.gz')
    if dataset == 'mp3d':
      scene_path = os.path.join(scenes_dir, dataset, env_id, '%s.glb' % env_id)
    elif dataset == 'gibson':
      scene_path = os.path.join(scenes_dir, dataset, '%s.glb' % env_id)
    else:
      raise ValueError

    config = habitat.get_config(path_to_cfg)
    config.defrost()
    config.DATASET.SPLIT = 'val_mini'
    config.DATASET.SCENES_DIR = scenes_dir
    config.DATASET.DATA_PATH = data_path
    config.SIMULATOR.SCENE = scene_path
    config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'SEMANTIC_SENSOR']
    config.freeze()
    return habitat.Env(config=config)

  def nearest_navigable_sample(self, pos):
    idx = self.idx_of_unscaled_pos[tuple(pos)]
    return self.nearest_navigable_samples[idx, :]

  def unscale_pos(self, pos):
    min_tick = np.concatenate((self.gw_ticks[0, :], np.zeros(1)))
    gw_tick_size = np.concatenate((self.gw_tick_size, np.ones(1)))
    rtn = (pos - min_tick) / gw_tick_size
    return rtn.astype(int)

  def cohere(self, pos=None):
    assert self.render_mode
    if pos is None:
      pos = self.pos
    new_pos = self.nearest_navigable_sample(pos)
    ang = int(pos[2])
    rotation = utils.quat_of_ang(ang)
    new_pos = new_pos[[0, 2, 1]]
    assert self.base_env._sim.set_agent_state(new_pos, rotation)

  def step(self, action, action_info={}):
    obs, r, done, info = super().step(action)
    if self.render_mode:
      self.cohere()
      state_colors = None
      if 'belief_logits' in action_info:
        state_colors = np.exp(action_info['belief_logits'])
        mask = state_colors > 1 / len(state_colors)
        state_colors[mask] = np.log(state_colors[mask])
        state_colors[mask] -= np.min(state_colors[mask])
        state_colors[mask] /= np.max(state_colors[mask])
        state_colors[~mask] = 0
      info['img'] = self.visualize(state_colors=state_colors)
    info['obs_str'] = self.str_of_obs[obs]
    return obs, r, done, info

  def reset(self):
    obs, info = super().reset()
    if self.render_mode:
      self.cohere()
      info['img'] = self.visualize()
    info['obs_str'] = self.str_of_obs[obs]
    return obs, info

  def get_segmentation(self, rgb_obs, cat_idxes, cat_colors):
    sem_obs = self.base_env.render(mode='semantic')
    imgs = []
    for cat_idx, cat_color in zip(cat_idxes, cat_colors):
      if cat_idx is not None:
        cat = self.str_of_obs[cat_idx]
        obj_idxes = self.obj_idxes_of_cat[cat]
        img = np.isin(sem_obs, obj_idxes).astype(int)
        img = rgb_obs * img[:, :, np.newaxis]
        img[:, :, :3] = img[:, :, :3] * np.array(utils.rgb_of_color[cat_color])[np.newaxis, np.newaxis, :]
        imgs.append(img)
    foreground = np.sum(np.array(imgs), axis=0)

    background_mask = (np.isclose(foreground, 0).all(axis=2)).astype(int)
    background_alpha = np.ones((rgb_obs.shape[0], rgb_obs.shape[1], 1)).astype(int) * 32
    background = np.concatenate((rgb_obs * background_mask[:, :, np.newaxis], background_alpha), axis=2)

    foreground_mask = 1 - background_mask[:, :, np.newaxis]
    foreground_alpha = np.ones((foreground.shape[0], foreground.shape[1], 1)).astype(int) * 255
    foreground = np.concatenate((foreground, foreground_mask * foreground_alpha), axis=2)

    img = background + foreground
    return img.astype(int)

  def visualize(self, *args, **kwargs):
    assert self.render_mode
    birdseye = super().visualize(*args, **kwargs)
    fps = self.base_env.render(mode='rgb')
    if self.prev_agent_obs is not None:
      seg_obses = [self.prev_agent_obs, self.prev_shown_obs]
      seg_colors = ['gray', 'orange']
    else:
      seg_obses = [self.prev_shown_obs]
      seg_colors = ['gray']
    aug = lambda x: np.concatenate((x, np.ones((x.shape[0], x.shape[1], 1)) * 255), axis=2).astype(int) if x.shape[2] == 3 else x
    if all(x is None for x in seg_obses):
      seg = aug(fps) * 0
    else:
      seg = self.get_segmentation(fps, seg_obses, seg_colors)
    layout_img = self.viz_cat_locs([self.str_of_obs[cat_idx] for cat_idx in seg_obses if cat_idx is not None], seg_colors)
    img = np.concatenate((aug(fps), seg, aug(layout_img), aug(birdseye)), axis=1)
    return img.astype(int)

  def render(self, *args, **kwargs):
    pass


class GuideEnv(Env):

  def __init__(
    self,
    base_env,
    user_model,
    n_obs_per_act=1
    ):

    self.name = 'guide'
    self.base_env = base_env
    self.user_model = user_model
    self.n_obs_per_act = n_obs_per_act
    self.max_ep_len = self.base_env.max_ep_len

    assert (self.max_ep_len - 1) % self.n_obs_per_act == 0

    self.curr_step = None

  def step(self, action, action_info={}):
    if action is not None:
      assert self.curr_step % self.n_obs_per_act == 0
      self.base_env.show(action, info=action_info)
      if hasattr(self.user_model, 'is_human') and self.user_model.is_human:
        self.base_env.render()
      user_action = self.user_model(action)
    else:
      assert self.curr_step % self.n_obs_per_act != 0
      user_action = self.base_env.noop_action
    prev_state = deepcopy(self.base_env.prev_state)
    user_belief = self.user_model.belief(prev_state) if prev_state is not None else np.nan
    action_info = {}
    if hasattr(self.user_model, 'belief_logits'):
      action_info['belief_logits'] = self.user_model.belief_logits
    obs, r, done, info = self.base_env.step(user_action, action_info=action_info)
    info['user_action'] = user_action
    info['user_belief_in_true_state'] = user_belief
    info['user_belief_acc'] = np.exp(user_belief)
    if hasattr(self.base_env, 'is_obs_informative'):
      info['informative_obs'] = self.base_env.is_obs_informative(action)
    self.curr_step += 1
    return obs, r, done, info

  def reset(self):
    obs, info = self.base_env.reset()
    self.user_model.reset(obs, info)
    self.curr_step = 0
    return obs, info

  def reset_init_order(self, *args, **kwargs):
    self.base_env.reset_init_order(*args, **kwargs)

  def render(self, *args, **kwargs):
    self.base_env.render(*args, **kwargs)


class CLFEnv(Env):

  def __init__(self, dataset, max_ep_len=None, render_mode=True, shuffle_obs_idxes=False):
    self.name = 'clf'
    self.dataset = dataset
    self.n_actions = self.dataset['n_classes']
    self.n_feat_dim = self.dataset['feats'].shape[1]
    self.render_mode = render_mode
    self.noop_action = 0

    if max_ep_len is None:
      raise ValueError
    self.max_ep_len = max_ep_len
    n = self.n_feat_dim // max_ep_len
    self.n_obs_dim = max_ep_len + n

    self.shuffle_obs_idxes = shuffle_obs_idxes

    self.obs_idxes = None
    self.curr_step = None
    self.viewer = None
    self.feats = None
    self.label = None
    self.prev_state = None
    self.shown_obs_idxes = None
    self.img_idx_idx = None

    self.shuffled_train_idxes = deepcopy(self.dataset['train_idxes'])
    np.random.shuffle(self.shuffled_train_idxes)
    self.shuffled_val_idxes = deepcopy(self.dataset['val_idxes'])
    np.random.shuffle(self.shuffled_val_idxes)
    self.set_val_mode(False)
    self.reset_init_order()

  def set_val_mode(self, val_mode):
    self.img_idxes = self.shuffled_val_idxes if val_mode else self.shuffled_train_idxes

  def _state(self):
    return self.label

  def feat_idxes_of_obs_idxes(self, obs_idxes):
    n = self.n_feat_dim // self.max_ep_len
    feat_idxes = []
    for obs_idx in sorted(obs_idxes):
      feat_idxes.extend(list(range(obs_idx*n, (obs_idx+1)*n)))
    return feat_idxes

  def format_obs(self, obs_idx, feats=None):
    if obs_idx is None:
      return None
    if feats is None:
      feats = self.feats
    ohe = utils.onehot_encode(obs_idx, self.max_ep_len)
    feat_idxes = self.feat_idxes_of_obs_idxes([obs_idx])
    feats_subset = feats[feat_idxes]
    return np.concatenate((ohe, feats_subset))

  def _obs_idx(self):
    if self.curr_step < len(self.obs_idxes):
      return self.obs_idxes[self.curr_step]
    elif self.curr_step == len(self.obs_idxes):
      return None
    else:
      raise ValueError

  def _obs(self):
    obs_idx = self._obs_idx()
    return self.format_obs(obs_idx)

  def reward(self, prev_state, action, state):
    return 1 if action == self.label else 0

  def oracle_policy(self, obs):
    return self.label

  def step(self, action, action_info={}):
    self.curr_step += 1
    succ = (action == self.label)
    oot = self.curr_step >= self.max_ep_len

    obs = self._obs()
    state = self._state()
    r = self.reward(self.prev_state, action, state)
    done = oot
    obs_idx = self._obs_idx()
    info = {'state': state, 'obs_idx': obs_idx}
    if self.render_mode:
      info['img'] = self.visualize()[:, :, 0]
    info['succ'] = succ
    self.prev_state = state
    if self.curr_step == 1:
      info['goal'] = self.feats

    return obs, r, done, info

  def reset_init_order(self):
    self.img_idx_idx = 0

  def reset(self):
    img_idx = self.img_idxes[self.img_idx_idx]
    self.img_idx_idx = (self.img_idx_idx + 1) % len(self.img_idxes)
    self.feats = self.dataset['feats'][img_idx, :]
    self.label = self.dataset['labels'][img_idx]

    self.obs_idxes = list(range(self.max_ep_len))
    if self.shuffle_obs_idxes:
      np.random.shuffle(self.obs_idxes)

    self.shown_obs_idxes = []
    self.curr_step = 0
    obs = self._obs()
    self.prev_state = self._state()
    obs_idx = self._obs_idx()
    info = {
      'state': self.prev_state,
      'obs_idx': obs_idx
    }
    if self.render_mode:
      info['img'] = self.visualize()[:, :, 0]
    return obs, info

  def show(self, obs, info={}):
    self.shown_obs_idxes.append(info['action_idx'])

  def mask_img(self, shown_obs_idxes, feats):
    mask = np.zeros(feats.size)
    vis_feat_idxes = self.feat_idxes_of_obs_idxes(shown_obs_idxes)
    mask[vis_feat_idxes] = 1.
    last_vis_feat_idxes = vis_feat_idxes
    feats = deepcopy(feats)
    feats[last_vis_feat_idxes] = np.maximum(0.25, feats[last_vis_feat_idxes])
    vis_feats = mask * feats
    shape = self.dataset['img_shape']
    shape = np.concatenate((shape, [1]))
    return vis_feats.reshape(shape)

  def visualize(self):
    assert self.render_mode
    return self.mask_img(self.shown_obs_idxes, self.feats)


class CarEnv(Env):

  def __init__(
    self,
    encoder_model,
    dynamics_model,
    delay=0,
    render_mode=True
    ):

    self.base_env = gym.make('CarRacing-v0')
    self.n_act_dim = 3  # steer, gas, brake
    self.max_ep_len = 1000
    self.name = 'carracing'
    self.n_z_dim = 32
    self.rnn_size = 256
    self.n_obs_dim = self.n_z_dim + 2 * self.rnn_size
    self.delay = delay
    self.orig_delay = delay
    self.encoder_model = encoder_model
    self.dynamics_model = dynamics_model
    self.render_mode = render_mode

    self.noop_action = np.zeros(self.n_act_dim)

    self.prev_zch = None
    self.zch_queue = None
    self.obs_queue = None
    self.viewer = None
    self.prev_state = None
    self.img = None
    self.curr_step = None
    self.win_activated = False

    filename = os.path.join(utils.wm_dir, 'log', 'carracing.cma.16.64.best.json')
    self.expert_model = carracing_model.make_model()
    self.expert_model.load_model(filename)

  def set_practice_mode(self, mode):
    if mode:
      self.delay = 0
    else:
      self.delay = self.orig_delay

  def _encode_obs(self, obs):
    obs = utils.crop_car_frame(obs)
    z = self.encoder_model.encode_frame(obs)
    return z

  def reset(self):
    obs = self.base_env.reset()
    z = self._encode_obs(obs)
    c = np.zeros(self.rnn_size)
    h = np.zeros(self.rnn_size)
    zch = self._merge_zch(z, c, h)

    obs = utils.crop_car_frame(obs)

    self.last_zch = zch
    self.last_obs = obs
    self.actions = []

    self.curr_step = 0

    self.prev_zch = zch
    self.prev_state = self.extract_state(zch)
    info = {'state': self.prev_state, 'init_state': self.prev_state, 'true_future_obs': zch}
    info['pred_img'] = self.viz_zch(zch)
    info['delayed_img'] = obs
    info['img'] = obs
    self.img = obs

    self.win_activated = False
    return zch, info

  def _update_zch(self, prev_zch, z, action):
    _, prev_c, prev_h = self._split_zch(prev_zch)
    data = self.dynamics_model.compute_next_obs(
      z[np.newaxis, :],
      action[np.newaxis, :],
      init_state=(prev_c[np.newaxis, :], prev_h[np.newaxis, :]))
    z = data['next_obs'][0]
    c, h = data['next_state']
    c = c[0]
    h = h[0]
    zch = self._merge_zch(z, c, h)
    return zch

  def forward_sim(self, zch, actions):
    z, c, h = self._split_zch(zch)
    for action in actions:
      zch = self._update_zch(zch, z, action)
      z, c, h = self._split_zch(zch)
    return zch

  def step(self, action, action_info={}):
    obs, r, done, info = self.base_env.step(action)
    z = self._encode_obs(obs)
    zch = self._update_zch(self.prev_zch, z, action)
    _, c, h = self._split_zch(zch)
    zch = self._merge_zch(z, c, h)

    self.prev_zch = zch
    self.prev_state = self.extract_state(zch)
    info['state'] = self.prev_state
    obs = utils.crop_car_frame(obs)
    if self.delay == 0 or (self.curr_step // self.delay) % 2 == 0:
      forward_sim_obs = zch
      self.last_zch = zch
      self.last_obs = obs
      self.actions = []
      info['delay'] = False
    else:
      self.actions.append(action)
      forward_sim_obs = self.forward_sim(self.last_zch, self.actions)
      info['delay'] = True
    pred_img = self.viz_zch(forward_sim_obs)
    info['forward_sim_obs'] = forward_sim_obs

    info['pred_img'] = pred_img
    info['delayed_img'] = self.viz_zch(self.last_zch)
    info['img'] = self.viz_zch(zch)
    info['true_future_obs'] = zch
    self.curr_step += 1
    return self.last_zch, r, done, info

  def viz_zch(self, zch):
    z, c, h = self._split_zch(zch)
    img = self.encoder_model.decode_latent(z)
    return img

  def show(self, action, info={}):
    self.img = info['img']

  def render(self, *args, **kwargs):
    super().render(*args, **kwargs)
    # useful for auto-focusing game window during user study,
    # but extremely annoying during simulation experiments
    #if not self.win_activated and self.curr_step == 1:
    #  self.win_activated = True
    #  self.viewer.window.activate()

  def visualize(self, *args, **kwargs):
    return self.img

  def _split_zch(self, zch):
    c = zch[self.n_z_dim:self.n_z_dim + self.rnn_size]
    h = zch[-self.rnn_size:]
    z = zch[:self.n_z_dim]
    return z, c, h

  def _merge_zch(self, z, c, h):
    zch = np.concatenate((z, c, h))
    return zch

  def oracle_policy(self, zch):
    z, c, h = self._split_zch(zch)
    self.expert_model.state = tf.nn.rnn_cell.LSTMStateTuple(c=c[np.newaxis, :], h=h[np.newaxis, :])
    return self.expert_model.get_action(z)

  def extract_state(self, zch):
    z, c, h = self._split_zch(zch)
    return h


class LanderEnv(Env):

  def __init__(self, base_env):
    self.base_env = base_env

    self.name = 'lander'
    self.n_obs_dim = 8
    self.n_actions = 4
    self.max_ep_len = 1000
    self.noop_action = 0

    self.prev_state = None
    self.curr_step = None
    self.win_activated = False

  def set_practice_mode(self, mode):
    self.base_env.unwrapped.practice = mode

  def step(self, action, action_info={}):
    obs, r, done, info = self.base_env.step(action)
    self.prev_state = obs
    self.curr_step += 1
    oot = self.curr_step >= self.max_ep_len
    done = done or oot
    info['tilt'] = abs(obs[4])
    info['nonnoop_tilt'] = abs(obs[4]) if action != self.noop_action else np.nan
    if r == 100:
      info['succ'] = True
    return obs, r, done, info

  def reset(self):
    self.curr_step = 0
    obs = self.base_env.reset()
    self.prev_state = obs
    info = {}
    self.win_activated = False
    return obs, info

  def show(self, action, info={}):
    ang = action[4]
    self.base_env.unwrapped.render_ang = ang

  def oracle_policy(self, obs, info={}, mult=10.):
    ang = obs[4]
    p = 1 / (1 + np.exp(-mult * ang))
    if p <= 1/3: # tilted left
      return 1 # fire left thruster
    elif 1/3 < p and p <= 2/3: # not tilted
      return 0 # noop
    else: # tilted right
      return 3 # fire right thruster

  def tf_policy(self, obs, mult=50., hard_thresh_mult=10):
    ang = obs[:, 4:5]
    p = tf.sigmoid(mult * ang)
    p_right = tf.sigmoid(hard_thresh_mult * (p - 2/3))
    p_left = tf.sigmoid(-hard_thresh_mult * (p - 1/3))
    p_noop = 1 - (p_right + p_left)
    p_main = tf.zeros((tf.shape(obs)[0], 1))
    probs = tf.concat([p_noop, p_left, p_main, p_right], axis=1)
    probs /= tf.reduce_sum(probs, axis=1, keepdims=True)
    return probs

  def render(self, *args, **kwargs):
    self.base_env.render(*args, **kwargs)
    if not self.win_activated and self.curr_step == 1:
      self.win_activated = True
      self.base_env.unwrapped.viewer.window.activate()

