# Adapted from https://github.com/rddy/ReQueST/blob/master/rqst/utils.py

from __future__ import division

from copy import deepcopy
import os
import random
import functools
import collections
import json
import time
from os.path import expanduser

import gym
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import scipy
import tensorflow as tf
from IPython.core.display import display
from IPython.core.display import HTML
from matplotlib import animation
import sklearn.neighbors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

home_dir = expanduser('~')

sensei_dir = os.path.join(home_dir, 'sensei')
deps_dir = os.path.join(sensei_dir, 'deps')
data_dir = os.path.join(sensei_dir, 'data')
gw_data_dir = os.path.join(data_dir, 'gridworld')
hab_data_dir = os.path.join(data_dir, 'habitat')
gw_human_data_dir = os.path.join(data_dir, 'gridworld-human')
clf_data_dir = os.path.join(data_dir, 'clf')
car_data_dir = os.path.join(data_dir, 'carracing')
lander_data_dir = os.path.join(data_dir, 'lander')
car_human_data_dir = os.path.join(data_dir, 'carracing-human')

# https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz
mnist_dir = os.path.join(deps_dir, 'mnist')

# https://github.com/hardmaru/WorldModelsExperiments/tree/master/carracing
wm_dir = os.path.join(deps_dir, 'WorldModelsExperiments', 'carracing')

hab_layout_path = os.path.join(hab_data_dir, 'layout.pkl')

for path in [car_human_data_dir, gw_data_dir, hab_data_dir, gw_human_data_dir, clf_data_dir, car_data_dir, lander_data_dir]:
  if not os.path.exists(path):
    os.makedirs(path)

def make_tf_session(gpu_mode=False):
  if not gpu_mode:
    kwargs = {'config': tf.ConfigProto(device_count={'GPU': 0})}
  else:
    kwargs = {}
  sess = tf.InteractiveSession(**kwargs)
  return sess


def get_tf_vars_in_scope(scope):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


tf_init_vars_cache = {}


def init_tf_vars(sess, scopes=None, use_cache=False):
  """Initialize TF variables"""
  if scopes is None:
    sess.run(tf.global_variables_initializer())
  else:
    global tf_init_vars_cache
    init_ops = []
    for scope in scopes:
      if not use_cache or scope not in tf_init_vars_cache:
        tf_init_vars_cache[scope] = tf.variables_initializer(
            get_tf_vars_in_scope(scope))
      init_ops.append(tf_init_vars_cache[scope])
    sess.run(init_ops)


def save_tf_vars(sess, scope, save_path):
  """Save TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=save_path)


def load_tf_vars(sess, scope, load_path):
  """Load TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.restore(sess, load_path)


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=1,
              size=256,
              activation=tf.nn.relu,
              output_activation=tf.nn.softmax):
  """Build MLP model"""
  out = input_placeholder
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for _ in range(n_layers):
      out = tf.layers.dense(out, size, activation=activation)
    out = tf.layers.dense(out, output_size, activation=output_activation)
  return out


def onehot_encode(i, n):
  x = np.zeros(n)
  x[i] = 1
  return x


def onehot_decode(x):
  return np.argmax(x)


def run_ep(policy,
           env,
           render=False,
           max_ep_len=None):
  """Run episode"""

  if env.name == 'carracing' or (env.name == 'guide' and env.base_env.name == 'carracing'):
    render = True

  if max_ep_len is None:
    max_ep_len = env.max_ep_len

  obs, info = env.reset()
  state = info.get('state', None)

  if hasattr(policy, 'reset'):
    policy.reset(obs, info)

  if render:
    env.render()

  done = False
  prev_state = deepcopy(state)
  prev_obs = deepcopy(obs)
  rollout = []

  countdown_envs = ['lander','carracing']
  guided_human = env.name == 'guide' and hasattr(env.user_model, 'is_human') and env.user_model.is_human
  solo_human = hasattr(policy, 'is_human') and policy.is_human
  if env.name == 'guide':
    env_name = env.base_env.name
  else:
    env_name = env.name
  is_countdown_env = env_name in countdown_envs
  if is_countdown_env and (guided_human or solo_human):
    input('-----\nPress enter to start playing: ')
    T = 5
    for t in range(T):
      print('Episode starting in %d seconds' % (T-t))
      time.sleep(1)
    print('-----\n')

  start_time = time.time()
  succ = False
  for t in range(max_ep_len):
    if done:
      break

    try:
      action = policy(obs, info)
    except TypeError:
      action = policy(obs)

    try:
      action_info = policy.get_action_info()
    except AttributeError:
      action_info = {}#'agent_obs': obs}

    policy_start_time = time.time()

    obs, r, done, info = env.step(action, action_info=action_info)
    state = info.get('state', None)

    policy_eval_time = time.time() - policy_start_time

    if env.name == 'guide':
      stored_obs = deepcopy(action)
      action = info['user_action']
      info['agent_obs'] = prev_obs
    else:
      stored_obs = prev_obs

    info['policy_eval_time'] = policy_eval_time

    rollout.append(deepcopy((prev_state, stored_obs, action, r, state, info)))
    prev_state = deepcopy(state)
    prev_obs = deepcopy(obs)
    if render:
      env.render()

    succ = info.get('succ', False)

  if env_name == 'gridworldnav' and (guided_human or solo_human):
    print('-----\nThe episode is over. %s' % ('You reached the goal :)' if succ else 'You did not reach the goal :('))
    print('-----\n')

  return rollout


def vectorize_rollouts(rollouts, max_ep_len, preserve_trajs=False):
  """Unzips rollouts into separate arrays for obses, actions, etc."""
  # preserve_trajs = False -> flatten rollouts, lose episode structure
  assert all(len(rollout) <= max_ep_len for rollout in rollouts)
  data = {'states': [], 'obses': [], 'actions': []}
  for rollout in rollouts:
    more_states, more_obses, more_actions = [
        list(x) for x in zip(*rollout)
    ][:3]
    if preserve_trajs:
      more_states = pad(np.array(more_states), max_ep_len)
      more_obses = pad(np.array(more_obses), max_ep_len)
      more_actions = pad(np.array(more_actions), max_ep_len)
    data['states'].append(more_states)
    data['obses'].append(more_obses)
    data['actions'].append(more_actions)

  if not preserve_trajs:
    data = {k: sum(v, []) for k, v in data.items()}

  data = {k: np.array(v) for k, v in data.items()}

  if preserve_trajs:
    data['traj_lens'] = np.array([
        len(rollout) for rollout in rollouts
    ])  # remember where padding begins
    data['masks'] = build_mask(data['traj_lens'], max_ep_len)

    goals = [rollout[0][-1].get('goal', None) for rollout in rollouts]
    assert all(goal is not None for goal in goals)
    data['goals'] = np.array(goals)

  idxes = list(range(len(data['obses'])))
  random.shuffle(idxes)
  data = {k: v[idxes] for k, v in data.items()}

  return data


def split_rollouts(rollouts, train_frac=0.9):
  """Train-test split
  Useful for sample_batch
  """
  idxes = list(range(rollouts['obses'].shape[0]))
  random.shuffle(idxes)
  if train_frac is not None:
    n_train_examples = int(train_frac * len(idxes))
    train_idxes = idxes[:n_train_examples]
    val_idxes = idxes[n_train_examples:]
  else:
    train_idxes = idxes
    val_idxes = idxes

  rollouts.update({
      'train_idxes': train_idxes,
      'val_idxes': val_idxes
  })
  return rollouts


def pad(arr, max_len):
  """Zero-padding
  Useful for vectorize_rollouts(, preserve_trajs=True) and split_prefs"""
  n = arr.shape[0]
  if n > max_len:
    raise ValueError
  elif n == max_len:
    return arr
  else:
    shape = [max_len - n]
    shape.extend(arr.shape[1:])
    padding = np.zeros(shape)
    return np.concatenate((arr, padding), axis=0)


def build_mask(traj_lens, max_len):
  """Mask for traj data with variable-length trajs
  """
  return np.array([
      (np.arange(0, max_len, 1) < traj_len).astype(np.float) for traj_len in traj_lens
  ])


def elts_at_idxes(x, idxes):
  if type(x) == list:
    return [x[i] for i in idxes]
  else:
    return x[idxes]


def sample_batch(size, data, data_keys, idxes_key, class_idxes_key=None):
  if size < len(data[idxes_key]):
    if class_idxes_key is None:
      idxes = random.sample(data[idxes_key], size)
    else:
      # sample class-balanced batch
      idxes = []
      idxes_of_class = data[class_idxes_key]
      n_classes = len(idxes_of_class)
      for c, idxes_of_c in idxes_of_class.items():
        k = int(np.ceil(size / n_classes))
        if k > len(idxes_of_c):
          idxes_of_c_samp = idxes_of_c
        else:
          idxes_of_c_samp = random.sample(idxes_of_c, k)
        idxes.extend(idxes_of_c_samp)
      if len(idxes) > size:
        random.shuffle(idxes)
        idxes = idxes[:size]
  else:
    idxes = data[idxes_key]
  batch = {k: elts_at_idxes(data[k], idxes) for k in data_keys}
  return batch


col_means = lambda x: np.nanmean(x, axis=0)
col_stderrs = lambda x: np.nanstd(
    x, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(x), axis=0))
err_bar_mins = lambda x: col_means(x) - col_stderrs(x)
err_bar_maxs = lambda x: col_means(x) + col_stderrs(x)


def make_perf_mat(perf_evals, y_key):
  n = len(perf_evals[0][y_key])
  max_len = max(len(perf_eval[y_key]) for perf_eval in perf_evals)

  def pad(lst, n):
    if len(lst) < n:
      lst += [np.nan] * (n - len(lst))
    return lst

  return np.array([pad(perf_eval[y_key], max_len) for perf_eval in perf_evals])


def smooth(xs, win=10):
  psums = np.concatenate((np.zeros(1), np.cumsum(xs)))
  rtn = (psums[win:] - psums[:-win]) / win
  rtn[0] = xs[0]
  return rtn


def plot_perf_evals(perf_evals,
                    x_key,
                    y_key,
                    label='',
                    smooth_win=None,
                    color=None):
  y_mat = make_perf_mat(perf_evals, y_key)
  y_mins = err_bar_mins(y_mat)
  y_maxs = err_bar_maxs(y_mat)
  y_means = col_means(y_mat)

  if smooth_win is not None:
    y_mins = smooth(y_mins, win=smooth_win)
    y_maxs = smooth(y_maxs, win=smooth_win)
    y_means = smooth(y_means, win=smooth_win)

  xs = max([perf_eval[x_key] for perf_eval in perf_evals], key=lambda x: len(x))
  xs = xs[:len(y_means)]

  kwargs = {}
  if color is not None:
    kwargs['color'] = color

  plt.fill_between(
      xs,
      y_mins,
      y_maxs,
      where=y_maxs >= y_mins,
      interpolate=True,
      label=label,
      alpha=0.5,
      **kwargs)
  plt.plot(xs, y_means, **kwargs)


def strip_guide_noops(rollout):
  assert rollout[-1][1] is not None
  return [x for x in rollout if x[1] is not None]


def stderr(xs):
  n = (~np.isnan(xs)).sum()
  return np.nanstd(xs) / np.sqrt(n)


label_of_perf_met = {
  'user_belief_in_true_state': 'Log-Likelihood of True State',
  'user_belief_acc': 'Likelihood of True State',
  'succ': 'Success Rate',
  'rollout_len': 'Time to Completion',
  'rew': 'Reward',
  'dist_to_goal': 'Distance to Goal',
  'return': 'Return',
  'policy_eval_time': 'Decision-Making Time',
  'tilt': 'Tilt',
  'nonnoop_tilt': 'Tilt During User Intervention',
  'crash': 'Off-Road Rate',
  'min_dist': 'Distance from Road',
  'informative_obs': 'Informative Observation Rate',
  'time_to_label': 'Recognition Time'
}

color_of_guide = {
  'iden': 'gray',
  'unif': 'red',
  'oracle': 'green',
  'naive': 'teal',
  'learned': 'orange',
  'prac': 'blue'
}

label_of_guide = {
  'iden': 'Unassisted (Baseline)',
  'unif': 'Random (Baseline)',
  'oracle': 'Oracle (Baseline)',
  'naive': 'Naive ASE (Baseline)',
  'learned': 'ASE (Our Method)',
  'prac': 'Practice'
}


def compute_perf_metrics(rollouts, env, max_ep_len=None):
  if max_ep_len is None:
    max_ep_len = env.max_ep_len

  rollouts = [strip_guide_noops(rollout) for rollout in rollouts]

  metrics = {}
  non_rew_metrics = ['time_to_label', 'succ', 'user_belief_in_true_state', 'user_belief_acc', 'dist_to_goal', 'tilt', 'policy_eval_time', 'nonnoop_tilt', 'informative_obs', 'crash', 'min_dist']
  for key in non_rew_metrics:
    vals = [x[-1][key] for rollout in rollouts for x in rollout if key in x[-1]]
    metrics[key] = np.nanmean(vals)
    metrics['%s_stderr' % key] = stderr(vals)

  rews = [x[3] for rollout in rollouts for x in rollout]
  metrics['rew'] = np.nanmean(rews)
  metrics['rew_stderr'] = stderr(rews)

  returns = [sum(x[3] for x in rollout) for rollout in rollouts]
  metrics['return'] = np.nanmean(returns)
  metrics['return_stderr'] = stderr(returns)

  rollout_lens = [len(rollout) for rollout in rollouts]
  metrics['rollout_len'] = np.nanmean(rollout_lens)
  metrics['rollout_len_stderr'] = stderr(rollout_lens)

  for key in (non_rew_metrics + ['rew']):
    arr = [[] for _ in range(max_ep_len)]
    for rollout in rollouts:
      for i, x in enumerate(rollout):
        if key == 'rew':
          arr[i].append(x[3])
        elif key in x[-1]:
          arr[i].append(x[-1][key])
    metrics['%s_t' % key] = [np.nanmean(x) for x in arr]
    metrics['%s_stderr_t' % key] = [stderr(x) for x in arr]

  return metrics


def crop_car_frame(obs):
  obs = obs[0:84, :, :].astype(np.float) / 255.0
  obs = scipy.misc.imresize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs


def evaluate_policy(sess,
                    env,
                    policy,
                    n_eval_rollouts=10):

  unassisted_human = env.name != 'guide' and hasattr(policy, 'is_human') and policy.is_human
  guided_human = env.name == 'guide' and hasattr(env.user_model, 'is_human') and env.user_model.is_human
  render = unassisted_human or guided_human
  eval_rollouts = [
      run_ep(policy, env, render=render) for _ in range(n_eval_rollouts)
  ]
  perf = compute_perf_metrics(eval_rollouts, env)

  return {
      'perf': perf,
      'rollouts': eval_rollouts,
  }


def converged(val_losses, ftol, min_iters=2, eps=1e-9):
  return len(val_losses) >= max(2, min_iters) and (
      val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) /
      (eps + abs(val_losses[-2])) < ftol)


def sample_from_categorical(logits):
  noise = np.random.gumbel(loc=0, scale=1, size=logits.size)
  return (logits + noise).argmax()


def update_beliefs(env, belief_logits, obs, prev_act, obs_model, dynamics_model, eps=1e-9):
  if env.name in ['gridworldnav', 'habitatnav']:
    assert len(obs_model.shape) == len(belief_logits.shape) + 1
    if len(obs_model.shape) == 2:
      shaped_obs_model = obs_model[np.newaxis, :, :]
      shaped_belief_logits = belief_logits[np.newaxis, :]
    elif len(obs_model.shape) == 3: # ensemble
      shaped_obs_model = obs_model
      shaped_belief_logits = belief_logits
    else:
      raise ValueError
    updated_belief_logits = deepcopy(shaped_obs_model[:, obs, :])
    if prev_act is not None:
      updated_belief_logits += scipy.special.logsumexp(
        dynamics_model[np.newaxis, :, :, prev_act] + shaped_belief_logits[:, np.newaxis, :], axis=2)
    updated_belief_logits -= scipy.special.logsumexp(updated_belief_logits, axis=1, keepdims=True)
    if len(obs_model.shape) == 2: # no ensemble
      updated_belief_logits = updated_belief_logits[0, :]
    return np.maximum(updated_belief_logits, -1e9)
  else:
    raise NotImplementedError


def expand_dims(tensor, axis):
  return functools.reduce(tf.expand_dims, axis, tensor)


def spike_and_slab(eps, n, i):
  return np.log(eps * onehot_encode(i, n) + (1 - eps) * unif(n))


def unif(n):
  return np.ones(n) / n


def ens_agreement(logits, eps=1e-9):
  """Numpy version of ensemble disagreement"""
  # logits[ens_mem_idx, state]
  probs = np.exp(logits)
  mean = np.mean(probs, axis=0, keepdims=True)
  kl = np.sum(probs * (logits - np.log(eps + mean)), axis=1)
  return -np.mean(kl)


def batch_onehot_encode(mat, n):
  batch_size = mat.shape[0]
  idxes = mat.ravel().astype(int)
  if len(mat.shape) > 1:
    max_ep_len = mat.shape[1]
    ohe = np.zeros((batch_size, max_ep_len, n))
    xs = np.repeat(np.arange(0, batch_size, 1), max_ep_len)
    ys = np.tile(np.arange(0, max_ep_len, 1), batch_size)
    ohe[xs, ys, idxes] = 1.
  else:
    ohe = np.zeros((batch_size, n))
    xs = np.arange(0, batch_size, 1)
    ohe[xs, idxes] = 1.
  return ohe


def enumerate_gw_poses(x_ticks, y_ticks):
  all_poses = list(zip(*[x.ravel() for x in np.meshgrid(x_ticks, y_ticks)]))
  all_poses = [[x, y, ang] for x, y in all_poses for ang in range(4)]
  return np.array(all_poses)


thetas = np.arange(0, 2, 0.5) * np.pi
axis = np.array([0., 1., 0.])
quats = [quaternion.from_rotation_vector(theta * axis) for theta in thetas]
def quat_of_ang(ang):
  return quats[ang]


def sample_gw_goals(
  navigable_poses,
  n_tasks=None,
  goals_path=None,
  verbose=True
  ):

  goals = deepcopy(navigable_poses)

  if n_tasks is not None:
    assert n_tasks <= goals.shape[0]
    idxes = np.random.sample(goals.shape[0], n_tasks)
    goals = goals[idxes, :]

  if verbose:
    plt.scatter(goals[:, 0], goals[:, 1], linewidth=0, color='orange', s=100, marker='*')
    plt.xlim([-0.5, gw_size-0.5])
    plt.ylim([-0.5, gw_size-0.5])
    plt.show()

  if goals_path is not None:
    with open(goals_path, 'wb') as f:
      pickle.dump(goals, f, pickle.HIGHEST_PROTOCOL)

  return goals


def gw_ticks(mins, maxs, gw_size):
  x_step = (maxs[0] - mins[0]) / (gw_size - 1)
  y_step = (maxs[1] - mins[1]) / (gw_size - 1)
  gw_xs = np.arange(mins[0], maxs[0] + x_step/2, x_step)
  gw_ys = np.arange(mins[1], maxs[1] + y_step/2, y_step)
  assert gw_xs.size == gw_size, gw_xs.size
  assert gw_ys.size == gw_size, gw_ys.size
  gw_ticks = np.stack([gw_xs, gw_ys], axis=1)
  gw_tick_size = np.array([x_step, y_step])
  return gw_ticks, gw_tick_size


def play_nb_vid(frames):
  fig = plt.figure(figsize=(10, 5))
  plt.axis('off')
  ims = [[plt.imshow(frame, animated=True)] for frame in frames]
  plt.close()
  anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
  display(HTML(anim.to_html5_video()))
  return anim


def make_mnist_dataset(verbose=False):
  load = lambda fname: np.load(os.path.join(mnist_dir, fname))

  def load_imgs(X):
    X = X.T
    d = int(np.sqrt(X.shape[1]))
    X = X.reshape((X.shape[0], d, d))
    return X

  load_labels = lambda X: X.T.ravel().astype(int)
  X = load('mnist.npz')
  train_imgs = load_imgs(X['train'])
  train_labels = load_labels(X['train_labels'])
  test_imgs = load_imgs(X['test'])
  test_labels = load_labels(X['test_labels'])

  imgs = np.concatenate((train_imgs, test_imgs), axis=0)
  labels = np.concatenate((train_labels, test_labels))
  n_classes = len(np.unique(labels))
  img_shape = imgs.shape[1:]
  feats = imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) / 255.
  feats = (feats > 0.5).astype(float)
  train_idxes = list(range(train_labels.size))
  val_idxes = list(range(train_labels.size, train_labels.size + test_labels.size))

  dataset = {
      'img_shape': img_shape,
      'n_classes': n_classes,
      'feats': feats,
      'labels': labels,
      'train_idxes': train_idxes,
      'val_idxes': val_idxes
  }

  return dataset


def bal_weights_of_batch(batch_elts):
  batch_size = len(batch_elts)
  weights = np.ones(batch_size)
  idxes_of_elt = collections.defaultdict(list)
  for idx, elt in enumerate(batch_elts):
    idxes_of_elt[elt].append(idx)
  for elt, idxes in idxes_of_elt.items():
    weights[idxes] = 1. / len(idxes)
  return weights


def load_wm_pretrained_model(jsonfile, scope, sess):
  with open(jsonfile, 'r') as f:
    params = json.load(f)

  t_vars = tf.trainable_variables(scope=scope)
  idx = 0
  for var in t_vars:
    pshape = tuple(var.get_shape().as_list())
    p = np.array(params[idx])
    assert pshape == p.shape, 'inconsistent shape'
    pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
    assign_op = var.assign(pl)
    sess.run(assign_op, feed_dict={pl.name: p / 10000.})
    idx += 1


def smooth_matrix(mat, n, eps=1e-9):
  return (1 - eps) * mat + eps * np.ones(mat.shape) / n


class LSTMCellWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell):

  def __init__(self, *args, **kwargs):
    super(LSTMCellWrapper, self).__init__(*args, **kwargs)
    self._inner_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(*args, **kwargs)

  @property
  def state_size(self):
    return self._inner_cell.state_size

  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def call(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state


def tf_lognormal(y, mean, logstd):
  """Log-normal distribution"""
  log_sqrt_two_pi = np.log(np.sqrt(2.0 * np.pi))
  return -0.5 * ((y - mean) / tf.exp(logstd))**2 - logstd - log_sqrt_two_pi


class StutteredPolicy(object):

  def __init__(self, policy, n_steps_per_act):
    self.policy = policy
    self.n_steps_per_act = n_steps_per_act

    self.curr_step = None

  def reset(self, obs, info={}):
    self.curr_step = 0
    try:
      self.policy.reset(obs, info=info)
    except AttributeError:
      pass

  def __call__(self, obs, info={}):
    self.curr_step += 1
    if (self.curr_step - 1) % self.n_steps_per_act != 0:
      return None
    else:
      return self.policy(obs, info=info)


class CarGuidePolicy(object):

  def __init__(self, guide_name):
    self.guide_name = guide_name

    self.action_info = None

  def reset(self, *args, **kwargs):
    self.action_info = None

  def __call__(self, obs, info={}):
    if self.guide_name == 'iden':
      img = info['delayed_img']
      shown_obs = obs
    elif self.guide_name == 'naive':
      img = info['pred_img']
      shown_obs = info.get('forward_sim_obs', obs)
    elif self.guide_name == 'oracle':
      img = info['img']
      shown_obs = info.get('true_future_obs', obs)
    else:
      raise ValueError
    self.action_info = {'img': img}
    return shown_obs

  def get_action_info(self):
    return self.action_info


class CLFIdenPolicy(object):

  def __init__(self):
    self.action_info = None

  def reset(self, *args, **kwargs):
    self.action_info = None

  def __call__(self, obs, info={}):
    action_idx = info['obs_idx']
    self.action_info = {'action_idx': action_idx}
    return obs

  def get_action_info(self):
    return self.action_info


class CLFUnifPolicy(object):

  def __init__(self, env):
    self.env = env

    self.action_idxes = None
    self.curr_step = None
    self.action_info = None

  def reset(self, obs, info={}):
    self.action_idxes = deepcopy(self.env.obs_idxes)
    random.shuffle(self.action_idxes)
    self.curr_step = 0
    self.action_info = None

  def __call__(self, obs, info={}):
    action_idx = self.action_idxes[self.curr_step]
    self.action_info = {'action_idx': action_idx}
    self.curr_step += 1
    return self.env.format_obs(action_idx)

  def get_action_info(self):
    return self.action_info


def lander_unif_guide_policy(obs, info={}):
  shown_obs = deepcopy(obs)
  shown_obs[4] = -np.pi + np.random.random() * np.pi
  return shown_obs

def distort_lander_obs(obs, distortion_factor):
  distorted_obs = deepcopy(obs)
  distorted_obs[4] /= distortion_factor
  return distorted_obs


def angular_dist(p, q):
  return np.minimum(np.abs(p-q), 2*np.pi - np.abs(p-q))


def rand_tiebreak_min(arr, key):
  min_val = None
  min_elt = None
  n_mins = 0
  for elt in arr:
    val = key(elt)
    if min_val is None or val < min_val:
      min_val = val
      min_elt = elt
      n_mins = 1
    elif val == min_val:
      n_mins += 1
      if np.random.random() < 1. / n_mins: # reservoir sampling
        min_elt = elt
  return min_elt


def discretize_p_value(p):
  if p < 0.0001:
    return '<.0001'
  elif p < 0.001:
    return '<.001'
  elif p < 0.01:
    return '<.01'
  elif p < 0.05:
    return '<.05'
  else:
    return '>.05'


rgb_of_color = {
  'orange': (1, 0.5, 0),
  'gray': (0.5, 0.5, 0.5)
}
