from __future__ import division

from copy import deepcopy
import random

import numpy as np
import tensorflow as tf
import scipy.stats

from . import utils
from .models import TFModel


class TabularObsModel(TFModel):

  def __init__(
    self,
    dynamics_model,
    prior_obs_model,
    q_func,
    *args,
    prior_coeff=0.,
    warm_start=False,
    user_init_belief_conf,
    obs_params_only=False,
    **kwargs
    ):
    super().__init__(*args, **kwargs)

    self.prior_coeff = prior_coeff
    self.warm_start = warm_start
    self.user_init_belief_conf = user_init_belief_conf
    self.obs_params_only = obs_params_only

    self.obs_logits_eval = prior_obs_model[np.newaxis, :, :]

    self.dynamics_model = tf.convert_to_tensor(dynamics_model, dtype=tf.float32)
    self.prior_obs_model = tf.convert_to_tensor(prior_obs_model, dtype=tf.float32)
    self.q_func = tf.convert_to_tensor(q_func, dtype=tf.float32)

    self.data_keys = ['obses', 'actions', 'masks', 'goals', 'init_beliefs']

    self.max_ep_len = None

  def build_graph(self):
    self.obs_ph = tf.placeholder(tf.float32, [None, self.max_ep_len, self.env.n_obses])
    self.act_ph = tf.placeholder(tf.float32, [None, self.max_ep_len, self.env.n_actions])
    self.mask_ph = tf.placeholder(tf.float32, [None, self.max_ep_len])
    self.goal_ph = tf.placeholder(tf.float32, [None, self.env.n_goals])
    self.init_belief_ph = tf.placeholder(tf.float32, [None, self.env.n_states])
    self.member_mask_ph = tf.placeholder(tf.float32, [self.n_ens_mem])

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      if self.warm_start:
        assert not self.obs_params_only
        raw_obs_logits = tf.Variable(self.prior_obs_model[np.newaxis, :, :], dtype=tf.float32)
      else:
        if not self.obs_params_only:
          raw_obs_logits = tf.Variable(tf.ones([self.n_ens_mem, self.env.n_obses, self.env.n_states], dtype=tf.float32))
        else:
          if self.env.n_obj_sets is None:
            obs_weights = tf.Variable(tf.zeros([self.n_ens_mem, self.env.n_obses, 1]), dtype=tf.float32)
            self.obs_weights = tf.sigmoid(obs_weights)
          else:
            obs_weights = tf.Variable(tf.zeros([self.n_ens_mem, 1, 1]))
            obs_weights = tf.sigmoid(obs_weights)
            obs_weights = tf.concat([obs_weights, tf.ones([self.n_ens_mem, self.env.n_obj_sets-1, 1])], axis=1)
            tiles = [tf.tile(obs_weights[:, i:i+1, :], [1, self.env.n_obses//self.env.n_obj_sets, 1]) for i in range(self.env.n_obj_sets)]
            obs_weights = tf.concat(tiles, axis=1)
            self.obs_weights = obs_weights
          raw_obs_probs = self.obs_weights * tf.exp(self.prior_obs_model[np.newaxis, :, :]) + (1 - self.obs_weights) * (np.ones((1, self.env.n_obses, self.env.n_states)) / self.env.n_obses)
          raw_obs_logits = tf.log(raw_obs_probs)
    self.obs_logits = raw_obs_logits - tf.expand_dims(tf.reduce_logsumexp(raw_obs_logits, axis=1), 1)

    def update(belief_logits, obs_t, act_t):
      # (n_ens_mem, 1, n_obses, n_states) * (1, batch_size, n_obses, 1) -> (n_ens_mem, batch_size, n_obses, n_states)
      obs_logits = tf.expand_dims(self.obs_logits, 1) * utils.expand_dims(obs_t, [0, 3])
      # (n_ens_mem, batch_size, n_obses, n_states) -> (n_ens_mem, batch_size, n_states)
      obs_logits = tf.reduce_sum(obs_logits, axis=2)
      # (1, 1, n_states, n_states, n_actions) * (1, batch_size, 1, 1, n_actions) -> (1, batch_size, n_states, n_states, n_actions)
      trans_logits = utils.expand_dims(self.dynamics_model, [0, 1]) * utils.expand_dims(act_t, [0, 2, 3])
      # (1, batch_size, n_states, n_states, n_actions) -> (1, batch_size, n_states, n_states)
      trans_logits = tf.reduce_sum(trans_logits, axis=4)
      # (1, batch_size, n_states, n_states) + (n_ens_mem, batch_size, 1, n_states) -> (n_ens_mem, batch_size, n_states, n_states)
      trans_logits += tf.expand_dims(belief_logits, 3)
      # (n_ens_mem, batch_sizes, n_states, n_states) -> (n_ens_mem, batch_size, n_states)
      trans_logits = tf.reduce_logsumexp(trans_logits, axis=3)
      new_belief_logits = obs_logits + trans_logits
      new_belief_logits -= tf.expand_dims(tf.reduce_logsumexp(new_belief_logits, axis=2), 2)
      return new_belief_logits

    # (batch_size, n_states) -> (1, batch_size, n_states)
    belief_logits = tf.expand_dims(self.init_belief_ph, 0)
    seq_belief_logits = []
    for t in range(self.max_ep_len):
      belief_logits = update(
        belief_logits,
        self.obs_ph[:, t, :],
        self.act_ph[:, t, :]
      )
      seq_belief_logits.append(belief_logits)
    # [(n_ens_mem, batch_size, n_states)] -> (n_ens_mem, batch_size, max_ep_len, n_states)
    seq_belief_logits = tf.stack(seq_belief_logits, axis=2)

    # (1, 1, 1, n_states, n_actions, n_goals) * (1, batch_size, 1, 1, 1, n_goals) -> (1, batch_size, 1, n_states, n_actions, n_goals)
    q_vals = utils.expand_dims(self.q_func, [0, 1, 2]) * utils.expand_dims(self.goal_ph, [0, 2, 3, 4])
    # (1, batch_size, 1, n_states, n_actions, n_goals) -> (1, batch_size, 1, n_states, n_actions)
    q_vals = tf.reduce_sum(q_vals, axis=5)
    # (1, batch_size, 1, n_states, n_actions) * (1, batch_size, max_ep_len, 1, n_actions) -> (1, batch_size, max_ep_len, n_states, n_actions)
    q_vals *= utils.expand_dims(self.act_ph, [0, 3])
    # (1, batch_size, max_ep_len, n_states, n_actions) -> (1, batch_size, max_ep_len, n_states)
    q_vals = tf.reduce_sum(q_vals, axis=4)

    act_logits = seq_belief_logits + q_vals
    # (n_ens_mem, batch_size, max_ep_len, n_states) -> (n_ens_mem, batch_size, max_ep_len)
    act_logits = tf.reduce_logsumexp(act_logits, axis=3)
    # (n_ens_mem, batch_size, max_ep_len) -> (n_ens_mem)
    likelihood_loss = -tf.reduce_sum(act_logits * self.mask_ph, axis=[1, 2]) / tf.reduce_sum(self.mask_ph)
    # (n_ens_mem) -> (1)
    self.likelihood_loss = tf.reduce_sum(likelihood_loss * self.member_mask_ph)

    prior_obs_probs = tf.exp(self.prior_obs_model)
    # (1, n_obses, n_states) * (n_ens_mem, n_obses, n_states) -> (n_ens_mem, n_obses, n_states)
    log_loss = -tf.expand_dims(prior_obs_probs, 0) * self.obs_logits
    # (n_ens_mem, n_obses, n_states) -> (n_ens_mem, n_states)
    log_loss = tf.reduce_sum(log_loss, axis=1)
    # (n_ens_mem, n_states) -> (n_ens_mem)
    prior_loss = tf.reduce_mean(log_loss, axis=1)
    # (n_ens_mem) -> (1)
    self.prior_loss = tf.reduce_sum(prior_loss * self.member_mask_ph)

    self.loss = self.prior_coeff * self.prior_loss + self.likelihood_loss

  def format_batch(self, batch, member_mask):
    feed_dict = {
      self.obs_ph: batch['obses'],
      self.act_ph: batch['actions'],
      self.mask_ph: batch['masks'],
      self.goal_ph: batch['goals'],
      self.init_belief_ph: batch['init_beliefs'],
      self.member_mask_ph: member_mask
    }
    return feed_dict

  def format_train_data(self, data):
    fmted_data = deepcopy(data)

    max_ep_len = data['obses'].shape[1]
    if self.max_ep_len is None or self.max_ep_len < max_ep_len:
      self.max_ep_len = max_ep_len
      self.build_graph()

    fmted_data['obses'] = utils.batch_onehot_encode(data['obses'], self.env.n_obses)
    fmted_data['actions'] = utils.batch_onehot_encode(data['actions'], self.env.n_actions)
    fmted_data['goals'] = utils.batch_onehot_encode(data['goals'], self.env.n_goals)

    init_states = utils.batch_onehot_encode(data['states'][:, 0], self.env.n_states)
    unif = (1 - self.user_init_belief_conf) * utils.unif(self.env.n_states)
    fmted_data['init_beliefs'] = self.user_init_belief_conf * init_states + unif
    return fmted_data

  def train(self, data, *args, **kwargs):
    fmted_data = self.format_train_data(data)
    super().train(fmted_data, *args, **kwargs)
    self.obs_logits_eval = self.sess.run(self.obs_logits)

  def load(self, *args, **kwargs):
    super().load(*args, **kwargs)
    self.obs_logits_eval = self.sess.run(self.obs_logits)


class EncoderObsModel(TFModel):

  def __init__(
    self,
    *args,
    n_hidden=32,
    use_recon=True,
    use_latent_beliefs=True,
    **kwargs
    ):

    super().__init__(*args, **kwargs)

    self.n_hidden = n_hidden
    self.use_recon = use_recon
    self.use_latent_beliefs = use_latent_beliefs

    self.data_keys = ['obses', 'actions', 'goals', 'traj_lens']

    self.obs_ph = tf.placeholder(tf.float32, [None, self.env.max_ep_len, self.env.n_obs_dim])
    self.act_ph = tf.placeholder(tf.float32, [None, self.env.max_ep_len, self.env.n_actions])
    self.target_ph = tf.placeholder(tf.float32, [None, self.env.n_feat_dim])
    self.traj_len_ph = tf.placeholder(tf.int32, [None])
    self.member_mask_ph = tf.placeholder(tf.float32, [self.n_ens_mem])
    self.init_hidden_c_ph = tf.placeholder(tf.float32, [self.n_ens_mem, None, self.n_hidden])
    self.init_hidden_h_ph = tf.placeholder(tf.float32, [self.n_ens_mem, None, self.n_hidden])
    self.bal_weights_ph = tf.placeholder(tf.float32, [None, self.env.n_feat_dim])

    ens_mem_recons = []
    self.next_hidden = []
    init_hidden_cs = tf.unstack(self.init_hidden_c_ph, axis=0)
    init_hidden_hs = tf.unstack(self.init_hidden_h_ph, axis=0)
    self.outputs = []
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
      for ens_mem_idx, init_hidden_c, init_hidden_h in zip(range(self.n_ens_mem), init_hidden_cs, init_hidden_hs):
        ens_mem_scope = str(ens_mem_idx)
        ens_mem_rnn_scope = ens_mem_scope+'-rnn'
        init_hidden = tf.nn.rnn_cell.LSTMStateTuple(init_hidden_c, init_hidden_h)
        outputs, next_hidden = tf.nn.dynamic_rnn(
          lstm_cell,
          self.obs_ph,
          initial_state=init_hidden,
          sequence_length=self.traj_len_ph,
          scope=ens_mem_rnn_scope,
          dtype=tf.float32
        )
        self.outputs.append(outputs)
        self.next_hidden.append(next_hidden)
        if self.use_recon:
          recons = []
          for hidden in tf.unstack(outputs, axis=1):
            recon = self.build_decoder(hidden, ens_mem_scope)
            recons.append(recon)
          recons = tf.stack(recons, axis=1)
          ens_mem_recons.append(recons)
    self.outputs = tf.stack(self.outputs, axis=0)

    if self.use_recon:
      ens_mem_recons = tf.stack(ens_mem_recons, axis=0)
      recons = utils.expand_dims(self.member_mask_ph, [1, 2, 3]) * ens_mem_recons
      recons = tf.reduce_sum(recons, axis=0)
      recons = tf.sigmoid(recons)
      last_recons = recons[:, -1, :]
      log_losses = (last_recons - self.target_ph)**2
      recon_loss = tf.reduce_sum(log_losses * self.bal_weights_ph) / tf.reduce_sum(self.bal_weights_ph)
      self.loss = recon_loss

      self.build_decoded_state()

    self.hidden_c = None
    self.hidden_h = None
    self.belief_state = None

  def build_decoded_state(self):
    self.state_ph = tf.placeholder(tf.float32, [None, self.n_ens_mem, self.n_hidden])
    decoded_state_of_ens_mem = []
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for ens_mem_idx in range(self.n_ens_mem):
        ens_mem_scope = str(ens_mem_idx)
        decoded_state = self.build_decoder(self.state_ph[:, ens_mem_idx, :], ens_mem_scope)
        decoded_state_of_ens_mem.append(decoded_state)
    self.decoded_state = tf.stack(decoded_state_of_ens_mem, axis=1)

  def build_decoder(self, hidden, scope):
    recon = utils.build_mlp(
      hidden,
      self.env.n_feat_dim,
      scope,
      n_layers=3,
      size=self.n_hidden,
      activation=tf.nn.relu,
      output_activation=None
    )
    return recon

  def format_batch(self, batch, member_mask):
    init_hidden = np.zeros((self.n_ens_mem, batch['obses'].shape[0], self.n_hidden))
    bal_weights = utils.bal_weights_of_batch(batch['goals'].ravel()).reshape(batch['goals'].shape)
    feed_dict = {
      self.obs_ph: batch['obses'],
      self.act_ph: batch['actions'],
      self.target_ph: batch['goals'],
      self.traj_len_ph: batch['traj_lens'],
      self.init_hidden_c_ph: init_hidden,
      self.init_hidden_h_ph: init_hidden,
      self.member_mask_ph: member_mask,
      self.bal_weights_ph: bal_weights
    }
    return feed_dict

  def format_train_data(self, data):
    fmted_data = deepcopy(data)
    fmted_data['actions'] = utils.batch_onehot_encode(data['actions'], self.env.n_actions)
    return fmted_data

  def train(self, data, *args, **kwargs):
    fmted_data = self.format_train_data(data)
    super().train(fmted_data, *args, **kwargs)

  def get_state(self):
    return self.belief_state

  def reset(self, obs, info={}):
    self.hidden_c = np.zeros((self.n_ens_mem, 1, self.n_hidden))
    self.hidden_h = np.zeros((self.n_ens_mem, 1, self.n_hidden))
    self.update(obs, info=info)

  def make_feed_dict(self, obs, hidden_c, hidden_h):
    padded_obs = utils.pad(obs[np.newaxis, :], self.env.max_ep_len)
    feed_dict = {
      self.obs_ph: padded_obs[np.newaxis, :, :],
      self.traj_len_ph: [1],
      self.init_hidden_c_ph: hidden_c,
      self.init_hidden_h_ph: hidden_h
    }
    return feed_dict

  def belief_state_from_hidden(self, next_hidden):
    return np.array([hidden.h[0, :] for hidden in next_hidden])

  def update_hidden_state(self, next_hidden):
    self.hidden_c = np.array([hidden.c for hidden in next_hidden])
    self.hidden_h = np.array([hidden.h for hidden in next_hidden])
    self.belief_state = self.belief_state_from_hidden(next_hidden)

  def update(self, obs, info={}, true_state=None):
    feed_dict = self.make_feed_dict(obs, self.hidden_c, self.hidden_h)
    next_hidden = self.sess.run(self.next_hidden, feed_dict=feed_dict)
    if true_state is not None: # hypothetical update
      belief_state = self.belief_state_from_hidden(next_hidden)
      belief_in_true_state = self.belief(true_state, belief_state=belief_state)
      return belief_in_true_state
    else:
      self.update_hidden_state(next_hidden)

  def decode_states(self, states):
    feed_dict = {
      self.state_ph: states
    }
    decoded_states = self.sess.run(self.decoded_state, feed_dict=feed_dict)
    return decoded_states

  def decoded_dist(self, state, belief_state):
    states = np.array([state, belief_state])
    decoded_states = self.decode_states(states)
    return -np.mean(np.sum((decoded_states[0] - decoded_states[1])**2, axis=1))

  def belief(self, state, belief_state=None):
    if belief_state is None:
      belief_state = self.belief_state
    if self.use_latent_beliefs:
      return -np.mean(np.sum((state - belief_state)**2, axis=1))
    else:
      return self.decoded_dist(state, belief_state)


class EncoderUserObsModel(EncoderObsModel):

  def __init__(
    self,
    *args,
    n_classes=10,
    n_layers=1,
    layer_size=32,
    final_action_only=True,
    **kwargs
    ):

    super().__init__(*args, use_recon=False, **kwargs)

    self.n_classes = n_classes
    self.n_layers = n_layers
    self.layer_size = layer_size
    self.final_action_only = final_action_only

    all_logits = []
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      for ens_mem_idx, hiddens in enumerate(tf.unstack(self.outputs, axis=0)):
        logits_for_ens_mem = []
        ens_mem_scope = str(ens_mem_idx)
        for hidden in tf.unstack(hiddens, axis=1):
          logits = self.build_decoder(hidden, ens_mem_scope)
          logits_for_ens_mem.append(logits)
        logits_for_ens_mem = tf.stack(logits_for_ens_mem, axis=1)
        all_logits.append(logits_for_ens_mem)
    all_logits = tf.stack(all_logits, axis=0)

    self.logits = all_logits

    logits = utils.expand_dims(self.member_mask_ph, [1, 2, 3]) * all_logits
    logits = tf.reduce_sum(logits, axis=0)
    if self.final_action_only:
      logits = logits[:, -1:, :]
      actions = self.act_ph[:, -1:, :]
      axis = 1
    else:
      actions = self.act_ph
      axis = 2
    losses = -tf.reduce_sum(logits * actions, axis=axis)
    likelihood_loss = tf.reduce_mean(losses)

    self.loss = likelihood_loss

    self.build_decoded_state()

  def build_decoder(self, hidden, scope):
    logits = utils.build_mlp(
      hidden,
      self.n_classes,
      scope,
      n_layers=self.n_layers,
      size=self.layer_size,
      activation=tf.nn.relu,
      output_activation=None
    )
    logits -= tf.reduce_logsumexp(logits, axis=1, keep_dims=True)
    return logits

  def decoded_dist(self, state, belief_state):
    states = np.array([state, belief_state])
    decoded_states = self.decode_states(states)
    p = np.exp(np.swapaxes(decoded_states[0], 0, 1))
    q = np.exp(np.swapaxes(decoded_states[1], 0, 1))
    return -np.mean(scipy.stats.entropy(p, q))

  def update(self, obs, info={}, true_state=None):
    if true_state is not None:
      return super().update(obs, info=info, true_state=true_state)
    feed_dict = self.make_feed_dict(obs, self.hidden_c, self.hidden_h)
    logits, next_hidden = self.sess.run([self.logits, self.next_hidden], feed_dict=feed_dict)
    logits = logits[:, 0, 0, :]
    self.update_hidden_state(next_hidden)
    return logits


class LanderObsModel(TFModel):

  def __init__(
    self,
    *args,
    n_hidden=32,
    n_layers=0,
    policy=None,
    **kwargs
    ):

    super().__init__(*args, **kwargs)

    assert policy is not None

    self.n_hidden = n_hidden
    self.n_layers = n_layers

    self.data_keys = ['obses', 'actions']

    self.obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.act_ph = tf.placeholder(tf.float32, [None, self.env.n_actions])
    self.bal_weights_ph = tf.placeholder(tf.float32, [None])

    self.perceived_obs = self.build_model(self.obs_ph, self.scope)

    probs = policy(self.perceived_obs)
    eps = 1e-9
    logits = tf.log(probs + eps)
    likelihood_losses = -tf.reduce_sum(logits * self.act_ph, axis=1)
    likelihood_loss = tf.reduce_sum(self.bal_weights_ph * likelihood_losses) / tf.reduce_sum(self.bal_weights_ph)
    self.loss = likelihood_loss

    self.belief_state = None

  def build_model(self, obs, scope):
    raw = utils.build_mlp(
      obs[:, 4:5],
      1,
      scope,
      n_layers=self.n_layers,
      size=self.n_hidden,
      activation=tf.nn.relu,
      output_activation=tf.nn.sigmoid
    )
    ang = -np.pi + raw * 2 * np.pi
    perceived_obs = tf.concat([obs[:, :4], ang, obs[:, 5:]], axis=1)
    return perceived_obs

  def format_batch(self, batch, member_mask):
    bal_weights = utils.bal_weights_of_batch(np.argmax(batch['actions'], axis=1))
    feed_dict = {
      self.obs_ph: batch['obses'],
      self.act_ph: batch['actions'],
      self.bal_weights_ph: bal_weights
    }
    return feed_dict

  def format_train_data(self, data):
    fmted_data = deepcopy(data)
    fmted_data['actions'] = utils.batch_onehot_encode(data['actions'], self.env.n_actions)
    return fmted_data

  def train(self, data, *args, **kwargs):
    fmted_data = self.format_train_data(data)
    super().train(fmted_data, *args, **kwargs)

  def get_state(self):
    return self.belief_state

  def reset(self, obs, info={}):
    pass

  def update(self, obs, info={}):
    pass

  def belief(self, state, belief_state=None):
    return np.nan

  def compute_perceived_obses(self, obses):
    feed_dict = {self.obs_ph: obses}
    return self.sess.run(self.perceived_obs, feed_dict=feed_dict)

  def hypothetical_beliefs_in_true_state(self, perceived_obses, true_state):
    p = perceived_obses[:, 4]
    q = true_state[4]
    beliefs_in_true_state = -utils.angular_dist(p, q)
    return beliefs_in_true_state
