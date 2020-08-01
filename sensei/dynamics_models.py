# Adapted from https://github.com/rddy/ReQueST/blob/master/rqst/dynamics_models.py

from __future__ import division

from copy import deepcopy
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from .models import TFModel
from . import utils

class MDNRNNDynamicsModel(TFModel):
  """Adapted from https://github.com/hardmaru/WorldModelsExperiments/blob/master/car/
  """

  def __init__(self,
               encoder,
               *args,
               grad_clip=1.0,
               num_mixture=5,
               use_layer_norm=0,
               use_recurrent_dropout=0,
               recurrent_dropout_prob=0.90,
               use_input_dropout=0,
               input_dropout_prob=0.90,
               use_output_dropout=0,
               output_dropout_prob=0.90,
               rnn_size=256,
               n_z_dim=32,
               n_act_dim=3,
               **kwargs):

    super().__init__(*args, **kwargs)

    self.encoder = encoder
    self.grad_clip = grad_clip
    self.num_mixture = num_mixture
    self.use_layer_norm = use_layer_norm
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_prob = recurrent_dropout_prob
    self.use_input_dropout = use_input_dropout
    self.input_dropout_prob = input_dropout_prob
    self.use_output_dropout = use_output_dropout
    self.output_dropout_prob = output_dropout_prob

    self.rnn_size = rnn_size
    self.n_z_dim = n_z_dim
    self.n_act_dim = n_act_dim

    self.input_x_ph = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, self.n_z_dim + self.n_act_dim])
    self.output_x_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, None, self.n_z_dim])
    self.seq_lens_ph = tf.placeholder(dtype=tf.int32, shape=[None])
    self.initial_state_c_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.rnn_size])
    self.initial_state_h_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.rnn_size])

    initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_state_c_ph,
                                                  self.initial_state_h_ph)

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      dyn_out_vars = self.build_dyn_net(
          self.input_x_ph,
          seq_lens=self.seq_lens_ph,
          initial_state=initial_state)
    self.__dict__.update(dyn_out_vars)

    self.loss = tf.reduce_mean(
        self.build_loss(self.out_logmix, self.out_mean, self.out_logstd,
                        self.output_x_ph))

    self.obs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.n_z_dim])
    self.act_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.n_act_dim])
    data = self.next_obs(self.obs_ph, self.act_ph, init_state=initial_state)
    self.next_obs_pred = data['next_obs_mean']
    self.next_state_pred = data['hidden_state']

  def build_loss(self, logmix, mean, logstd, output_x):
    shape = tf.shape(output_x)
    n_trajs = shape[0]
    traj_len = shape[1]
    shape = [n_trajs, traj_len, self.n_z_dim, self.num_mixture]

    mean = tf.reshape(mean, shape)
    logmix = tf.reshape(logmix, shape)
    logstd = tf.reshape(logstd, shape)
    output_x = tf.expand_dims(output_x, 3)

    v = logmix + utils.tf_lognormal(output_x, mean, logstd)
    v = tf.reduce_logsumexp(v, axis=3)
    v = tf.reduce_mean(v, axis=2)
    return -v

  def build_dyn_net(self, input_x, seq_lens=None, initial_state=None):
    cell_fn = utils.LSTMCellWrapper

    if self.use_recurrent_dropout:
      cell = cell_fn(
          self.rnn_size,
          layer_norm=self.use_layer_norm,
          dropout_keep_prob=self.recurrent_dropout_prob)
    else:
      cell = cell_fn(self.rnn_size, layer_norm=self.use_layer_norm)

    if self.use_input_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.input_dropout_prob)
    if self.use_output_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.output_dropout_prob)

    NOUT = self.n_z_dim * self.num_mixture * 3

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.rnn_size, NOUT])
      output_b = tf.get_variable('output_b', [NOUT])

    hidden_states, last_state = tf.nn.dynamic_rnn(
        cell,
        input_x,
        initial_state=initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN',
        sequence_length=seq_lens)
    c_states = hidden_states[0].c
    hidden_states = hidden_states[0].h

    output = tf.reshape(hidden_states, [-1, self.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, self.num_mixture * 3])
    out_logmix, out_mean, out_logstd = tf.split(output, 3, 1)
    out_logmix -= tf.reduce_logsumexp(out_logmix, 1, keepdims=True)

    return {
        'hidden_states': hidden_states,
        'c_states': c_states,
        'last_state': last_state,
        'out_logmix': out_logmix,
        'out_mean': out_mean,
        'out_logstd': out_logstd,
        'cell': cell
    }

  def build_dyn_out_vars(self, obs, act, init_state=None):
    """
    Args:
     obs: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_z_dim)
     act: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_act_dim)
    Returns:
     a dict containing the output of a call to build_dyn_net
    """
    # add dummy dim for n_trajs=1, concat obs/act
    input_x = tf.concat([obs, act], axis=2)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      if init_state is None:
        init_state = self.cell.zero_state(
            batch_size=tf.shape(obs)[0], dtype=tf.float32)
      dyn_out_vars = self.build_dyn_net(
          input_x, seq_lens=None, initial_state=init_state)
    return dyn_out_vars

  def next_obs(self,
               obs,
               act,
               init_state=None,
               temperature=1e-6,
               mixact_seq=None):
    """
    Args:
     obs: a tf.Tensor with dimensions (batch_size, n_z_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a dict, where 'next_obs_mean' maps to a tf.Tensor with dimensions (batch_size, n_z_dim)
    """
    dyn_out_vars = self.build_dyn_out_vars(
        tf.expand_dims(obs, 1), tf.expand_dims(act, 1), init_state=init_state)

    traj_len = 1
    n_trajs = tf.shape(obs)[0]
    shape = [n_trajs, traj_len, self.n_z_dim, self.num_mixture]

    mean = tf.reshape(dyn_out_vars['out_mean'], shape)

    def f(logmix):
      logmix = tf.reshape(logmix, shape)
      logmix -= tf.reduce_logsumexp(logmix, axis=3, keep_dims=True)
      return tf.exp(logmix)

    if mixact_seq is not None:
      mix_coeffs = f(mixact_seq)
    else:
      mix_coeffs = f(dyn_out_vars['out_logmix'] / temperature)

    mixed_means = tf.reduce_sum(mix_coeffs * mean, axis=3)

    next_obs = mixed_means[:, -1, :]
    hidden_state = dyn_out_vars['last_state']
    return {'next_obs_mean': next_obs, 'hidden_state': hidden_state}

  def compute_next_obs(self, obs, act, init_state=None, temperature=0.1):
    """
    Args:
     trajs: a np.array with dimensions (n_trajs, traj_len, n_z_dim + 2*rnn_size)
     acts: a np.array with dimensions (n_trajs, traj_len - 1, n_act_dim)
     init_state: either None, or a tuple containing
      (a np.array with dimensions (n_trajs, 2*rnn_size),
       a np.array with dimensions (n_trajs, 2*rnn_size))
    Returns:
     a dict, where
      'next_obs' maps to a np.array with dimensions (n_trajs, n_z_dim + 2*rnn_size)
       that contains encoded frames concatenated with hidden states
      'next_state' maps to a tuple containing (a np.array with dimensions  (n_trajs, 2*rnn_size),
       a np.array with dimensions (n_trajs, 2*rnn_size)) that just contains hidden states
    """
    n_trajs = obs.shape[0]
    if init_state is None:
      init_state_c = init_state_h = np.zeros((n_trajs, self.rnn_size))
    else:
      init_state_c, init_state_h = init_state

    feed_dict = {
        self.obs_ph: obs,
        self.act_ph: act,
        self.initial_state_c_ph: init_state_c,
        self.initial_state_h_ph: init_state_h
    }

    next_obs_pred, next_state_pred = self.sess.run(
        [self.next_obs_pred, self.next_state_pred], feed_dict=feed_dict)

    return {
        'next_obs':
            np.concatenate(
                (next_obs_pred, next_state_pred.c, next_state_pred.h), axis=1),
        'next_state':
            next_state_pred
    }

def load_wm_pretrained_rnn(encoder, sess):
  scope = 'mdn_rnn'
  dynamics_model = MDNRNNDynamicsModel(
      encoder,
      sess,
      learning_rate=0.0001,
      kl_tolerance=0.5,
      rnn_size=256,
      n_z_dim=32,
      scope=scope,
      scope_file=os.path.join(utils.car_data_dir, 'dyn_scope.pkl'),
      tf_file=os.path.join(utils.car_data_dir, 'dyn.tf'))
  jsonfile = os.path.join(utils.wm_dir, 'rnn', 'rnn.json')
  utils.load_wm_pretrained_model(jsonfile, scope, sess)
  return dynamics_model
