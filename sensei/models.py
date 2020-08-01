# Adapted from https://github.com/rddy/ReQueST/blob/master/rqst/models.py

from __future__ import division

import pickle
import uuid
import os
import random

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from . import utils


class TFModel(object):

  def __init__(self,
               sess,
               env=None,
               n_ens_mem=1,
               scope_file=None,
               tf_file=None,
               scope=None,
               *args,
               **kwargs):

    # scope vs. scope in scope_file
    if scope is None:
      if scope_file is not None and os.path.exists(scope_file):
        with open(scope_file, 'rb') as f:
          scope = pickle.load(f)
      else:
        scope = str(uuid.uuid4())

    self.env = env
    self.sess = sess
    self.n_ens_mem = n_ens_mem
    self.tf_file = tf_file
    self.scope_file = scope_file
    self.scope = scope

    self.loss = None

  def save(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'wb') as f:
      pickle.dump(self.scope, f, pickle.HIGHEST_PROTOCOL)

    utils.save_tf_vars(self.sess, self.scope, self.tf_file)

  def load(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'rb') as f:
      self.scope = pickle.load(f)

    self.init_tf_vars()
    utils.load_tf_vars(self.sess, self.scope, self.tf_file)

  def init_tf_vars(self):
    utils.init_tf_vars(self.sess, [self.scope])

  def compute_batch_loss(self, feed_dict, update=True):
    if update:
      loss_eval, _ = self.sess.run([self.loss, self.update_op],
                                   feed_dict=feed_dict)
    else:
      loss_eval = self.sess.run(self.loss, feed_dict=feed_dict)
    return loss_eval

  def train(self,
            data,
            iterations=100000,
            ftol=1e-4,
            batch_size=32,
            learning_rate=1e-3,
            val_update_freq=100,
            bootstrap_prob=1.,
            verbose=False,
            show_plots=None):

    if self.loss is None:
      return

    if show_plots is None:
      show_plots = verbose

    var_list = utils.get_tf_vars_in_scope(self.scope)
    opt_scope = str(uuid.uuid4())
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=var_list)

    utils.init_tf_vars(self.sess, [self.scope, opt_scope])

    member_masks = [
        utils.onehot_encode(member_idx, self.n_ens_mem)
        for member_idx in range(self.n_ens_mem)
    ]

    def bootstrap(train_idxes, mem_idx):
      guar_idxes = [
          x for i, x in enumerate(train_idxes)
          if i % self.n_ens_mem == mem_idx
      ]
      nonguar_idxes = [
          x for i, x in enumerate(train_idxes)
          if i % self.n_ens_mem != mem_idx
      ]
      n_train_per_mem = int(np.ceil(bootstrap_prob * len(nonguar_idxes)))
      return guar_idxes + random.sample(nonguar_idxes, n_train_per_mem)

    train_idxes_key_of_mem = []
    for mem_idx in range(self.n_ens_mem):
      train_idxes_key = 'train_idxes_of_mem_%d' % mem_idx
      train_idxes_key_of_mem.append(train_idxes_key)
      data[train_idxes_key] = bootstrap(data['train_idxes'], mem_idx)

    val_losses = []
    val_batch = utils.sample_batch(
        size=len(data['val_idxes']),
        data=data,
        data_keys=self.data_keys,
        idxes_key='val_idxes')
    formatted_val_batch = self.format_batch(val_batch, utils.unif(self.n_ens_mem))

    if verbose:
      print('-----')
      print('iters total_iters train_loss val_loss')

    for t in range(iterations):
      for mem_idx, member_mask in enumerate(member_masks):
        batch = utils.sample_batch(
            size=batch_size,
            data=data,
            data_keys=self.data_keys,
            idxes_key=train_idxes_key_of_mem[mem_idx])

        formatted_batch = self.format_batch(batch, member_mask)
        train_loss = self.compute_batch_loss(formatted_batch, update=True)

      if t % val_update_freq == 0:
        val_loss = self.compute_batch_loss(formatted_val_batch, update=False)

        if verbose:
          print('%d %d %f %f' % (t, iterations, train_loss, val_loss))

        val_losses.append(val_loss)

        if ftol is not None and utils.converged(val_losses, ftol):
          break

    if verbose:
      print('-----\n')

    if show_plots:
      plt.xlabel('Gradient Steps')
      plt.ylabel('Validation Loss')
      grad_steps = np.arange(0, len(val_losses), 1) * val_update_freq
      plt.plot(grad_steps, val_losses)
      plt.show()
