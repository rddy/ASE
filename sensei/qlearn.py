from __future__ import division

from copy import deepcopy

import numpy as np
import scipy
from matplotlib import pyplot as plt


def tabular_soft_q_iter(R, T, maxiter=5000, verbose=False, Q_init=None, learning_rate=1, ftol=0, gamma=0.99):
  '''Adapted from https://github.com/rddy/isql/blob/master/1.0-tabular-ime.ipynb'''

  T = deepcopy(T) # s',s,a
  T = np.exp(T)
  T = np.swapaxes(T, 0, 1) # s,s',a
  T = np.swapaxes(T, 1, 2) # s,a,s'

  n, m = R.shape[:2]
  Q = np.zeros((n, m)) if Q_init is None else deepcopy(Q_init)
  prevQ = deepcopy(Q)
  if verbose:
    diffs = []
  for iter_idx in range(maxiter):
    V = scipy.special.logsumexp(prevQ, axis=1)
    V_broad = V.reshape((1, 1, n))
    Q = np.sum(T * (R + gamma * V_broad), axis=2)
    Q = (1 - learning_rate) * prevQ + learning_rate * Q
    diff = np.mean((Q - prevQ)**2)/(np.std(Q)**2)
    if verbose:
      diffs.append(diff)
    if diff < ftol:
      break
    prevQ = deepcopy(Q)
  if verbose:
    plt.xlabel('Number of Iterations')
    plt.ylabel('Avg. Squared Bellman Error')
    plt.title('Soft Q Iteration')
    plt.plot(diffs)
    plt.yscale('log')
    plt.show()

  Q -= scipy.special.logsumexp(Q, axis=1, keepdims=True)

  return Q
