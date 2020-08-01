from __future__ import division

import sys
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

from sensei import utils
from sensei import ase
from sensei import gw_user_study
from sensei import lander_user_study
from sensei import car_user_study

n_users = 12

# counterbalance order of unassisted vs. ASE conditions across 12 users
# cond_order_of_user = (['A'] * (n_users // 2)) + (['B'] * (n_users // 2))
# random.shuffle(cond_order_of_user)
cond_order_of_user = ['A', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'B', 'A', 'A', 'B']
assert len(cond_order_of_user) == n_users

def load_guide_evals(guide_evals_path):
  if os.path.exists(guide_evals_path):
    with open(guide_evals_path, 'rb') as f:
      guide_evals = pickle.load(f)
  else:
    guide_evals = {}
  return guide_evals

def dump_guide_evals(
  guide_evals_path,
  guide_evals,
  verbose=True
  ):
  with open(guide_evals_path, 'wb') as f:
    pickle.dump(guide_evals, f, pickle.HIGHEST_PROTOCOL)

  if verbose:
    print('-----')
    for k, v in guide_evals.items():
      print('guide: ' + k)
      ase.print_guide_perf(v['perf'])
      print('-----')
    print('')

def eval_guide(
  sess,
  guide_env,
  guide_name,
  guide_model,
  n_eval_rollouts
  ):
  return ase.evaluate_baseline_guides(
    sess,
    guide_env,
    {guide_name: guide_model},
    n_eval_rollouts=n_eval_rollouts
  )

def print_phase(phase, user_study):
  print('-----\nEntering phase %d' % phase)
  print('\nInstructions:')
  print(user_study.instructions[phase])
  print('-----\n')

def main():
  user_id = sys.argv[1]
  env_name = sys.argv[2]
  if env_name == 'lander':
    user_study = lander_user_study
    data_dir = utils.lander_data_dir
  elif env_name == 'gridworld':
    user_study = gw_user_study
    data_dir = utils.gw_human_data_dir
  elif env_name == 'car':
    user_study = car_user_study
    data_dir = utils.car_human_data_dir
  else:
    raise ValueError
  if len(sys.argv) > 3:
    phases = [int(phase) for phase in sys.argv[3:]]
  else:
    if env_name == 'gridworld':
      phases = list(range(5))
    elif env_name == 'lander':
      phases = [0, 1, 3, 4]
    elif env_name == 'car':
      phases = list(range(3))
  verbose = True

  sess = utils.make_tf_session(gpu_mode=False)

  user_data_dir = os.path.join(data_dir, user_id)
  if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

  exp_vars = user_study.build_exp(sess, user_data_dir)
  iden_guide_policy = exp_vars['iden_guide_policy']
  guide_env = exp_vars['guide_env']
  env = exp_vars['env']
  if env_name in ['lander', 'gridworld']:
    guide_model = exp_vars['guide_model']
    guide_train_kwargs = exp_vars['guide_train_kwargs']
  if env_name in ['gridworld', 'car']:
    naive_guide_model = exp_vars['naive_guide_model']
  guide_evals_path = os.path.join(user_data_dir, 'guide_evals.pkl')

  if 0 in phases:
    print_phase(0, user_study)

    guide_evals = load_guide_evals(guide_evals_path)

    env.set_practice_mode(True)
    guide_env.user_model.set_practice_mode(True)
    prac_guide_eval = eval_guide(sess, guide_env, 'iden', iden_guide_policy, user_study.n_prac_rollouts)

    prac_guide_eval['prac'] = prac_guide_eval['iden']
    del prac_guide_eval['iden']
    guide_evals.update(prac_guide_eval)
    dump_guide_evals(guide_evals_path, guide_evals, verbose=verbose)

  env.set_practice_mode(False)
  guide_env.user_model.set_practice_mode(False)

  if int(user_id) > 11:
    cond_order = 'A'
  else:
    cond_order = cond_order_of_user[int(user_id)]
  if cond_order == 'A':
    phase_order = [1, 2]
  elif cond_order == 'B':
    phase_order = [2, 1]
  else:
    raise ValueError

  def run_phase(phase):
    if phase == 1 and 1 in phases:
      print_phase(1, user_study)

      guide_evals = load_guide_evals(guide_evals_path)

      guide_evals.update(eval_guide(sess, guide_env, 'iden', iden_guide_policy, user_study.n_eval_rollouts))
      dump_guide_evals(guide_evals_path, guide_evals, verbose=verbose)
      input('Please fill out the "%s - After Phase 1" in the spreadsheet, then hit enter: ' % env_name.title())

    if phase == 2 and 2 in phases:
      assert env_name in ['gridworld', 'car']
      print_phase(2, user_study)

      guide_evals = load_guide_evals(guide_evals_path)

      guide_evals.update(eval_guide(sess, guide_env, 'naive', naive_guide_model, user_study.n_eval_rollouts))
      dump_guide_evals(guide_evals_path, guide_evals, verbose=verbose)
      input('Please fill out the "%s - After Phase 2" in the spreadsheet, then hit enter: ' % env_name.title())

  for phase in phase_order:
    run_phase(phase)

  if 3 in phases:
    assert env_name in ['gridworld', 'lander']
    print_phase(3, user_study)

    guide_evals = load_guide_evals(guide_evals_path)

    if env_name in ['gridworld', 'lander']:
      init_train_rollouts = guide_evals['iden']['rollouts']
      if 'naive' in guide_evals:
        init_train_rollouts.extend(guide_evals['naive']['rollouts'])

      if env_name == 'gridworld':
        # pool the data from the first k users to train the model for the k-th participant,
        # since we do not expect the model to vary between users,
        # and because the small amount of data collected for each individual user
        # was too noisy to learn an accurate model from.
        prev_users = list(range(int(user_id))) + [12]
        # user 12 = user from pilot study, where we debugged the experiments
        # and collected data to help train the user model for the first few users
        for prev_user_id in prev_users:
          prev_user_data_dir = os.path.join(data_dir, str(prev_user_id))
          prev_guide_evals_path = os.path.join(prev_user_data_dir, 'guide_evals.pkl')
          prev_guide_evals = load_guide_evals(prev_guide_evals_path)
          init_train_rollouts.extend(prev_guide_evals['iden']['rollouts'])
          if 'naive' in prev_guide_evals:
            init_train_rollouts.extend(prev_guide_evals['naive']['rollouts'])

      guide_optimizer = ase.InteractiveGuideOptimizer(sess, env, guide_env)

      input('Press enter to continue: ')
      print('')

      train_logs = [guide_optimizer.run(
        guide_model,
        n_train_batches=user_study.n_train_batches,
        n_rollouts_per_batch=user_study.n_rollouts_per_batch,
        guide_train_kwargs=guide_train_kwargs,
        verbose=verbose,
        init_train_rollouts=init_train_rollouts,
        n_eval_rollouts=None
      )]

      train_logs_path = os.path.join(user_data_dir, 'train_logs.pkl')
      with open(train_logs_path, 'wb') as f:
        pickle.dump(train_logs, f, pickle.HIGHEST_PROTOCOL)

      guide_model.save()

  if 4 in phases:
    assert env_name in ['gridworld', 'lander']
    print_phase(4, user_study)

    guide_evals = load_guide_evals(guide_evals_path)

    init_train_rollouts = guide_evals['iden']['rollouts']
    guide_optimizer = ase.InteractiveGuideOptimizer(sess, env, guide_env)
    guide_optimizer.run(
      guide_model,
      n_train_batches=0,
      n_rollouts_per_batch=0,
      guide_train_kwargs={'iterations': 0, 'verbose': False},
      verbose=verbose,
      init_train_rollouts=init_train_rollouts,
      n_eval_rollouts=None
    )
    guide_model.load()

    guide_evals.update(eval_guide(sess, guide_env, 'learned', guide_model, user_study.n_eval_rollouts))
    input('Please fill out the "%s - After Phase 4" survey in the spreadsheet, then hit enter: ' % env_name.title())

    dump_guide_evals(guide_evals_path, guide_evals, verbose=verbose)

if __name__ == '__main__':
  main()
