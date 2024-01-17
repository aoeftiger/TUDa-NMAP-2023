#!/usr/bin/env python

# for jupyterhub at TUDa
import sys
sys.path.append('./requirements/lib/python3.8/site-packages/')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk', font_scale=1., rc={'lines.linewidth': 3})
sns.set_style('ticks',
              {'grid.linestyle': 'none', 'axes.edgecolor': '0',
               'axes.linewidth': 1.2, 'legend.frameon': True,
               'xtick.direction': 'out', 'ytick.direction': 'out',
               'xtick.top': True, 'ytick.right': True,
              })

from scipy.constants import m_p, c, e

from qlearning.plot_utils import (
    print_qtable, plot_q_table, plot_greedy_policy)
from qlearning.core import Maze, QLearner
from actor_critic.awake_env import e_trajectory
from actor_critic.core import (
    ClassicalDDPG, trainer, plot_training_log, run_correction)
