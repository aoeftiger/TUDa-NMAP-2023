#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 3})
sns.set_style('ticks',
              {'grid.linestyle': 'none', 'axes.edgecolor': '0',
               'axes.linewidth': 1.2, 'legend.frameon': True,
               'xtick.direction': 'out', 'ytick.direction': 'out',
               'xtick.top': True, 'ytick.right': True,
              })

from scipy.constants import m_p, c, e

from cpymad.madx import Madx

from scipy.interpolate import interp1d

def set_correctors(theta_vector, madx):
    madx.input('''
    k0nl_GS01MU1A = {};
    k0nl_GS02MU1A = {};
    k0nl_GS03MU1A = {};
    k0nl_GS04MU2A = {};
    k0nl_GS05MU1A = {};
    k0nl_GS06MU2A = {};
    k0nl_GS07MU1A = {};
    k0nl_GS08MU1A = {};
    k0nl_GS09MU1A = {};
    k0nl_GS10MU1A = {};
    k0nl_GS11MU1A = {};
    k0nl_GS12MU1A = {};
    '''.format(*theta_vector)
              )