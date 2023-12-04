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

import sys
from cpymad.madx import Madx

from scipy.interpolate import interp1d

import PyNAFF

import pysixtrack
from pysixtrack import elements

def M_drift(L):
    return np.array([
        [1, L],
        [0, 1]
    ])

def M_dip_x(L, rho0):
    return np.array([
        [np.cos(L / rho0), rho0 * np.sin(L / rho0)],
        [-1 / rho0 * np.sin(L / rho0), np.cos(L / rho0)]
    ])

def M_dip_y(L, rho0):
    return M_drift(L)

def M_quad_x(L, k):
    ksq = np.sqrt(k + 0j)
    return np.array([
        [np.cos(ksq * L), 1 / ksq * np.sin(ksq * L)],
        [-ksq * np.sin(ksq * L), np.cos(ksq * L)]
    ]).real

def M_quad_y(L, k):
    ksq = np.sqrt(k + 0j)
    return np.array([
        [np.cosh(ksq * L), 1 / ksq * np.sinh(ksq * L)],
        [ksq * np.sinh(ksq * L), np.cosh(ksq * L)]
    ]).real

def track(M, u, up):
    '''Apply M to each individual [u;up] vectors value.'''
    return np.einsum('ij,...j->i...', M, np.vstack((u, up)).T)

def track_sext_4D(x, xp, y, yp, mL):
    xp += 0.5 * mL * (y * y - x * x)
    yp += mL * x * y
    return x, xp, y, yp
