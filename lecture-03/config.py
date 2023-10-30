#!/usr/bin/env python

m = 1
g = 1
L = 1

dt = 0.1

def hamiltonian(theta, p):
    T = p * p / (2 * m * L*L)
    U = m * g * L * (1 - np.cos(theta))
    return T + U

def solve_leapfrog(theta, p, dt=dt):
    theta_half = theta + dt / 2 * p / (m * L*L)
    p_next = p - dt * m * g * L * np.sin(theta_half)
    theta_next = theta_half + dt / 2 * p_next / (m * L*L)
    return (theta_next, p_next)

# more stuff:

# for TUDa jupyter hub (https://tu-jupyter-i.ca.hrz.tu-darmstadt.de/)
# --> install dependencies via 
# $ pip install -r requirements_noversions.txt --prefix=`pwd`/requirements
import sys
sys.path.append('./requirements/lib/python3.8/site-packages/')

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

def set_axes(xlim=(-np.pi * 1.1, np.pi * 1.1), ylim=(-3, 3)):
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

def plot_hamiltonian(xlim=(-np.pi * 1.1, np.pi * 1.1), ylim=(-3, 3)):
    TH, PP = np.meshgrid(np.linspace(*xlim, num=100), 
                         np.linspace(*ylim, num=100))
    HH = hamiltonian(TH, PP)
    plt.contourf(TH, PP, HH, cmap=plt.get_cmap('hot_r'), levels=12, zorder=0)
    plt.colorbar(label=r'$\mathcal{H}(\theta, p)$')
    set_axes(xlim, ylim)
