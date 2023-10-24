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

def emittance(theta, p):
    N = len(theta)
    
    # subtract centroids
    theta = theta - 1/N * np.sum(theta)
    p = p - 1/N * np.sum(p)
    
    # compute Σ matrix entries
    theta_sq = 1/N * np.sum(theta * theta)
    p_sq = 1/N * np.sum(p * p)
    crossterm = 1/N * np.sum(theta * p)
    
    # determinant of Σ matrix
    epsilon = np.sqrt(theta_sq * p_sq - crossterm * crossterm)
    return epsilon

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

def plot_macro_evolution(results_thetas, results_ps):
    n_steps, N = results_thetas.shape
    
    centroids_theta = 1/N * np.sum(results_thetas, axis=1)
    centroids_p = 1/N * np.sum(results_ps, axis=1)
    
    var_theta = 1/N * np.sum(results_thetas * results_thetas, axis=1)
    var_p = 1/N * np.sum(results_ps * results_ps, axis=1)

    results_emit = np.zeros(n_steps, dtype=np.float32)
    for k in range(n_steps):
        results_emit[k] = emittance(results_thetas[k], results_ps[k])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    plt.sca(ax[0])
    plt.plot(centroids_theta, label=r'$\langle\theta\rangle$')
    plt.plot(centroids_p, label=r'$\langle p\rangle$')
    plt.scatter([0, 0], [centroids_theta[0], centroids_p[0]], marker='*', c='k')
    plt.xlabel('Steps $k$')
    plt.ylabel('Centroid amplitude')
    plt.legend()
    
    plt.sca(ax[1])
    plt.plot(var_theta, label=r'$\langle\theta^2\rangle$')
    plt.plot(var_p, label=r'$\langle p^2\rangle$')
    plt.scatter([0, 0], [var_theta[0], var_p[0]], marker='*', c='k')
    plt.xlabel('Steps $k$')
    plt.ylabel('Variance')
    plt.legend()
    
    plt.sca(ax[2])
    plt.plot(results_emit)
    plt.scatter([0], [results_emit[0]], marker='*', c='k')
    plt.xlabel('Steps $k$')
    plt.ylabel('RMS emittance $\epsilon$')
    
    plt.tight_layout()
    
    return ax

def get_boundary_ids(thetas, ps):
    psf = ps.flatten(); tsf = thetas.flatten()
    ps_min = ps.min(); ps_max = ps.max(); thetas_min = thetas.min(); thetas_max = thetas.max()

    seq = np.arange(len(psf))
    i_right = seq[(psf == 0) * (tsf == thetas_max)]
    i_bottom = seq[psf == ps_min]
    i_left = seq[(psf == 0) * (tsf == thetas_min)]
    i_top = seq[psf == ps_max]
    
    return np.concatenate((i_right, i_top, i_left, i_bottom, i_right))
