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

from tqdm.notebook import tnrange

from skimage.transform import radon, iradon

from PIL import Image

def plot_mp(z, dp, rfb, n_bins=40):
    dpmax = rfb.dp_max(rfb.z_ufp_separatrix)
    zz = np.linspace(rfb.z_left, rfb.z_right, num=1000)
    Z, DP = np.meshgrid(zz, np.linspace(-dpmax*1.1, dpmax*1.1, num=100))
    H = rfb.hamiltonian(Z, DP)
    plt.contour(Z, DP * 1e3, H, 20, cmap=plt.get_cmap('coolwarm_r'))
    # plt.scatter(z, dp, alpha=0.6)
    my_cmap = plt.get_cmap('hot_r').copy()
    my_cmap.set_under('w',1)
    plt.hist2d(z, dp * 1e3, bins=n_bins, cmap=my_cmap)
    plt.plot(zz, rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.plot(zz, -rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.xlim(rfb.z_left, rfb.z_right)
    plt.ylim(-dpmax*1.1 * 1e3, dpmax*1.1 * 1e3)
    plt.colorbar().set_label('# macro-particles', fontsize=20)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$\delta$ [$10^{-3}$]', fontsize=20)
    plt.title('macro-particle distribution', fontsize=20, y=1.04)
    return zz, Z, DP

def plot_tomo(phasespace, z_rec, dp_rec, rfb):
    dpmax = rfb.dp_max(rfb.z_ufp_separatrix)
    Z, DP = np.meshgrid(z_rec, dp_rec)
    H = rfb.hamiltonian(Z, DP)
    plt.contour(Z, DP * 1e3, H, 20, cmap=plt.get_cmap('coolwarm_r'))
    my_cmap = plt.get_cmap('hot_r').copy()
    my_cmap.set_under('w',1)
    plt.pcolormesh(Z, DP * 1e3, phasespace.T, cmap=my_cmap)
    plt.plot(z_rec, rfb.separatrix(z_rec) * 1e3, c='purple', lw=2)
    plt.plot(z_rec, -rfb.separatrix(z_rec) * 1e3, c='purple', lw=2)
    plt.xlim(rfb.z_left, rfb.z_right)
    plt.ylim(-dpmax*1.1 * 1e3, dpmax*1.1 * 1e3)
    plt.colorbar().set_label('norm. density', fontsize=20)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$\delta$ [$10^{-3}$]', fontsize=20)
    plt.title('tomographic reconstruction', fontsize=20, y=1.04)