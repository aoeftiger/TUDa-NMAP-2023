#!/usr/bin/env python

# more stuff:

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

from tqdm.notebook import tqdm, trange

def plot_rfwave(phi_s=0.5, regime='classical'):
    phi = np.linspace(-1.5, 7, 1000)

    plt.plot(phi, np.sin(phi), c='k')

    if regime == 'classical':
        focusing = 1
        c_philow = 'orange'
        c_phihigh = 'blue'
    elif regime == 'relativistic':
        focusing = -1
        c_philow = 'blue'
        c_phihigh = 'orange'
    else:
        ValueError('Did not recognise regime ("classical" or "relativistic").')

    focusing *= np.sign(np.cos(phi_s))

    plt.scatter([phi_s+0.4], [np.sin(phi_s+0.4)], c=c_phihigh, zorder=10)
    plt.annotate('', (phi_s+0.4 - focusing * 0.3 + 0.3, np.sin(phi_s + 0.1 + 0.3 - focusing * 0.3)), 
                 xytext=(phi_s+0.4 + 0.3, np.sin(phi_s+0.1 + 0.3)), zorder=10,
                 arrowprops={'width': 2, 'shrink': 0.1, 'color': c_phihigh})
    plt.scatter([phi_s-0.4], [np.sin(phi_s-0.4)], c=c_philow, zorder=10)
    plt.annotate('', (phi_s-0.4 + focusing * 0.3 + 0.3, np.sin(phi_s - 0.1 - 0.3 + focusing * 0.3)), 
                 xytext=(phi_s-0.4 + 0.3, np.sin(phi_s-0.1 - 0.3)), zorder=10,
                 arrowprops={'width': 2, 'shrink': 0.1, 'color': c_philow})

    plt.axvline(phi_s, c='gray', zorder=0)
    plt.axhline(np.sin(phi_s), c='gray', ls='--', zorder=0)
    
    plt.text(phi_s + 0.2, -0.15, r'$\varphi_s$', c='gray', fontsize='x-small')
    plt.text(-0.5, np.sin(phi_s) + 0.1, r'$\Delta W_0$', c='gray', ha='right', 
             fontsize='x-small', bbox={'color': 'white'})
    plt.text(phi_s + 0.2, -1.05, 'later', c='gray', fontsize='x-small', bbox={'color': 'white'})
    plt.text(phi_s - 0.2, -1.05, 'earlier', ha='right', c='gray', fontsize='x-small', bbox={'color': 'white'})
    
    plt.plot([np.pi - phi_s]*2, [0, np.sin(phi_s)], c='gray', ls=':', zorder=0)
    plt.text(np.pi - phi_s, -0.15, r'$\pi-\varphi_s$', c='gray', fontsize='x-small', ha='center')

    plt.xticks([2*np.pi], ["   $2\pi$"], fontsize='x-small')
    plt.yticks([])
    
    plt.text(7.5, -0.2, r'$\varphi$', c='k', ha='right');
    plt.text(-0.2, 1, r'$qV$', c='k', ha='right');

    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # make arrows
    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)
    
    return ax
