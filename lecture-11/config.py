#!/usr/bin/env python

import warnings
from typing import Optional, Union

import time
import botorch
import gpytorch
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)

try:
    from botorch.fit import fit_gpytorch_mll
except ImportError:
    print ("You are using a botorch version older than 0.7.2,"
           " aliasing the legacy `fit_gpytorch_model` function")
    from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gym.spaces.utils import unflatten
from gym.wrappers import RescaleAction
from IPython.display import clear_output, display
from scipy.constants import c, e, m_p

from ares_ea.bo_utils import (
    ProximalAcquisitionFunction,
    get_new_bound,
    plot_acq_with_gp,
    sample_gp_posterior_plot,
    sample_gp_prior_plot,
    scale_action,
)
from ares_ea.env import ARESEA

np.random.seed(3)
warnings.filterwarnings("ignore")

#plt.rcParams['figure.figsize'] = 12, 7
#plt.rcParams['savefig.dpi'] = 300
#plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams["axes.titley"] = 1.05

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import seaborn as sns
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 3})
sns.set_style('ticks',
              {'grid.linestyle': 'none', 'axes.edgecolor': '0',
               'axes.linewidth': 1.2, 'legend.frameon': True,
               'xtick.direction': 'out', 'ytick.direction': 'out',
               'xtick.top': True, 'ytick.right': True,
              })

