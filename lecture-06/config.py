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

# CERN PS Simulation

def beta(gamma):
    '''Speed β in units of c from relativistic Lorentz factor γ.'''
    return np.sqrt(1 - gamma**-2)

def gamma(p):
    '''Relativistic Lorentz factor γ from total momentum p.'''
    return np.sqrt(1 + (p / (mass * c))**2)

charge = e
mass = m_p

class Machine(object):
    gamma_ref = 3.13
    circumference = 2 * np.pi * 100
    voltage = 200e3
    harmonic = 7
    alpha_c = 0.027
    phi_s = 0.456
    
    def __init__(self, gamma_ref=gamma_ref, circumference=circumference,
                 voltage=voltage, harmonic=harmonic, 
                 alpha_c=alpha_c, phi_s=phi_s):
        '''Override default settings by giving explicit arguments.'''
        self.gamma_ref = gamma_ref
        self.circumference = circumference
        self.voltage = voltage
        self.harmonic = harmonic
        self.alpha_c = alpha_c
        self.phi_s = phi_s
    
    def eta(self, deltap):
        '''Phase-slip factor for a particle.'''
        p = self.p0() + deltap
        return self.alpha_c - gamma(p)**-2

    def p0(self):
        '''Momentum of synchronous particle.'''
        return self.gamma_ref * beta(self.gamma_ref) * mass * c

    def update_gamma_ref(self):
        '''Advance the energy of the synchronous particle
        according to the synchronous phase by one turn.
        '''
        deltap_per_turn = charge * self.voltage / (
            beta(self.gamma_ref) * c) * np.sin(self.phi_s)
        new_p0 = self.p0() + deltap_per_turn
        self.gamma_ref = gamma(new_p0)

def track_one_turn(z_n, deltap_n, machine):
    m = machine
    # half drift
    z_nhalf = z_n - m.eta(deltap_n) * deltap_n / m.p0() * m.circumference / 2
    # rf kick
    amplitude = charge * m.voltage / (beta(gamma(m.p0())) * c)
    phi = m.phi_s - m.harmonic * 2 * np.pi * z_nhalf / m.circumference
    
    m.update_gamma_ref()
    deltap_n1 = deltap_n + amplitude * (np.sin(phi) - np.sin(m.phi_s))
    # half drift
    z_n1 = z_nhalf - m.eta(deltap_n1) * deltap_n1 / m.p0() * m.circumference / 2
    return z_n1, deltap_n1

def T(deltap, machine):
    '''Kinetic energy term in Hamiltonian.'''
    return -0.5 * machine.eta(deltap) / machine.p0() * deltap**2

def U(z, machine, beta_=None):
    '''Potential energy term in Hamiltonian.
    If beta is not given, compute it from synchronous particle.
    '''
    m = machine
    if beta_ is None:
        beta_ = beta(gamma(m.p0()))
    ampl = charge * m.voltage / (beta_ * c * 2 * np.pi * m.harmonic)
    phi = m.phi_s - 2 * np.pi * m.harmonic / m.circumference * z
    # convenience: define z at unstable fixed point
    z_ufp = -m.circumference * (np.pi - 2 * m.phi_s) / (2 * np.pi * m.harmonic)
    # convenience: offset by potential value at unstable fixed point
    # such that unstable fixed point (and separatrix) have 0 potential energy
    return ampl * (-np.cos(phi) + 
                   2 * np.pi * m.harmonic / m.circumference * (z - z_ufp) * np.sin(m.phi_s) +
                   -np.cos(m.phi_s))

def hamiltonian(z, deltap, machine):
    return T(deltap, machine) + U(z, machine, beta_=beta(gamma(machine.p0() + deltap)))

def plot_hamiltonian(machine, zleft=-50, zright=50, dpmax=0.01, cbar=True):
    '''Plot Hamiltonian contours across (zleft, zright) and (-dpmax, dpmax).'''
    Z, DP = np.meshgrid(np.linspace(zleft, zright, num=1000), 
                        np.linspace(-dpmax, dpmax, num=1000))
    H = hamiltonian(Z, DP * machine.p0(), machine) / machine.p0()
    
    plt.contourf(Z, DP, H, cmap=plt.get_cmap('hot_r'), levels=12,
                 zorder=0, alpha=0.5)
    plt.xlabel('$z$ [m]')
    plt.ylabel(r'$\delta$')
    if cbar:
        colorbar = plt.colorbar(label=r'$\mathcal{H}(z,\Delta p)\,/\,p_0$')
        colorbar.ax.axhline(0, lw=2, c='b')
    plt.contour(Z, DP, H, colors='b', linewidths=2, levels=[0])
    
def plot_rf_overview(machine):
    m = machine
    z_range = np.linspace(-150, 40, num=1000)
    # z location of unstable fixed point:
    z_ufp = -m.circumference * (np.pi - 2 * m.phi_s) / (2 * np.pi * m.harmonic)

    fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    plt.sca(ax[0])
    plt.plot(z_range, 1e-3 * m.voltage * np.sin(m.phi_s - 2 * np.pi * m.harmonic / m.circumference * z_range))
    plt.axhline(0, c='gray', lw=2)
    plt.axhline(1e-3 * m.voltage * np.sin(m.phi_s), c='purple', lw=2, ls='--')
    plt.axvline(0, c='purple', lw=2)
    plt.axvline(z_ufp, c='red', lw=2)
    plt.ylabel('rf wave $V(z)$ [kV]')

    plt.sca(ax[1])
    plt.plot(z_range, 1e6 * U(z_range, m) / m.p0())
    plt.axhline(0, c='gray', lw=2)
    plt.ylabel(r'$U(z)\,/\,p_0\cdot 10^6$')

    plt.scatter([z_ufp], [0], marker='*', c='white', edgecolor='red', zorder=10)
    plt.scatter([0], [U(0, m) / m.p0()], marker='d', c='white', edgecolor='purple', zorder=10)

    plt.sca(ax[2])
    plot_hamiltonian(m, zleft=z_range[0], zright=z_range[-1], cbar=False)
    plt.scatter([z_ufp], [0], marker='*', c='white', edgecolor='red', zorder=10)
    plt.scatter([0], [0], marker='d', c='white', edgecolor='purple')
    plt.xlabel('$z$ [m]')
    plt.ylabel('$\delta$')
    plt.subplots_adjust(hspace=0)
    
    return fig, ax

def emittance(z, deltap):
    N = len(z)
    
    # subtract centroids
    z = z - 1/N * np.sum(z)
    deltap = deltap - 1/N * np.sum(deltap)
    
    # compute Σ matrix entries
    z_sq = 1/N * np.sum(z * z)
    deltap_sq = 1/N * np.sum(deltap * deltap)
    crossterm = 1/N * np.sum(z * deltap)
    
    # determinant of Σ matrix
    epsilon = np.sqrt(z_sq * deltap_sq - crossterm * crossterm)
    return epsilon

def plot_dist(stat_dist_class, rfb, sigma_z=None, H0=None):
    '''Plot properties of stationary distribution class stat_dist_class
    for the given RF bucket and the bunch length (used for the guessed
    distribution Hamiltonian limit H0).
    Args:
        - stat_dist_class: Stationary Distribution class
          (e.g. rfbucket_matching.ThermalDistribution)
        - rfb: RFBucket instance
        - sigma_z: bunch length to be matched
    '''
    try:
        z_sfp = np.atleast_1d(rfb.z_sfp)
        H_sfp = rfb.hamiltonian(z_sfp, 0, make_convex=True)
        Hmax = max(H_sfp)
        stat_exp = stat_dist_class(
            lambda *args: rfb.hamiltonian(*args, make_convex=True), #always convex Hamiltonian
            Hmax=Hmax,
        )
    except TypeError:
        stat_exp = stat_dist_class
        
    if H0:
        stat_exp.H0 = H0
    else:
        stat_exp.H0 = rfb.guess_H0(sigma_z, from_variable='sigma')

    dpmax = rfb.dp_max(rfb.z_ufp_separatrix)
    zz = np.linspace(rfb.z_left, rfb.z_right, num=1000)
    Z, DP = np.meshgrid(zz, np.linspace(-dpmax*1.1, dpmax*1.1, num=500))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    plt.sca(ax[0, 0])
    plt.title('phase space distribution', fontsize=20, y=1.04)
    plt.contourf(Z, DP * 1e3, stat_exp.function(Z, DP), 20, cmap=plt.get_cmap('hot_r'))
    plt.colorbar().set_label(r'$\psi(z, \delta)$', fontsize=20)
    plt.plot(zz, rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.plot(zz, -rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$\delta$ [$10^{-3}$]', fontsize=20)

    plt.sca(ax[0, 1])
    plt.title('Hamiltonian contours', fontsize=20, y=1.04)
    plt.contourf(Z, DP * 1e3, -stat_exp.H(Z, DP), 20, cmap=plt.get_cmap('coolwarm'))
    plt.colorbar().set_label(r'$\mathcal{H}(z,\delta)$', fontsize=20)
    plt.plot(zz, rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.plot(zz, -rfb.separatrix(zz) * 1e3, c='purple', lw=2)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$\delta$ [$10^{-3}$]', fontsize=20)

    plt.sca(ax[1, 0])
    plt.title('line density', fontsize=20, y=1.04)
    plt.plot(zz, np.sum(stat_exp.function(Z, DP), axis=0), antialiased=False)
    plt.xlabel(r'$z$', fontsize=20)
    plt.ylabel(r'$\lambda(z) = \int\; d\delta \; \psi(z, \delta)$', fontsize=20)

    plt.sca(ax[1, 1])
    plt.title('Hamiltonian distribution', fontsize=20, y=1.04)
    hhs = -stat_exp.H(Z, DP).ravel()
    counts = stat_exp.function(Z, DP).ravel()
    perm = np.argsort(hhs)
    plt.plot(hhs[perm], counts[perm])
    plt.xlabel(r'$\mathcal{H}$', fontsize=20)
    plt.ylabel(r'$\psi(\mathcal{H})$', fontsize=20)
    plt.ylim(-0.1, 1.1)
    plt.axvline(0, *plt.ylim(), color='purple', lw=2)
    plt.text(0, np.mean(plt.ylim()), '\nseparatrix', color='purple', rotation=90, fontsize=12)

    plt.tight_layout()
    
    return fig

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
    plt.title('macro-particle generation', fontsize=20, y=1.04)
    return zz, Z, DP