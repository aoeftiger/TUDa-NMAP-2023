import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_greedy_policy(policy, target, fire, ax):
    """ Add greedy actions to maze. """
    for k, v in policy.items():
        if (k == target).all():
            continue
            
        coords = {
            'up': (k[0] + 0.5, k[1] + 0.25, 0., 0.5),
            'down': (k[0] + 0.5, k[1] + 0.75, 0., -0.5),
            'right': (k[0] + 0.25, k[1] + 0.5, 0.5, 0.),
            'left': (k[0] + 0.75, k[1] + 0.5, -0.5, 0.)
        }

        ax.arrow(*coords[v],
                 color='tab:red', length_includes_head=True,
                 head_width=0.2, head_length=0.2)


def plot_v_table(v_table, target, fire, ax):
    """ Add V-values to maze. """
    for k, v in v_table.items():
        if (k == target).all():
            continue
        ax.text(k[0] + 0.5, k[1] + 0.5, s=f'{v:.1f}', ha='center', va='center',
                color='tab:red', fontsize=12)
        

def plot_q_table(q_table, target, fire, ax):
    """ Show Q-values for each state, action pair. """
    for k, v in q_table.items():
        if (k == target).all():
            continue

        # Highlight preferred action (according to Q-values)
        cols = {}
        preferred_direction = max(v, key=v.get)
        for d in ['up', 'down', 'right', 'left']:
            cols[d] = 'tab:blue'
            if d == preferred_direction:
                cols[d] = 'tab:red'

        # UP
        ax.arrow(k[0] + 0.5, k[1] + 0.55, 0., 0.15,
                 color=cols['up'], length_includes_head=True,
                 head_width=0.1, head_length=0.1)
        q_up = v['up']
        ax.text(k[0] + 0.5, k[1] + 0.85, s=f'{q_up:.1f}', ha='center', va='center',
                color=cols['up'], fontsize=10)

        # DOWN
        ax.arrow(k[0] + 0.5, k[1] + 0.45, 0., -0.15,
                 color=cols['down'], length_includes_head=True,
                 head_width=0.1, head_length=0.1)       
        q_down = v['down']
        ax.text(k[0] + 0.5, k[1] + 0.15, s=f'{q_down:.1f}', ha='center', va='center',
                color=cols['down'], fontsize=10)

        # RIGHT
        ax.arrow(k[0] + 0.55, k[1] + 0.5, 0.15, 0.,
                 color=cols['right'], length_includes_head=True,
                 head_width=0.1, head_length=0.1)
        q_right = v['right']
        ax.text(k[0] + 0.8, k[1] + 0.6, s=f'{q_right:.1f}', ha='center', va='center',
                color=cols['right'], fontsize=10)

        # LEFT
        ax.arrow(k[0] + 0.45, k[1] + 0.5, -0.15, 0.,
                 color=cols['left'], length_includes_head=True,
                 head_width=0.1, head_length=0.1)
        q_left = v['left']
        ax.text(k[0] + 0.2, k[1] + 0.6, s=f'{q_left:.1f}', ha='center', va='center',
                color=cols['left'], fontsize=10)


def print_qtable(q_table):
    """ Pretty print the Q-table values. """
    tab = PrettyTable([['s \ a', 'up', 'down', 'left', 'right']][0])
    for k, v in q_table.items():
        vals = [k]
        for a in ['up', 'down', 'left', 'right']:
            vals.append(np.round(v[a], 1))
        tab.add_rows([vals])
    print(tab)
