import logging.config
import math
import random
from enum import Enum
import scipy.optimize as opt
import gym

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym import spaces

from actor_critic.utils import twissReader


class e_trajectory(gym.Env):

    def __init__(self, max_episode_length=30, **kwargs):
        self.current_action = None
        self.initial_conditions = []
        self.__version__ = "0.0.1"
        logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.reward_scale = 1000  # 1000
        self.MAX_TIME = max_episode_length
        self.is_finalized = False
        self.current_episode = -1

        # For internal stats...
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.seed()
        self.twissH, self.twissV = twissReader.readAWAKEelectronTwiss()

        self.bpmsH = self.twissH.getElements("BP")
        self.bpmsV = self.twissV.getElements("BP")

        self.correctorsH = self.twissH.getElements("MCA")
        self.correctorsV = self.twissV.getElements("MCA")

        self.responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        self.responseV = self._calculate_response(self.bpmsV, self.correctorsV)

        self.positionsH = np.zeros(len(self.bpmsH.elements))
        self.settingsH = np.zeros(len(self.correctorsH.elements))
        self.positionsV = np.zeros(len(self.bpmsV.elements))
        self.settingsV = np.zeros(len(self.correctorsV.elements))

        # for ele in self.bpmsH:
        #     print('bpmEle', ele.name)
        #
        # for ele in self.correctorsH:
        #     print('kickEle', ele.name)

        self.plane = Plane.horizontal

        high = np.ones(len(self.correctorsH.elements))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        high = np.ones(len(self.bpmsH.elements))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.opt_results = []

        if 'action_scale' in kwargs:
            self.action_scale = kwargs.get('action_scale')
        else:
            self.action_scale = 3e-4

        if 'state_scale' in kwargs:
            self.state_scale = kwargs.get('state_scale')
        else:
            # ORIG M.S.: self.state_scale = 100  # worked well with this value.
            self.state_scale = 100. / 1.7  # Also works, and is what real machine env. is using
        self.kicks_0 = np.zeros(len(self.correctorsH.elements))

        self.threshold = -0.002*self.state_scale * self.reward_scale  #corresponds to what exactly?

    def step(self, action):

        state, reward = self._take_action(action)
        self.state = state

        self.action_episode_memory[self.current_episode].append(action)

        self.current_steps += 1
        if self.current_steps > self.MAX_TIME:
            self.is_finalized = True

        return_reward = reward * self.state_scale

        self.rewards[self.current_episode].append(return_reward)


        return_state = np.array(state * self.state_scale)
        if (return_reward > self.threshold) or (return_reward < 15*self.threshold):
            self.is_finalized = True

        return return_state, return_reward, self.is_finalized, {}

    def step_opt(self, action):
        state, reward = self._take_action(action,is_optimisation=True)
        self.state = state

        return_reward = reward * self.state_scale
        #print(return_reward)
        self.opt_results.append(return_reward)
        self.rewards[self.current_episode].append(return_reward)
        return return_reward

    def _take_action(self, action, is_optimisation = False):

        kicks = action * self.action_scale

        state, reward = self._get_state_and_reward(kicks, self.plane,is_optimisation)
        # print('reward', reward)

        return state, reward

    def _get_reward(self, trajectory):
        rms = self.reward_scale * np.sqrt(np.mean(np.square(trajectory)))
        return (rms * (-1.))

    def _get_state_and_reward(self, kicks, plane, is_optimisation):
        self.TOTAL_COUNTER += 1
        if (plane == Plane.horizontal):
            rmatrix = self.responseH

        if (plane == Plane.vertical):
            rmatrix = self.responseV

        state = self._calculate_trajectory(rmatrix, self.kicks_0+kicks)

        if(not is_optimisation):
            self.kicks_0 = self.kicks_0+kicks

        reward = self._get_reward(state)
        return state, reward

    def _calculate_response(self, bpmsTwiss, correctorsTwiss):
        bpms = bpmsTwiss.elements
        correctors = correctorsTwiss.elements
        bpms.pop(0)
        correctors.pop(-1)

        rmatrix = np.zeros((len(bpms), len(correctors)))

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if (bpm.mu > corrector.mu):
                    rmatrix[i][j] = math.sqrt(bpm.beta * corrector.beta) * math.sin(
                        (bpm.mu - corrector.mu) * 2. * math.pi)
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix


    def _calculate_trajectory(self, rmatrix, delta_settings):

        delta_settings = np.squeeze(delta_settings)
        return rmatrix.dot(delta_settings)

    def reset(self, init_outside_threshold=False):

        self.is_finalized = False

        while True:
            if (self.plane == Plane.horizontal):
                self.settingsH = np.random.uniform(-1., 1., len(self.settingsH))
                self.kicks_0 = self.settingsH * self.action_scale
            if (self.plane == Plane.vertical):
                self.settingsV = np.random.uniform(-1.,1.,len(self.settingsV))
                self.kicks_0 = self.settingsV * self.action_scale

            if (self.plane == Plane.horizontal):
                init_positions = np.zeros(len(self.positionsH))  # self.positionsH
                rmatrix = self.responseH

            if (self.plane == Plane.vertical):
                init_positions = np.zeros(len(self.positionsV))  # self.positionsV
                rmatrix = self.responseV

            state = self._calculate_trajectory(rmatrix, self.kicks_0)
            self.state = state

            if (self.plane == Plane.horizontal):
                self.positionsH = state

            if (self.plane == Plane.vertical):
                self.positionsV = state

            # Rescale for agent
            return_initial_state = np.array(state * self.state_scale)
            self.initial_conditions.append([return_initial_state])
            return_value = return_initial_state

            if init_outside_threshold:
                if self._get_reward(return_value) < self.threshold:
                    break
            else:
                break

        self.current_episode += 1
        self.current_steps = 0
        self.action_episode_memory.append([])
        self.rewards.append([])

        return return_value

    def seed(self, seed=None):
        random.seed(seed)

    def setPlane(self, plane):
        if (plane == Plane.vertical or plane == Plane.horizontal):
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def plot_trajectory(self, col='tab:red'):
        """ Pass the env instance to plot the current beam trajectory """
        plt.figure(figsize=(7, 4))
        scaling = 1000.
        plt.plot(self.state * scaling, marker='o', c=col)
        plt.xlabel('Beam position monitor index')
        plt.ylabel('Beam position (mm)')

        max_pos = np.max(np.abs(scaling * self.state)) * 1.2
        plt.ylim(-max_pos, max_pos)
        plt.show()


class Plane(Enum):
    horizontal = 0
    vertical = 1
