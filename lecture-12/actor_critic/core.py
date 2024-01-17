import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

from tqdm.notebook import tqdm

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# Loosely based on:
# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
# Generator functions for classical actor and critic models
def generate_classical_critic(n_dims_state_space: int, n_dims_action_space: int, hidden_layers):
    """ Initializes DDPG critic network represented by classical neural
    network.
    :param n_dims_state_space: number of dimensions of state space.
    :param n_dims_action_space: number of dimensions of action space.
    :return: keras dense feed-forward network model. """
    input_state = Input(shape=n_dims_state_space)
    input_action = Input(shape=n_dims_action_space)
    x = concatenate([input_state, input_action], axis=-1)
    for i, j in enumerate(hidden_layers[:-1]):
        x = Dense(j, activation=tf.keras.activations.relu)(x)
    x = Dense(hidden_layers[-1], activation=tf.keras.activations.linear)(x)

    return tf.keras.Model([input_state, input_action], x)


def generate_classical_actor(n_dims_state_space: int, n_dims_action_space: int, hidden_layers):
    """ Initializes DDPG actor network represented by classical neural
    network.
    :param n_dims_state_space: number of dimensions of state space.
    :param n_dims_action_space: number of dimensions of action space.
    :return: keras dense feed-forward network model. """
    input_state = Input(shape=n_dims_state_space)
    x = input_state
    for i in hidden_layers:
        x = Dense(i, activation=tf.nn.relu)(x)
    x = Dense(n_dims_action_space, activation='tanh')(x)
    return tf.keras.Model(input_state, x)


class ReplayBuffer:
    """ Implements simple replay buffer for experience replay. """

    def __init__(self, size, state_dim, action_dim):
        self.obs1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def push(self, obs, act, rew, next_obs, done):
        """ Add a new experience to the buffer. """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = np.asarray([rew])
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """ Sample batch_size memories from the buffer. """
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1),
                temp_dict['s2'], temp_dict['d'])


class ClassicalDDPG:
    def __init__(self, state_space, action_space, gamma=0.99,
                 tau_critic=1e-2, tau_actor=1e-2,
                 learning_rate_critic=1e-3, learning_rate_actor=1e-3,
                 grad_clip_actor=np.inf, grad_clip_critic=np.inf):
        """ Implements the classical DDPG agent where both actor and critic
        are represented by classical neural networks.
        :param state_space: openAI gym env state space
        :param action_space: openAI gym env action space
        :param gamma: reward discount factor
        :param tau_critic: soft update factor for critic target network
        :param tau_actor: soft update factor for actor target network
        :param learning_rate_schedule_critic: learning rate schedule for critic.
        :param learning_rate_schedule_actor: learning rate schedule for actor.
        """
        self.step_count = 0

        self.n_dims_state_space = len(state_space.high)
        self.n_dims_action_space = len(action_space.high)

        # Some main hyperparameters
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor

        self.grad_clip_actor = grad_clip_actor
        self.grad_clip_critic = grad_clip_critic

        # Main and target actor network initialization
        # ACTOR
        actor_hidden_layers = [400, 300]
        self.main_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space, actor_hidden_layers)
        self.target_actor_net = generate_classical_actor(
            self.n_dims_state_space, self.n_dims_action_space, actor_hidden_layers)

        # CRITIC
        critic_hidden_layers = [400, 300, 1]
        self.main_critic_net_1 = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space, critic_hidden_layers)
        self.target_critic_net_1 = generate_classical_critic(
            self.n_dims_state_space, self.n_dims_action_space, critic_hidden_layers)

        # Copy weights from main to target nets
        self.target_actor_net.set_weights(self.main_actor_net.get_weights())
        self.target_critic_net_1.set_weights(self.main_critic_net_1.get_weights())

        # Optimizers
        self.actor_optimizer = Adam(
            learning_rate_actor, amsgrad=False, epsilon=1e-8, beta_1=0.9, beta_2=0.999)
        self.critic_optimizer = Adam(
            learning_rate_critic, amsgrad=False, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            size=int(1e6), state_dim=self.n_dims_state_space,
            action_dim=self.n_dims_action_space)

    def get_proposed_action(self, state):
        """ Use actor network to obtain proposed action for en input state. """
        return self.main_actor_net.predict(state.reshape(1, -1), verbose=0)[0]

    def update(self, batch_size, *args):
        """ Calculate and apply the updates of the critic and actor
        networks based on batch of samples from experience replay buffer. """
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        s2 = np.asarray(s2, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)

        grads_critic_1 = self._get_gradients_critic(s, a, r, s2, d)
        self.critic_optimizer.apply_gradients(
            zip(grads_critic_1, self.main_critic_net_1.trainable_variables))

        # TD3 feature: delay, not active
        if self.step_count % 1 == 0:
            grads_actor = self._get_gradients_actor(s, batch_size)
            self.actor_optimizer.apply_gradients(
                zip(grads_actor, self.main_actor_net.trainable_variables))

            # Apply Polyak updates
            self._update_target_networks()

        self.step_count += 1

    def _get_gradients_critic(self, state, action, reward, next_state, d):
        """ Update the main critic network based on given batch of input
        states. """
        with tf.GradientTape() as tape:
            next_action = self.target_actor_net(next_state)

            q1 = self.target_critic_net_1([next_state, next_action])
            q_target = (reward + self.gamma * (1. - d) * q1)

            q_vals_1 = self.main_critic_net_1([state, action])
            q_loss_1 = tf.math.reduce_mean((q_target - q_vals_1) ** 2)

        grads_q_1 = tape.gradient(q_loss_1, self.main_critic_net_1.trainable_variables)

        return grads_q_1

    def _get_gradients_actor(self, states, batch_size):
        """ Update the main actor network based on given batch of input
        states. """
        with tf.GradientTape() as tape2:
            A_mu = self.main_actor_net(states)
            Q_mu = self.main_critic_net_1([states, A_mu])
            mu_loss = -tf.math.reduce_mean(Q_mu)
        grads_mu = tape2.gradient(mu_loss,
                                  self.main_actor_net.trainable_variables)
        return grads_mu

    def _update_target_networks(self):
        """ Apply Polyak update to both target networks. """
        # CRITICS
        target_weights = self.target_critic_net_1.get_weights()
        main_weights = self.main_critic_net_1.get_weights()
        new_target_weights = []
        for i in range(len(target_weights)):
            new_weights = (
                self.tau_critic * main_weights[i] +
                (1 - self.tau_critic) * target_weights[i]
            )
            new_target_weights.append(new_weights)
        self.target_critic_net_1.set_weights(new_target_weights)

        # ACTOR
        target_weights = self.target_actor_net.get_weights()
        main_weights = self.main_actor_net.get_weights()
        new_target_weights = []
        for i in range(len(target_weights)):
            new_weights = (
                self.tau_actor * main_weights[i] +
                (1 - self.tau_actor) * target_weights[i]
            )
            new_target_weights.append(new_weights)
        self.target_actor_net.set_weights(new_target_weights)

    def save_actor_critic_weights(self, filename):
        """ Save the weights of all four networks for reloading them
        later on. """
        self.main_actor_net.save_weights(filename + "_main_actor")
        self.target_actor_net.save_weights(filename + "_target_actor")
        self.main_critic_net_1.save_weights(filename + "_main_critic")
        self.target_critic_net_1.save_weights(filename + "_target_critic")


    def load_actor_critic_weights(self, filename):
        """ Load the weights of all four networks from file. """
        self.main_actor_net.load_weights(filename + "_main_actor")
        self.target_actor_net.load_weights(filename + "_target_actor")
        self.main_critic_net_1.load_weights(filename + "_main_critic")
        self.target_critic_net_1.load_weights(filename + "_target_critic")


def trainer(env, agent, n_steps, batch_size=32, action_noise=0.1,
            epsilon_greedy_i=0.9, epsilon_greedy_f=0.0, n_exploration_steps=0):
    """ Convenience function to run training with DDPG.
    :param env: openAI gym environment instance
    :param agent: ddpg instance
    :param n_steps: max. number of steps that training will run for
    :param batch_size: number of samples drawn from experience replay buffer
    at every step.
    :param action_noise: how much action noise to add
    :param n_exploration_steps: number of initial random steps in env.
    """
    episode_log = {
        'initial_rewards': [], 'final_rewards': [], 'n_total_steps': [],
        'n_random_steps': [], 'max_steps_above_reward_threshold': []
    }

    max_steps_per_episode = env.MAX_TIME
    epsilon_greedy_delta = (epsilon_greedy_i - epsilon_greedy_f) / n_steps

    n_total_steps_training = 0
    episode = 0
    pbar = tqdm(total=n_steps)
    while n_total_steps_training < n_steps:

        n_random_steps_episode = 0
        n_steps_episode = 0

        state = env.reset()
        # try:
        #     episode_log['initial_rewards'].append(env.calculate_reward(
        #         env.calculate_state(env.kick_angles)))
        # except AttributeError:
        episode_log['initial_rewards'].append(env._get_reward(state))

        # Episode loop
        epsilon = epsilon_greedy_i - epsilon_greedy_delta * n_total_steps_training

        for _ in range(max_steps_per_episode):
            eps_sample = np.random.uniform(0, 1, 1)
            if ((n_total_steps_training < n_exploration_steps) or
                    (eps_sample <= epsilon)):
                action = env.action_space.sample()
                n_random_steps_episode += 1
            else:
                action = agent.get_proposed_action(state)

            # NEW: ADD ACTION NOISE IN ANY CASE, DURING RANDOM EXPLORATION AND WHEN SAMPLING
            # FOLLOWING POLICY.
            action += action_noise * np.random.randn(agent.n_dims_action_space)
            action = np.clip(action, -1., 1.)

            next_state, reward, done, _ = env.step(action)

            n_steps_episode += 1
            n_total_steps_training += 1

            # Fill replay buffer
            terminal = done
            if n_steps_episode == max_steps_per_episode:
                done = True
                terminal = False
            agent.replay_buffer.push(state, action, reward, next_state, terminal)

            # EPISODE DONE
            if done:
                # Append log
                episode_log['final_rewards'].append(reward)
                episode_log['n_total_steps'].append(n_steps_episode)
                episode_log['n_random_steps'].append(n_random_steps_episode)

                # try:
                #     rew_thresh = env.reward_threshold
                # except AttributeError:
                rew_thresh = env.threshold
                if episode % 50 == 0:
                    print(f"EPISODE: {episode}, INITIAL REWARD: "
                          f"{round(episode_log['initial_rewards'][-1], 3)}, "
                          f"FINAL REWARD: "
                          f"{round(episode_log['final_rewards'][-1], 3)}, "
                          f"#STEPS: {n_steps_episode}."
                    )
                episode += 1

                if n_total_steps_training > n_exploration_steps:
                    # TRAINING ON RANDOM BATCHES FROM REPLAY BUFFER
                    for __ in range(n_steps_episode):
                        agent.update(batch_size, n_total_steps_training)
                break

            state = next_state
        pbar.update(n_steps_episode)

    return episode_log


def plot_training_log(env, agent, data, apply_scaling=True):
    """ Plot log data from agent training. """

    fig1, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    n_training_episodes = len(data['final_rewards'])
    if apply_scaling:
        # Undo all scalings that have been applied to the reward and multiply by
        # 1'000 to get from [m] -> [mm]
        scaling = 1. / (env.state_scale * env.reward_scale) * 1000
    else:
        scaling = 1.

    axs[0].plot(data['n_total_steps'], c='orange')
    axs[1].plot(np.array(data['initial_rewards'])[-n_training_episodes:] * scaling,
                c='tab:red', label='At episode start')
    axs[1].plot(np.array(data['final_rewards']) * scaling, c='lightsteelblue',
                label='At episode end')
    # try:
    #     axs[1].axhline(env.reward_threshold * scaling, color='k', ls='--')
    # except AttributeError:
    axs[1].axhline(env.threshold * scaling, color='k', ls='--')

    axs[0].set_ylabel('Episode length')
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Episode')
    # axs[0].legend(loc='upper right')
    axs[1].legend(loc='lower left')
    plt.tight_layout()
    plt.show()


def run_correction(env, agent):
    """ Run one correction and plot trajectory before and after. """
    scaling = 1. / (env.state_scale) * 1000

    plt.figure(figsize=(7, 4))
    state = env.reset(init_outside_threshold=True)
    plt.plot(scaling*state, marker='o', c='tab:red', label='Before correction')
    max_pos = np.max(np.abs(scaling * state))

    while True:
        a = agent.get_proposed_action(state)
        state, reward, done, _ = env.step(a)
        if done:
            break

    plt.plot(scaling*state, marker='D', c='lightsteelblue', label='After correction')
    
    max_pos = max(max_pos, np.max(np.abs(scaling * state))) * 1.2
    plt.ylim(-max_pos, max_pos)
    plt.legend(fontsize=13)
    plt.xlabel('Beam position monitor')
    plt.ylabel('Beam position (mm)')
    plt.show()
