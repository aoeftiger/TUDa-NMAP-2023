import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange
import json

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential


class Maze(object):
    """ Define a simple grid maze environment. """
    def __init__(self, height=4, width=4, target_position=None,
                 fire_reward=-10, fire_positions=None):
        self.height = height
        self.width = width
        self.previous_step_log = None
    
        if not target_position:
            target_position = [width - 1, height - 1]

        # Make sure target is not outside of maze boundaries
        if target_position[0] > width:
            target_position[0] = width - 1
        if target_position[1] > height:
            target_position[1] = height - 1

        # Initialize fields with fire
        if not fire_positions:
            fire_filling_factor = 0.2
            n_fires = int(self.height * self.width * fire_filling_factor)
            fire_positions = []
            for i in range(n_fires):
                while True:
                    x, y = self._get_random_position()
                    if (np.array([x, y]) == np.array(target_position)).all():
                        continue

                    clashing = False
                    for fp in fire_positions:
                        if (np.array([x, y]) == np.array(fp)).all():
                            clashing = True

                    if not clashing:
                        fire_positions.append(np.array([x, y]))
                        break

        self.target_position = np.asarray(target_position)
        self.fire_positions = np.asarray(fire_positions)

        self.fire_reward = fire_reward

        self.player_position = None
        self.reset()

        self.action_mapper = {
            'up': np.asarray([0, 1]),
            'down': np.asarray([0, -1]),
            'right': np.asarray([1, 0]),
            'left':  np.asarray([-1, 0]),
        }
    
    def step(self, action):
        """ Take a step in the environment by providing one of the
        actions 'up', 'down', 'left', or 'right'. """
        reward = 0. # by default, reward is 0 per step
        done = False
        
        action_vect = self.action_mapper[action]
        old_state = self.player_position
        new_state = self.player_position + action_vect

        # Check if bumped into wall
        if self._out_of_bounds(new_state):
            reward = -5
            new_state = old_state
        else:
            self.player_position = new_state
            reward = -1

        # Check if on any of the special fields
        for fp in self.fire_positions:
            if (self.player_position == fp).all():
                reward = self.fire_reward

        if (self.player_position == self.target_position).all():
            reward = 30
            done = True
            
        self.previous_step_log = (old_state, action, reward, new_state, done)

        return old_state, action, reward, new_state, done
    
    def _out_of_bounds(self, position):
        """ Check whether player position is at the edge of the maze """
        if ((position[0] < 0) or (position[0] >= self.width) or
                (position[1] < 0) or (position[1] >= self.height)):
            return True
        return False
    
    def reset(self, player_position=None):
        """ Reset the environment: place player randomly on a new field.
        Special fields are excluded. """
        if player_position:
            self.player_position = player_position
            return player_position

        while True:
            initial_position = self._get_random_position()

            on_fire_field = False
            for fp in self.fire_positions:
                if (initial_position == fp).all():
                    on_fire_field = True

            if not ((initial_position == self.target_position).all() or
                    on_fire_field):
                break
        self.player_position = initial_position
        return initial_position
    
    def _get_random_position(self):
        """ Return two random integers within boundaries of maze """
        x = np.random.randint(low=0, high=self.width, size=1)[0]
        y = np.random.randint(low=0, high=self.height, size=1)[0]
        return np.asarray([x, y])
    
    def plot(self, add_player_position=True, title=True):
        """ Plot main maze grid incl. special fields. """

        # Title shows the last step, if it exists
        if self.previous_step_log and title:
            state, action, reward, new_state, done = self.previous_step_log
            title = f'old_state: {state}\naction: {action}\nreward: {reward}\n'
            title += f'new_state: {new_state}\ndone: {done}'

        fig = plt.figure(figsize=(self.width, self.height))
        if title:
            plt.figtext(0.95, 0.8, title, va='top', fontsize=14)

        for h in range(self.height):
            plt.axhline(h, color='k', lw=1)
        for w in range(self.width):
            plt.axvline(w, color='k', lw=1)

        race_icon = plt.imread('img/race.png')
        plt.imshow(race_icon,
                   extent=[self.target_position[0] + 0.15,
                           self.target_position[0] + 0.85,
                           self.target_position[1] + 0.25,
                           self.target_position[1] + 0.75,],
                   aspect='equal')

        flame_icon = plt.imread('img/flame.png')
        for fp in self.fire_positions:
            plt.imshow(flame_icon,
                       extent=[fp[0] + 0.25,
                               fp[0] + 0.75,
                               fp[1] + 0.25,
                               fp[1] + 0.75,],
                       aspect='equal')
        
        if add_player_position:
            plt.plot(self.player_position[0] + 0.5, self.player_position[1] + 0.5,
                     marker='x', mew=4, ms=15, color='tab:red')

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.xticks([])
        plt.yticks([])
        return plt.gca()


class QTable(object):
    """ Implement Q-function as Q-table. """
    def __init__(self, env, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.action_list = list(env.action_mapper.keys())
        
        self.q = {}
        for w in range(env.width):
            for h in range(env.height):
                state = (w, h)
                self.q[state] = {'up': [0.], 'down': [0.],
                                 'left': [0.], 'right': [0.]}
                
        self.height = env.height
        self.width = env.width

        self._init_history()

    def _init_history(self):
        self.history = {}
        for k, v in self.q.items():
            self.history[k] = {}
            for kk, vv in v.items():
                self.history[k][kk] = []

    def _update_history(self):
        for k, v in self.q.items():
            for kk, vv in v.items():
                self.history[k][kk].append(vv[0])

    def save_q_table(self, filename):
        """ Save q table to json file. Note that history is lost
        in that process. """

        # Convert keys from tuple to strings, otherwise json does
        # not like it.
        q_save = {}
        for k, v in self.q.items():
            d_tmp = {}
            for kk, vv in self.q[k].items():
                d_tmp[kk] = vv[0]                
            q_save[str([int(i) for i in k])] = d_tmp

        with open(filename, 'w') as fid:
            json.dump(q_save, fid, indent=4)

    def load_q_table(self, filename):
        """ Load q table from json file. """
        self._init_history()
        with open(filename, "r") as fid:
            q_load = json.load(fid)

        # Turn key back into tuple...
        q_ = {}
        for k, v in q_load.items():
            w = int(k.split(',')[0].split('[')[-1])
            h = int(k.split(',')[-1].split(']')[0])
            q_[(w, h)] = {}
            for kk, vv in q_load[k].items():
                q_[(w, h)][kk] = [vv]
        self.q = q_

    def update(self, samples):
        """ Perform the TD rule update to the Q-table using a batch of
        samples. """
        self._update_history()

        states = samples[:][0]
        actions = samples[:][1]
        rewards = samples[:][2]
        next_states = samples[:][3]
        dones = samples[:][4]
        
        n_samples = len(states)
        for i in range(n_samples):
            state = tuple(states[i])
            next_state = tuple(next_states[i])

            target_q = rewards[i]
            if not dones[i]:
                max_a_q = np.max(list(self.q[next_state].values()))
                target_q = rewards[i] + self.gamma * max_a_q
            action = self.action_list[int(actions[i])]
            self.q[state][action] += (self.alpha * (target_q - self.q[state][action]))
        
    def get_greedy_action(self, state):
        """ Given the state, return the action that maximizes the Q-value given
        the current Q-table values. """
        q_a_dict = self.q[tuple(state)]
        return max(q_a_dict, key=q_a_dict.get)
    
    def get_v_table(self):
        """ Return the current state-value table. """
        v_table = {}
        for key, val in self.q.items():
            v_table[key] = np.max(list(val.values()))
        return v_table
    
    def get_q_table(self):
        """ Return the current Q-table. """
        q_table = {}
        for w in range(self.width):
            for h in range(self.height):
                q_table[(w, h)] = {}
                q_vals = self.q[(w, h)]
                for action in self.action_list:
                    q_table[(w, h)][action] = q_vals[action][0]
        return q_table
    
    def get_greedy_policy(self):
        """ Given the current Q-table values, pick the best actions in each state
        (greedily) and return that policy. """
        policy = {}
        for key, val in self.q.items():
            policy[key] = self.get_greedy_action(key)
        return policy


class QNet(object):
    """ Implement Q-function as Q-network. """
    def __init__(self, env, alpha=1e-3, gamma=0.99):
        self.action_list = list(env.action_mapper.keys())
        self.num_actions = len(self.action_list)
        self.dim_state = 2
        
        self.alpha = alpha
        self.gamma = gamma
        
        self.q_model = self.create_q_model()
        
        self.height = env.height
        self.width = env.width

    def create_q_model(self):
        """ Initialize the q-net with a fixed size and architecture. """
        model = Sequential() 
        model.add(Dense(32, activation="relu", input_dim=self.dim_state))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.alpha))
        return model
    
    def update(self, samples):
        """ Update the q-net weights according to the TD update rule,
        using a batch of experiences. """
        states = samples[:][0]
        actions = samples[:][1]
        rewards = samples[:][2]
        next_states = samples[:][3]
        dones = samples[:][4]

        # Calculate target Q according to TD(0) rule
        target_q = (rewards.flatten() + (1-dones) * self.gamma * np.amax(
            self.q_model.predict(next_states, verbose=0), axis=1))        
        target_f = self.q_model.predict(states, verbose=0)
        actions = np.int_(actions).flatten()
        for i in range(target_f.shape[0]):
            target_f[i, actions[i]] = target_q[i]

        # Update network weights
        self.q_model.fit(states, target_f, epochs=1, verbose=0)

    def get_greedy_action(self, state):
        """ Given the state, return the action that maximizes the Q-value given
        the current Q-net. """
        state = np.reshape(state, [1, self.dim_state])
        action_idx = np.argmax(self.q_model(state))
        return self.action_list[action_idx]
    
    def get_v_table(self):
        """ Create the v-table for each state  using the trained NN
        and return it. """
        v_table = {}
        for w in range(self.width):
            for h in range(self.height):
                v_table[(w, h)] = np.max(self.q_model.predict([[w, h]], verbose=0))
        return v_table
    
    def get_q_table(self):
        """ Create the Q-table for each state-action pair using the
        trained NN and return it. """
        q_table = {}
        for w in range(self.width):
            for h in range(self.height):
                q_table[(w, h)] = {}
                q_vals = self.q_model.predict([[w, h]], verbose=0)
                for i, action in enumerate(self.action_list):
                    q_table[(w, h)][action] = q_vals[0][i]
        return q_table
    
    def get_greedy_policy(self):
        """ Given the current q-net, find the greedy action and return the
        corresponding greedy policy. """
        policy = {}
        for w in range(self.width):
            for h in range(self.height):
                policy[(w, h)] = self.get_greedy_action((w, h))
        return policy

    def save_model(self, filename):
        """ Save the weights of the q-net model. """
        self.q_model.save_weights(filename)

    def load_model(self, filename):
        """ Load the weights of a saved q-net model. """
        self.q_model.load_weights(filename)


class QLearner:
    """ Main class to train Q-table and Q-net for the maze environment. """
    def __init__(self, env, alpha=1e-3, gamma=0.99, batch_size=128,
                 q_function='table', epsilon_init=0.99, epsilon_final=0.05):

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final

        # For experience replay
        dim_action = 1
        dim_state = 2
        self.buffer = ReplayBuffer(10000, dim_state, dim_action)

        if q_function == 'table':
            self.q_func = QTable(env, alpha, gamma)
        elif q_function == 'net':
            self.q_func = QNet(env, alpha, gamma)

    def train(self, n_episodes):
        """ Run the training loop for the given number of episodes. """
        epsilon = self.epsilon_init
        for i in trange(n_episodes):
            epsilon -= (self.epsilon_final - self.epsilon_init) / n_episodes
            state = self.env.reset()
            done = False
            while not done:
                if np.random.random(size=1) < epsilon:
                    # Pick action randomly
                    action = np.random.choice(
                        list(self.env.action_mapper.keys()))
                else:
                    # Pick action greedily
                    action = self.q_func.get_greedy_action(state)

                state, action, reward, next_state, done = (
                    self.env.step(action))
                
                action_idx = self.q_func.action_list.index(action)
                self.buffer.push(state, action_idx, reward, next_state, done)
            
            minibatch = self.buffer.sample(self.batch_size)
            self.q_func.update(minibatch)

    def plot_training_evolution(self):
        """ Show the evolution of Q-values over the training period. Note:
        only works for the q_function='table' for the moment. """
        if isinstance(self.q_func, QTable):
            plt.figure(figsize=(6, 4))
            plt.suptitle('Evolution of Q-values during training',
                         fontsize=16)

            cmap = {
                (0, 0): plt.get_cmap('Blues'),
                (0, 1): plt.get_cmap('Reds'),
                (1, 0): plt.get_cmap('Greens'),
            }

            for k, v in self.q_func.history.items():
                if k == (self.env.width-1, self.env.height-1):
                    continue

                for i, (kk, vv) in enumerate(v.items()):
                    vv = np.array(vv)
                    if self.env.width < 3 and self.env.height < 3:
                        c = cmap[k]((i+1)/5.)
                        plt.plot(vv, label=f"Q({k}, '{kk}')", c=c)
                    else:
                        plt.plot(vv, label=f"Q({k}, '{kk}')")

            plt.legend(ncol=3, fontsize=12, bbox_to_anchor=(1.02, 1))
            plt.xlabel('Episode')
            plt.ylabel('Q-value')
            plt.show()
        else:
            raise NotImplementedError(
                'Can only plot training evolution for QTable Q-functions.')


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

