from collections import namedtuple, deque
import numpy as np
import os
import gzip
import pickle

class ReplayBuffer:

    # TODO: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)
    # Replay buffer for experience replay. Stores transitions.

    #deque([], maxlen=1000000)

    def __init__(self):

        self.deque_size = 1000000
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=deque([], maxlen=self.deque_size), actions=deque([], maxlen=self.deque_size),
                                next_states=deque([], maxlen=self.deque_size), rewards=deque([], maxlen=self.deque_size),
                                dones=deque([], maxlen=self.deque_size))

    def add_transition(self, state, action, next_state, reward, done): #this is remember
        """
        This method adds a transition to the replay buffer.
        """
        self._data.states.append(state) #_data is memory
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
