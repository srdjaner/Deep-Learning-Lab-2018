import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer
import random

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.995, batch_size=64, epsilon=0.05):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self.Q = Q
        self.Q_target = Q_target

        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.neg_reward_counter = 0
        self.max_neg_rewards = 100

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # 2. sample next batch and perform batch update:
        #self.gas_actions = np.array([a == 3 for a in self.replay_buffer._data.actions])
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)

        td_target = batch_rewards
        td_target[np.logical_not(batch_dones)] += self.discount_factor * np.amax(self.Q_target.predict(self.sess, batch_next_states), 1)[np.logical_not(batch_dones)]
        #print(batch_actions)
        loss = self.Q.update(self.sess, batch_states, batch_actions, td_target)

        self.Q_target.update(self.sess)

        #if self.epsilon > self.epsilon_min:
        #   self.epsilon *= self.epsilon_decay
        #print(self.epsilon)


    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            act_values = self.Q.predict(self.sess, [state])
            action_id = np.argmax(self.Q.predict(self.sess, [state]))
            #print("I PREDICTED")
            #print("action_id_predicted: ", action_id)
            return action_id
        else:
            action_id = np.random.choice([0, 1, 2, 3, 4], p=[0.3,0.1,0.1,0.49, 0.01]) #straight, left, right, accelerate, brake
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
        # print("action_id: ", action_id)
            #print("action_id_random: ", action_id)
            return action_id


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0
