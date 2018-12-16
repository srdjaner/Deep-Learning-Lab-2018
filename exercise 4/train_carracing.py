# export DISPLAY=:0

import sys
sys.path.append("../")

import numpy as np
import gym
from dqn_car.dqn_agent import DQNAgent
from dqn_car.networks import NeuralNetwork, TargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats

def run_episode(env, agent, deterministic, history_length, skip_frames=2,  do_training=True, rendering=True, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    total_reward = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)
        action = action_id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r
            # reward = reward if not terminal else -10

            if rendering:
                env.render()

            if terminal:
                 break
#=============================IF NOT WORKING TRY REMOVING THIS==============================
        #early_done, punishment = agent.check_early_stop(reward, total_reward)
        #if early_done:
        #    reward += punishment
       # 
        #terminal = terminal or early_done
        #total_reward += reward
#============================TILL HERE======================================================
        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)
        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)
        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    return stats

def action_id_to_action(action_id):
    if (action_id==0): a = np.array([0.0, 0.0, 0.0]).astype('float32')
    if (action_id==1): a = np.array([-1.0, 0.0, 0.0]).astype('float32')
    if (action_id==2): a = np.array([1.0, 0.0, 0.0]).astype('float32')
    if (action_id==3): a = np.array([0.0, 0.8, 0.0]).astype('float32')
    if (action_id==4): a = np.array([0.0, 0.0, 0.2]).astype('float32')
    return a

def train_online(env, agent, num_episodes, history_length, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        timesteps = int(np.max([300, i]))
        stats = run_episode(env, agent, history_length=history_length, max_timesteps=timesteps, deterministic=False, do_training=True)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(0),
                                                      "left" : stats.get_action_usage(1),
                                                      "right" : stats.get_action_usage(2),
                                                      "accel" : stats.get_action_usage(3),
                                                      "brake" : stats.get_action_usage(4)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...

        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
        

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped

    state_dim = env.observation_space.shape[0]
    num_actions = 5
    history_length = 3
    

    Q = NeuralNetwork(state_dim, num_actions, history_length=history_length)
    Q_target = TargetNetwork(state_dim, num_actions, history_length=history_length)
    agent = DQNAgent(Q, Q_target, num_actions)

    train_online(env, agent, num_episodes=1000, history_length=history_length, model_dir="./models_carracing")
