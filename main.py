# ==========================================
# Title:  Main function
# Author: Ashish Mehta
# Date:   25 October 2018
# ==========================================

from model_learning import Model
from planner import Planner
from time import sleep
import math
import gym
import numpy as np
import matplotlib.pyplot as plt

def perfect_model(input_vec):
    """
    Perfect pendulum model used to calculate the next state
    :param input_vec: [costh, sinth, thdot, u]
    :return: [cos(newth), sin(newth), newthdot]
    """
    # costh, sinth, thdot, u = input_vec[0], input_vec[1], input_vec[2], input_vec[3]
    th = np.arctan2(input_vec[:, 1], input_vec[:,0])
    g = 10.
    m = 1.
    l = 1.
    dt = 0.05

    u = np.clip(input_vec[:, 3], -2, 2)

    newthdot = input_vec[:,2] + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newthdot = np.clip(newthdot, -8, 8)  # pylint: disable=E1111
    return np.array([np.cos(newth), np.sin(newth), newthdot]).T


def collect_data():
    M.random_sampler(200000)
    M.temporal_noise_sampler(200000)


def train_model():
    data = M.read_from_disk('data')
    train_dataset = M.dataset_generator(data[20000:, :], 16)
    val_dataset = M.dataset_generator(data[0:20000, :], 3)
    M.train_model(4, train_dataset, val_dataset)


def plan():

    observation = env.reset()
    env.render()

    # define start and goal states
    start_state = [math.atan2(observation[1], observation[0]), observation[2]]
    goal_state = [0.0, 0.0]

    scale_factor = 1
    steps = []
    scale_axis = []

    scale_factor = 1
    # plan a path using A*
    for i in range(10):
        P = Planner(perfect_model, scale_factor)

        path, n_steps = P.find_path(start_state=start_state, goal_state=goal_state)

        if path is None:
            print("Not able to compute path")
            n_steps = 0

        # action_seq = path[:, 2]
        # action_seq = action_seq[:-10]
        action_seq = path[1:, 2]
        for ind, act in enumerate(action_seq):
            ob, _, _, _ = env.step([act])
            state = [math.atan2(ob[1], ob[0]), ob[2]]
            # print('Expected node ', path[ind+1, :2])
            # print('Visited node ', P.state_to_node(state))
            # print('\n')
            env.render()

        scale_axis.append(scale_factor)
        steps.append(n_steps)
        print(i, scale_factor, n_steps)
        scale_factor = scale_factor*0.95

    plt.plot(scale_axis, steps)
    plt.xlabel("% discrete action-space")
    plt.ylabel("No. steps")
    plt.title("Discretization of action space vs no. of steps")
    plt.show()


    # Loop to replan from current state
    # while True:
    #     P = Planner(M.predict_using_model)
    #     start_state = state
    #     path = P.find_path(start_state=start_state, goal_state=goal_state)
    #     action_seq = path[:, 2]
    #     action_seq = action_seq[:-10]
    #     for ind, act in enumerate(action_seq):
    #         ob, _, _, _ = env.step([act])
    #         state = [math.atan2(ob[1], ob[0]), ob[2]]
    #         # print('Expected node ', path_seq[ind])
    #         # print('Visited node ', P.state_to_node(state))
    #         print('\n')
    #         env.render()
    #         sleep(0.2)


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    M = Model(False, "./tmp/model120000.ckpt")

    # Uncomment to collect data
    # collect_data()

    # Uncomment to train model
    train_model()

    # Uncomment to plan
    # plan()


    # state = env.reset()
    # while True:
    #     action = env.action_space.sample()
    #     env_pred, _, _, _ = env.step(state)
    #
    #     model_pred = perfect_model(np.append(state, action).reshape(1, 4))
    #
    #     print(env_pred-model_pred)
    #
    #     state = env_pred


