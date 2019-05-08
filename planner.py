# ==========================================
# Title:  A star planning Code
# Author: Ashish Mehta
# Date:   24 October 2018
# ==========================================

import numpy as np
from bisect import insort_left


class Planner(object):
    """
    Provided a model, this classes uses an A* based planner to plan trajectories to the goal.
    """

    def __init__(self, model_prediction, scale):
        """
        :param motion_model: Learned model of the environment
        """

        self._goal_state = np.array([0, 0], dtype=np.float32)
        self._goal_node = np.array([0, 0], dtype=np.int16)

        self._action_bins = int(scale*100)
        self._action_grid = np.linspace(start=-2, stop=2, num=self._action_bins, dtype=np.float32)

        self._theta_bins = int(720)
        self._theta_grid = np.linspace(start=-np.pi, stop=np.pi, num=self._theta_bins, dtype=np.float32)
        self._theta_grid_2d = np.repeat(self._theta_grid.reshape(1, -1), self._action_bins, axis=0)

        self._theta_dot_bins = int(100)
        self._theta_dot_grid = np.linspace(start=-8, stop=8, num=self._theta_dot_bins, dtype=np.float32)
        self._theta_dot_grid_2d = np.repeat(self._theta_dot_grid.reshape(1, -1), self._action_bins, axis=0)

        self._visited = np.zeros(shape=(self._theta_bins, self._theta_dot_bins), dtype=np.uint8)
        self._parent = np.full(shape=(self._theta_bins, self._theta_dot_bins, 3), fill_value=-1024, dtype=np.float32)

        self._model_prediction = model_prediction

    def state_to_node(self, state):
        """
        Compute the node in the State Space Graph.

        :param state: [theta, theta_dot] A 1D array containing a State.
        :return: [theta_node, theta_dot_node] A 1D array containing a Node.
        """
        node = np.array([np.argmin(np.abs(self._theta_grid - state[0])),
                         np.argmin(np.abs(self._theta_dot_grid - state[1]))], dtype=np.int16)

        return node

    def v_states_to_nodes(self, states):
        """
        Computing an array of states to nodes
        :param states:  A 2D array where every row contains a State. Shape = [n, 2]
        :return: A 2D array where every row contains a Node. Shape = [n, 2]
        """

        nodes = np.empty(shape=(states.shape[0], 2), dtype=np.int16)
        nodes[:, 0] = np.argmin(np.abs(self._theta_grid_2d[:states.shape[0], :] - states[:, 0:1]), axis=1)
        nodes[:, 1] = np.argmin(np.abs(self._theta_dot_grid_2d[:states.shape[0], :] - states[:, 1:2]), axis=1)
        return nodes

    def compute_unique_reachable_nodes(self, present_state_action):
        """
        Computes the unique reachable nodes given multiple present states and actions.
        :param present_state_action: A 2D array where every row contains a State and Action. Shape = [n, 3]
        :return: Unique Reachable States, Unique Reachable Nodes, Unique Indices
        """

        # Convert [theta, theta_dot, action] to [cos(theta), sin(theta) theta_dot, action] to pass to model
        state = np.empty(shape=(present_state_action.shape[0], 4), dtype=np.float32)
        state[:, 0] = np.cos(present_state_action[:, 0])
        state[:, 1] = np.sin(present_state_action[:, 0])
        state[:, 2] = present_state_action[:, 1]
        state[:, 3] = present_state_action[:, 2]
        reach_s = (self._model_prediction(state))

        # Covert [cos(theta), sin(theta) theta_dot, action] back to [theta, theta_dot, action]
        reachable_states = np.empty(shape=(reach_s.shape[0], 2), dtype=np.float32)
        reachable_states[:, 0] = np.arctan2(reach_s[:, 1], reach_s[:, 0])
        reachable_states[:, 1] = reach_s[:, 2]

        reachable_nodes = self.v_states_to_nodes(states=reachable_states)

        # Compute the unique reachable nodes and the respective index in the former node list
        unique_reachable_nodes, index_urn = np.unique(reachable_nodes, return_index=True, axis=0)

        # For each visited reachable unique nodes check if it has not been visited previously
        unvisited_reachable_nodes_bool = self._visited[unique_reachable_nodes[:, 0], unique_reachable_nodes[:, 1]] == 0
        index_urn = index_urn[unvisited_reachable_nodes_bool]
        unique_reachable_unvisited_states = reachable_states[index_urn]
        unique_reachable_unvisited_nodes = reachable_nodes[index_urn]

        # Mark all the visited nodes as 1 in _visited. The unvisited 1s are now being visited.
        self._visited[reachable_nodes[:, 0], reachable_nodes[:, 1]] = 1

        return unique_reachable_unvisited_states, unique_reachable_unvisited_nodes, index_urn

    def explore_config(self, present_config, present_node):
        """
        Method representing single search step of the A-star Search Algorithm.

        :param present_config: Tuple containing (f_cost, g_cost, theta, theta_dot).
        :param present_node: A 1D array containing the present node information. present_node = [theta_node, theta_dot_node]
        :return: A list of reachable configurations. [..., (new_f_cost, new_g_cost, new_theta, new_theta_dot), ...]
        """

        self._visited[present_node[0], present_node[1]] = 1

        # For all possible actions in action_bins
        present_state_action = np.empty(shape=(self._action_bins, 3), dtype=np.float32)
        present_state_action[:, 0:2] = present_config[2:4]
        present_state_action[:, 2] = self._action_grid

        reachable_states, reachable_nodes, index = \
            self.compute_unique_reachable_nodes(present_state_action=present_state_action)

        self._parent[reachable_nodes[:, 0], reachable_nodes[:, 1], 0:2] = present_node
        self._parent[reachable_nodes[:, 0], reachable_nodes[:, 1], 2] = self._action_grid[index]

        # Reachable configs contains [f_cost, g_cost, theta, theta_dot, action]
        reachable_configs = np.zeros(shape=(reachable_states.shape[0], 5), dtype=np.float32)
        reachable_configs[:, 2:4] = reachable_states
        reachable_configs[:, 4] = self._action_grid[index]

        # Update the g_cost of reachable states
        reachable_configs[:, 1] = present_config[1] #+ np.square(reachable_states[:, 0] - present_config[2])

        # Update the f_cost of reachable states
        reachable_configs[:, 0] = \
            self.compute_heuristic(state_action=reachable_configs[:, 2:5]) + reachable_configs[:, 1]

        # return a list of tuple of configs which are reachable
        reachable_configs = list(map(tuple, reachable_configs[:, :4]))

        return reachable_configs

    @staticmethod
    def compute_heuristic(state_action):
        """
        Heuristic Calculation.

        :param state_action: A 2D array where every row contains states and actions.
            state_action = n rows of [theta, theta_dot, action]. Shape = [n, 3]
        :return: A 1D array computing the heuristic of every state and action. Shape = [n,1]
        """
        state_square = np.square(state_action)
        state_square[:, 1] = 0.1 * state_square[:, 1]
        state_square[:, 2] = 0.001 * state_square[:, 2]
        heuristic = state_square.sum(axis=1)

        return heuristic

    def find_path(self, start_state, goal_state):
        """
        Method of the A-star search algorithm to find the path given a start and a goal state.

        :param start_state: A 1D array containing the Start State. State = [theta, theta_dot]
        :param goal_state: A 1D array containing the Goal State. Goal = [theta, theta_dot]
        :return: path: 2D array containing, [theta, theta_dot, actions]
        """
        # [theta, theta_dot, action] used to compute heuristic
        start_state_action = np.array([[start_state[0], start_state[1], 0]], dtype=np.float32)

        self._goal_state = goal_state
        self._goal_node = self.state_to_node(state=self._goal_state)

        start_state_g = 0
        [start_state_h] = self.compute_heuristic(state_action=start_state_action)
        start_state_f = start_state_g + start_state_h

        # Tuple (f_cost, g_cost, theta, theta_dot)
        start_config = np.array([start_state_f, start_state_g, start_state[0], start_state[1]], dtype=np.float32)

        start_node = self.state_to_node(state=start_config[2:4])
        self._parent[start_node[0], start_node[1], :] = [-1, -1, 0]

        # Sorted list of open config tuples
        open_config_list = [tuple(start_config)]
        self._visited[start_node[0], start_node[1]] = 1

        # print("Start State: {}".format(start_state))
        # print("Start Node: {}".format(start_node))
        # print("Goal State: {}".format(self._goal_state))
        # print("Goal Node: {}".format(self._goal_node))

        index = 0

        while open_config_list:
            # Pop the the list. since it is sorted we get the least f value config
            present_config = open_config_list.pop(0)
            present_state = np.array(present_config[2:4], dtype=np.float32)
            present_node = self.state_to_node(state=present_state)

            # Check for terminate condition
            if np.all(present_node == self._goal_node):
                # print("\rNumber of nodes searched: %d" % index)
                path = [self._parent[self._goal_node[0], self._goal_node[1]]]
                # Backtrack to find path
                while True:
                    parent_node = path[-1]
                    if np.all(parent_node == [-1, -1, 0]):
                        path.reverse()
                        break

                    path.append(self._parent[int(parent_node[0]), int(parent_node[1])])

                # print("Path found was: ")
                # for element in path:
                    # print("theta_node: %.3f , theta_dot_node: %.3f, action: %.3f" %
                    #       (element[0], element[1], element[2]))
                return np.array(path), index

            # Explore the neighbors of the current co   nfig
            reachable_configs_list = self.explore_config(present_config=present_config, present_node=present_node)
            for reachable_config in reachable_configs_list:
                # Insertion sort into the existing sorted config list
                insort_left(open_config_list, reachable_config)

            # print(present_node)
            index += 1
            # if index % 1000 == 0:
            #     print("\rNumber of nodes searched: %d" % index, end=" ")

        if len(open_config_list) == 0:
            print("\rNumber of nodes searched: %d" % index)
            print("Path was not found")