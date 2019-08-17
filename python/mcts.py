from __future__ import division

import time
import math
import random
import numpy as np
import logging


def random_policy(state, tree_helper):
    while not state.is_terminal():
        try:
            action = random.choice(tree_helper.get_possible_actions(state))
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
    return state.get_reward()


class TreeNode:
    def __init__(self, state, parent, prob=1):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0
        self.children = {}
        self.prob = prob

    @property
    def avg_reward(self):
        return self.total_reward/float(self.num_visits)

    @property
    def max_child_reward(self):
        max_child_reward = -1
        for name, child in self.children.items():
            if child.avg_reward > max_child_reward:
                max_child_reward = child.avg_reward
        return max_child_reward

    @property
    def min_child_reward(self):
        min_child_reward = 100000000
        for name, child in self.children.items():
            if child.avg_reward < min_child_reward:
                min_child_reward = child.avg_reward
        return min_child_reward

    @property
    def standardized_reward(self):
        return (self.avg_reward-self.parent.min_child_reward)/(self.parent.max_child_reward-self.parent.min_child_reward)

    @property
    def is_available(self):
        return np.random.uniform() < self.prob

    def get_ucbt(self, exploration_value):
        return self.avg_reward + (self.avg_reward * exploration_value * math.sqrt(2 * math.log(self.parent.num_visits) / self.num_visits))

    def __str__(self):
        #avg_reward = 0 if self.num_visits == 0 else self.total_reward/float(self.num_visits)
        to_return = "State: {0}\n\tVisits: {1}\n\tAvgReward: {2}\n\tStdized Reward: {3}\n\tQ+U: {4}".format(self.state,
                                                                                                               self.num_visits,
                                                                                                               self.avg_reward,
                                                                                                               self.standardized_reward,
                                                                                                               self.get_ucbt(1))
        return to_return


class MCTS:
    def __init__(self, root_state, tree_helper, time_limit=None, iteration_limit=None,
                 exploration_constant=1, num_rollouts=1):
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.search_limit = iteration_limit
            self.limit_type = 'iterations'

        self.root = TreeNode(root_state, None)
        self.tree_helper = tree_helper
        self.exploration_constant = exploration_constant
        self.num_rollouts = num_rollouts

    def search(self):
        # Do MCTS search until time or iteration limit reached
        if self.limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000
            i = 0
            while time.time() < time_limit:
                if i % 200 == 0:
                    logging.info("Done {0} rounds...".format(i))
                    logging.info(self.get_status_string())
                self.execute_round()
                i += 1
        else:
            for i in range(self.search_limit):
                if i % 200 == 0:
                    logging.info("Done {0} rounds...".format(i))
                    logging.info(self.get_status_string())
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        return self.get_action(self.root, best_child)

    def execute_round(self):
        node = self.select_node(self.root)
        for i in range(self.num_rollouts):
            reward = self.tree_helper.rollout(node.state)
            self.backpropogate(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node, self.exploration_constant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = self.tree_helper.get_possible_actions(node.state)
        for action in actions:
            if action.name not in node.children:
                new_node = TreeNode(node.state.take_action(action), parent=node, prob=action.prob)
                node.children[action.name] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_child(self, node, exploration_value):
        # Get best child after randomly removing some according to their availability probability

        # Determine which children are available
        available_children = [child for child in node.children.values() if child.is_available]
        while not available_children:
            available_children = [child for child in node.children.values() if child.is_available]

        best_value = float("-inf")
        best_nodes = []
        for child in available_children:
            node_value = child.get_ucbt(exploration_value)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)

        # Return random choice if >1 value tied for best
        return np.random.choice(best_nodes)

    def get_action(self, root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action

    def get_status_string(self):
        log_str = "Root node:\n{0}\nChildren:\n".format(self.root.state)
        for child in self.root.children:
            log_str += "{0}\n".format(self.root.children[child])
        return log_str
