from __future__ import annotations

import math
import random
import time
from collections import defaultdict


class UpperConfidenceBounds:
    def __init__(self) -> None:
        self.total = 0
        self.times_selected = {}

    def select(self, state, actions, qfunction):
        for action in actions:
            if action not in self.times_selected:
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        for action in actions:
            value = 0.05 * qfunction.get_q_value(state, action) + math.sqrt(
                (2 * math.log(max(self.total, 1))) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions.append(action)
        result = random.choice(max_actions)
        self.times_selected[result] += 1
        self.total += 1
        return result


class SingleAgentNode:
    visits = defaultdict(int)

    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.qfunction = qfunction
        self.bandit = bandit
        self.reward = reward
        self.action = action
        self.children = {}

    def is_fully_expanded(self):
        valid_actions = set(self.mdp.get_actions(self.state))
        valid_children = set(self.children)
        print(f"valid actions: {len(valid_actions)} \tnumber of children: {len(valid_children)}")
        return len(valid_actions) == len(valid_children)

    def select(self):
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
            return self
        actions = list(self.children.keys())
        action = self.bandit.select(self.state, actions, self.qfunction)
        return self.get_outcome_child_select(action).select()

    def expand(self):
        if self.mdp.is_terminal(self.state):
            return self, None
        next_actions = self.mdp.get_actions(self.state)
        valid_children = set(self.children.keys())
        valid_next_actions = set(next_actions)
        print(f"expanding...\tnumber of children: {len(valid_children)}\tnumber of actions: {len(valid_next_actions)}")
        actions = list(valid_next_actions - valid_children)
        if not actions:
            return self, next_actions
        action = random.choice(actions)
        self.children[action] = []
        return self.get_outcome_child_expand(action), next_actions

    def get_outcome_child_select(self, action):
        next_state, reward = self.mdp.execute_in_selection(self.state, action)
        for child in self.children[action]:
            if next_state.response == child.state.response:
                return child
        child = SingleAgentNode(self.mdp, self, next_state, self.qfunction, self.bandit, reward, action)
        self.children[action].append(child)
        return child

    def get_outcome_child_expand(self, action):
        next_state, reward = self.mdp.execute_in_expansion(self.state, action)
        for child in self.children[action]:
            if next_state.response == child.state.response:
                print("child is found")
                return child
        child = SingleAgentNode(self.mdp, self, next_state, self.qfunction, self.bandit, reward, action)
        self.children[action].append(child)
        return child

    def back_propagate(self, reward, child):
        action = child.action
        SingleAgentNode.visits[self.state] += 1
        SingleAgentNode.visits[(self.state, action)] += 1
        self.qfunction.update(self.state, action, 0.0, 1 / SingleAgentNode.visits[(self.state, action)], reward)
        if self.parent is not None:
            self.parent.back_propagate(self.reward + reward, self)

    def back_propagate_simple(self, reward):
        print("doing simple back propagation because we cannot expand a tree anymore.")
        action = (0.0,) * len(self.state.conversation)
        SingleAgentNode.visits[self.state] += 1
        SingleAgentNode.visits[(self.state, action)] += 1
        self.qfunction.update(self.state, action, 0.0, 1 / SingleAgentNode.visits[(self.state, action)], reward)
        if self.parent is not None:
            self.parent.back_propagate(self.reward + reward, self)


class SingleAgentMCTS:
    def __init__(self, mdp, qfunction, bandit=None) -> None:
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit or UpperConfidenceBounds()
        self.initial_actions = None

    def create_root_node(self):
        return SingleAgentNode(self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit)

    def simulate(self, node, seed=None):
        if seed is not None:
            random.seed(seed)
        state = node.state
        cumulative_reward = 0.0
        while not self.mdp.is_terminal(state):
            actions = self.mdp.get_actions(state)
            if not actions:
                break
            action = random.choice(list(actions))
            state, reward = self.mdp.execute_in_expansion(state, action)
            cumulative_reward += reward
        return cumulative_reward

    def mcts(self, timeout=100, root_node=None, seed=None):
        if root_node is None:
            root_node = self.create_root_node()
        start_time = time.time()
        current_time = time.time()
        rollout_count = 0
        do_nothing_count = 0
        print("time out for mcts given as: ", timeout)
        while current_time < start_time + timeout:
            selected_node = root_node.select()
            print(f"selected node depth: {selected_node.state.depth}")
            if not self.mdp.is_terminal(selected_node.state):
                child, action_in_expansion = selected_node.expand()
                if self.initial_actions is None:
                    self.initial_actions = action_in_expansion
                reward = self.simulate(child, seed=seed)
                print("cumulative reward after expansion and simulation: ", reward)
                selected_node.back_propagate(reward, child)
            else:
                do_nothing_count += 1
                print("fully expanded tree. using simple back propagation: ", do_nothing_count)
                selected_node.back_propagate_simple(0.0)
            rollout_count += 1
            print("time taken for one iteration of mcts: ", time.time() - current_time)
            current_time = time.time()
        print("number of rollouts achieved: ", rollout_count)
