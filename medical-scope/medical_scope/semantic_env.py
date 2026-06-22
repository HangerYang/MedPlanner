from __future__ import annotations

import random

import numpy as np

from .conversation import SemanticState


class SemanticConversationEnvironment:
    def __init__(self, transition_model, initial_embedding, max_depth, reward_function,
                 initial_embedding_history=()) -> None:
        self.transition_model = transition_model
        self.initial_embedding = tuple(initial_embedding)
        self.initial_embedding_history = tuple(initial_embedding_history)
        self.max_depth = int(max_depth)
        self.reward_function = reward_function
        self.state_to_action_map = {}
        self.state_action_to_response_map = {}

    def get_initial_state(self) -> SemanticState:
        print("getting initial state...")
        return SemanticState(
            tuple(self.initial_embedding),
            depth=1,
            embedding_history=self.initial_embedding_history,
        )

    def get_actions(self, state: SemanticState):
        historical_context = tuple(state.conversation)
        if historical_context in self.state_to_action_map:
            print("state already in state_to_action_map dict, use the actions!")
            return self.state_to_action_map[historical_context]
        actions = self.transition_model.sample_actions(historical_context)
        self.state_to_action_map[historical_context] = actions
        return actions

    def is_terminal(self, state: SemanticState) -> bool:
        return state.depth >= self.max_depth

    def get_reward(self, prev_state, action, new_state):
        return self.reward_function.get_reward(prev_state, action, new_state)

    def _extend_history(self, state: SemanticState, action, result_state_tuple: tuple) -> tuple:
        expert_emb = tuple((np.array(state.conversation) + np.array(action)).astype(np.float32).tolist())
        return state.embedding_history + (expert_emb, result_state_tuple)

    def execute_in_selection(self, state: SemanticState, action):
        historical_context = tuple(state.conversation)
        possible_responses = self.state_action_to_response_map[(historical_context, tuple(action))]
        result_tuple = random.choice(list(possible_responses))
        new_history = self._extend_history(state, action, result_tuple)
        selected_state = SemanticState(tuple(result_tuple), depth=state.depth + 2, embedding_history=new_history)
        return selected_state, self.get_reward(state, action, selected_state)

    def execute_in_expansion(self, state: SemanticState, action):
        historical_context = tuple(state.conversation)
        possible_responses = self.transition_model.transit(historical_context, action)
        self.state_action_to_response_map[(historical_context, tuple(action))] = possible_responses
        result_tuple = random.choice(list(possible_responses))
        new_history = self._extend_history(state, action, result_tuple)
        expanded_state = SemanticState(tuple(result_tuple), depth=state.depth + 2, embedding_history=new_history)
        return expanded_state, self.get_reward(state, action, expanded_state)

    def get_discount_factor(self) -> float:
        return 1.0
