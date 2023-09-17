import numpy as np

class DifferentialSystem():
    
    def __init__(self, number_player_one_actions, number_player_two_actions, transition_matrix):
        self.number_player_one_actions = number_player_one_actions
        self.number_player_two_actions = number_player_two_actions
        self.transition_matrix = transition_matrix
        self.state = [0] * (number_player_one_actions + number_player_two_actions)
        self.step_size = 0
        
    def step(self):
        simple_gradient = np.matmul(self.transition_matrix, self.state)
        player_one_payoff = np.dot(self.state[:self.number_player_one_actions], simple_gradient[:self.number_player_one_actions])
        player_two_payoff = np.dot(self.state[self.number_player_one_actions:], simple_gradient[self.number_player_one_actions:])
#         Need to subtract uniformly to make things add to one.
        player_one_directional_correction = np.divide(np.sum(simple_gradient[:self.number_player_one_actions]), self.number_player_one_actions)
        player_two_directional_correction = np.divide(np.sum(simple_gradient[self.number_player_one_actions:]), self.number_player_two_actions)
        combined_gradient = np.zeros_like(simple_gradient)
        combined_gradient[:self.number_player_one_actions] = simple_gradient[:self.number_player_one_actions] - player_one_directional_correction
        combined_gradient[self.number_player_one_actions:] = simple_gradient[self.number_player_one_actions:] - player_two_directional_correction
        new_state = self.state + self.step_size * combined_gradient
        new_state[:self.number_player_one_actions] = self.projection(new_state[:self.number_player_one_actions])
        new_state[self.number_player_one_actions:] = self.projection(new_state[self.number_player_one_actions:])
        self.state = new_state
        return player_one_payoff, player_two_payoff

    def projection(self,state):
        while min(state) < 0:
            minimum = min(state)
            correction = [0 if entry == 0 else np.divide(1 * minimum, np.count_nonzero(state) - 1) for entry in state]
            correction[np.argmin(state)] = -1 * minimum
            state += correction
        return state

    
    def run(self, initial_state, number_episodes, step_size):
        initial_state[:self.number_player_one_actions] = self.projection(initial_state[:self.number_player_one_actions])
        initial_state[self.number_player_one_actions:] = self.projection(initial_state[self.number_player_one_actions:])
        self.state = initial_state
        self.step_size = step_size
        player_one_strategies = np.zeros((number_episodes, self.number_player_one_actions))
        player_two_strategies = np.zeros((number_episodes, self.number_player_two_actions))
        player_one_utilities = np.zeros(number_episodes)
        player_two_utilities = np.zeros(number_episodes)
        for episode in range(number_episodes):
            player_one_strategies[episode] = self.state[:self.number_player_one_actions]
            player_two_strategies[episode] = self.state[self.number_player_one_actions:]
            player_one_utility, player_two_utility = self.step()
            player_one_utilities[episode] = player_one_utility
            player_two_utilities[episode] = player_two_utility
        return player_one_utilities, player_two_utilities, player_one_strategies, player_two_strategies