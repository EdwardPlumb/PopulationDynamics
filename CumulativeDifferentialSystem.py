import numpy as np

class CumulativeDifferentialSystem():
    
    def __init__(self, number_player_one_actions, number_player_two_actions, transition_matrix):
        self.number_player_one_actions = number_player_one_actions
        self.number_player_two_actions = number_player_two_actions
        self.transition_matrix = transition_matrix
        self.state = [0] * (number_player_one_actions + number_player_two_actions)
        self.step_size = 0
        
    def step(self):
        strategy = [entry for entry in self.state]
        # Projection is taken in two steps:
        # First subtract the average uniformly.
        player_one_average = np.divide(np.sum(self.state[:self.number_player_one_actions]) - 1.0, self.number_player_one_actions)
        player_two_average = np.divide(np.sum(self.state[self.number_player_one_actions:]) - 1.0, self.number_player_two_actions)
        strategy[:self.number_player_one_actions] -= player_one_average
        strategy[self.number_player_one_actions:] -= player_two_average
        # Second project to make sure everything is positive.
        strategy[:self.number_player_one_actions] = self.projection(strategy[:self.number_player_one_actions])
        strategy[self.number_player_one_actions:] = self.projection(strategy[self.number_player_one_actions:])
        # Update the cumulative payoffs
        payoffs_for_actions = np.matmul(self.transition_matrix, strategy)
        self.state += self.step_size * payoffs_for_actions
        return strategy

    def projection(self,strategy):
        while min(strategy) < 0:
            minimum = min(strategy)
            correction = [0 if index == 0 else np.divide(1 * minimum, np.count_nonzero(strategy) - 1) for index in strategy]
            correction[np.argmin(strategy)] = -1 * minimum
            strategy = np.add(strategy, correction)
        return strategy

    
    def run(self, initial_state, number_episodes, step_size):
        self.state = initial_state
        self.step_size = step_size
        player_one_states = np.zeros((number_episodes, self.number_player_one_actions))
        player_two_states = np.zeros((number_episodes, self.number_player_two_actions))
        player_one_strategies = np.zeros((number_episodes, self.number_player_one_actions))
        player_two_strategies = np.zeros((number_episodes, self.number_player_two_actions))
        for episode in range(number_episodes):
            player_one_states[episode] = self.state[:self.number_player_one_actions]
            player_two_states[episode] = self.state[self.number_player_one_actions:]
            strategies = self.step()
            player_one_strategies[episode] = strategies[:self.number_player_one_actions]
            player_two_strategies[episode] = strategies[self.number_player_one_actions:]
        return player_one_states, player_two_states, player_one_strategies, player_two_strategies