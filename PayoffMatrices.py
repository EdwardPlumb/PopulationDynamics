# Class for the Payoff Matrices
import numpy as np

class PayoffMatrices():
    
    def __init__(self,player_one_payoffs, player_two_payoffs):
        self.player_one_payoffs = player_one_payoffs
        self.player_two_payoffs = player_two_payoffs
        self.number_player_one_actions = len(player_one_payoffs)
        self.number_player_two_actions = len(player_one_payoffs[0])
        
        
    def transition_matrix(self):
        transitions = np.zeros((self.number_player_one_actions + self.number_player_two_actions, self.number_player_one_actions + self.number_player_two_actions))
        for index, payoff_row in enumerate(self.player_one_payoffs):
            transitions[index][self.number_player_one_actions:] = payoff_row
        player_two_payoffs = np.transpose(self.player_two_payoffs)
        for index, payoff_row in enumerate(player_two_payoffs):
            transitions[self.number_player_one_actions + index][:self.number_player_one_actions] = payoff_row
        return transitions