from abc import ABC, abstractmethod
from typing import Tuple
"""
Inherit this class for muzero to play
"""
class Game(ABC):

	@abstractmethod
	def __init__(self, seed=None):
		pass

	"""
    Apply action to the game.
    
    Args:
        action : action of the action_space to take.
    Returns:
        The new observation, the reward and a boolean if the game has ended.
    """
	@abstractmethod
	def step(self, action): 
		pass

	"""
    Return the current player.
    Returns:
        The current player, it should be an element of the players list in the config. 
    """
	@abstractmethod
	def to_play(self):
		pass

	"""
    Should return the legal actions at each turn, if it is not available, it can return
    the whole action space. At each turn, the game have to be able to handle one of returned actions.
    
    For complexe game where calculating legal moves is too long, the idea is to define the legal actions
    equal to the action space but to return a negative reward if the action is illegal.        
    Returns:
        An array of integers, subset of the action space.
    """
	@abstractmethod
	def legal_actions(self):
		pass

	"""
    Reset the game for a new game.
    
    Returns:
        Initial observation of the game.
    """
	@abstractmethod
	def reset(self):
		pass

	"""
    Properly close the game.
    """
	@abstractmethod
	def close(self):
		pass

	"""
    Display the game observation.
    """
	@abstractmethod
	def render(self):
		pass

	"""
    Asks the player to input an action
    Checks if the action is legal
    If not, ask again

    Returns:
		An integer represents the action
    """
	@abstractmethod
	def input_action(self):
		pass

	"""
    Translates the integer action to its actual meaning

    Returns:
    	Meaning of the input action
    """
	@abstractmethod
	def output_action(self, action):
		pass





