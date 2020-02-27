import datetime
import os
import torch
from abc import ABC, abstractmethod

class BaseConfig(ABC):
    @property 
    @abstractmethod
    def game_class_name(self):
        pass

    # Dimensions of the game observation, must be 3. For a 1D array, please reshape it to (1, 1, length of array)
    @property 
    @abstractmethod
    def observation_shape(self):
        pass

    # Fixed list of all possible actions. You should only edit the length
    @property
    @abstractmethod
    def action_space(self):
        pass

    # List of players. You should only edit the length
    @property
    @abstractmethod
    def players(self):
        pass
    
    # Number of previous observation to add to the current observation
    @property
    @abstractmethod
    def stacked_observations(self):
        return self._stacked_observations
    
    # "resnet" / "fullyconnected"
    @property
    @abstractmethod
    def network(self):
        pass
        
    # Number of games rendered when calling the MuZero test method
    @property
    @abstractmethod
    def test_episodes(self):
        pass

    # Path to store the model weights and TensorBoard logs
    @property
    @abstractmethod
    def results_path(self):
        pass

    # Total number of training steps (ie weights update according to a batch)
    @property
    @abstractmethod
    def training_steps(self):
        pass
    

    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game 

        ### Self-Play
        self.num_actors = 2  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 70  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of futur moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.self_play_delay = 0 # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid over/underfitting

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### Network
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.pooling_size = (2, 3)
        self.fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.hidden_layers = [64]


        ### Training
        self.batch_size = 128*3  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.window_size = 1000  # Number of self-play games to keep in the replay buffer
        self.td_steps = 50  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 0 # Number of seconds to wait after each training to adjust the self play / training ratio to avoid over/underfitting
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 10000


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
