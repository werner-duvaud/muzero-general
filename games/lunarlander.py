import datetime
import os

import gym
import numpy
import torch


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game


        ### Game
        self.observation_shape = (1, 1, 8)  # Dimensions of the game observation, must be 3D. For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(4)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observation to add to the current observation

        ### Self-Play
        self.num_actors = 2  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 2000  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of futur moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.self_play_delay = 0  # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid over/underfitting

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25


        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.pooling_size = 2
        self.fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.hidden_layers = [16]
        

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 8000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128*5  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.window_size = 2000  # Number of self-play games to keep in the replay buffer
        self.td_steps = 2000  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 0  # Number of seconds to wait after each training to adjust the self play / training ratio to avoid over/underfitting
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1
        self.lr_decay_steps = 1000


        ### Test
        self.test_episodes = 2  # Number of games rendered when calling the MuZero test method


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


class Game:
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("LunarLander-v2")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[observation]]), reward/5, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complexe game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return [i for i in range(4)]

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return numpy.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def input_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        pass

    def output_action(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        descriptions = [
            "Do nothing", 
            "Fire left orientation engine",
            "Fire main engine",
            "Fire right orientation engine",
        ]
        return "{}. {}".format(action_number, descriptions[action_number])
