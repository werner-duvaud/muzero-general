import gym
import numpy
import torch


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game


        ### Game
        self.observation_shape = 8  # Dimensions of the game observation
        self.action_space = [i for i in range(4)]  # Fixed list of all possible actions
        self.players = [i for i in range(1)]  # List of players


        ### Self-Play
        self.num_actors = 10  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
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
        self.encoding_size = 64
        self.hidden_size = 128


        ### Training
        self.results_path = "./pretrained"  # Path to store the model weights
        self.training_steps = 3000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.checkpoint_interval = 3  # Number of training steps before using the model for sef-playing
        self.window_size = 1000  # Number of self-play games to keep in the replay buffer
        self.td_steps = 10  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 0  # Number of seconds to wait after each training to adjust the self play / training ratio to avoid over/underfitting
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.00005  # Initial learning rate
        self.lr_decay_rate = 1
        self.lr_decay_steps = 100000


        ### Test
        self.test_episodes = 2  # Number of game played to evaluate the network


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.2 * self.training_steps:
            return float('inf')
        if trained_steps < 0.5 * self.training_steps:
            return 0.8
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
        return numpy.array(observation).flatten(), reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

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
