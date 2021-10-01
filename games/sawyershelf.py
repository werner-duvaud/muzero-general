import datetime
import math
import os

import gym
import numpy
import torch

from .abstract_game import AbstractGame
from games.shelf.sawyer_shelf import SawyerPegShelfEnvMultitask

class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Game
        self.observation_shape = (1, 1,
                                  24)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        numJoints=9
        maxSteps = 40
        self.action_space = numpy.ones(numJoints) #models.py just does len of this variable to determine the action space  #range of action space doesn't matter because gaussian network will
            # create actions without regard to limits  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_actors = 1  # Number of simultaneous threads self-playing to feed the replay buffer
        self.max_moves = maxSteps  # Maximum number of moves if game is not finished before
        self.num_simulations = maxSteps  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        #Progressive widening
        self.progressive_widening_C_pw = 1
        self.progressive_widening_a = 0.49

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 23  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels = 2  # Number of channels before heads of dynamic and prediction networks
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 128
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [256,256,256,256]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [256,256,256,256]  # Define the hidden layers in the reward network
        self.fc_value_layers = [256,256,256,256]  # Define the hidden layers in the value network
        self.fc_policy_layers = [256,256,256,256]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 500000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for sef-playing
        self.value_loss_weight = 0.5  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.training_device = "cuda" if torch.cuda.is_available() else "cpu"  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.window_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)

        # Prioritized Replay (See paper appendix Training)
        self.PER = True  # Select in priority the elements in the replay buffer which are unexpected for the network
        self.use_max_priority = True  # Use the n-step TD error as initial priority. Better for large replay buffer
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 1.0

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1 / 2  # Desired self played games per training step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

        self.log_video = True
        self.video_iter = 500

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1
        elif trained_steps < 0.75 * self.training_steps:
            return 0.1
        else:
            return 0.01


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        sparse = False
        self.env = SawyerPegShelfEnvMultitask(sparse=sparse, stepMinusOne=sparse)
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
        assert (len(action) == 9)
        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        #return [i for i in range(2)]
        numJoints = self.env.action_space.shape[0]
        return [-numpy.ones(numJoints), numpy.ones(numJoints)]

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

    def render(self,mode='human'):
        """
        Display the game observation.
        """
        ret = self.env.render(mode)
        if mode=='human':
            input("Press enter to take a step ")
        return ret

