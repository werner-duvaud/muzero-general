import datetime
import math
import os

import gym
import numpy
import torch


class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game


        ### Game
        self.observation_shape = (3, 11, 11)  # Dimensions of the game observation, must be 3. For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(11 * 11)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(2)]  # List of players. You should only edit the length
        self.stacked_observations = 2  # Number of previous observation to add to the current observation


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
        self.network = "resnet"  # "resnet" / "fullyconnected"
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
        self.results_path = os.path.join(os.path.dirname(__file__), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.training_steps = 10  # Total number of training steps (ie weights update according to a batch)
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


        ### Test
        self.test_episodes = 1  # Number of games rendered when calling the MuZero test method


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
        self.env = Gomoku()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complexe game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

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
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def input_action(self):
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def output_action(self, action):
        return self.env.action_to_human_input(action)


class Gomoku:
    def __init__(self):
        self.board_size = 11
        self.board = numpy.zeros((self.board_size, self.board_size)).astype(int)
        self.player = 1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size)).astype(int)
        self.player = 1
        return self.get_observation()

    def step(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        self.board[x][y] = self.player

        done = self.is_finished()

        reward = 1 if done else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((11, 11), self.player).astype(float)
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                            y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True
        return not has_legal_actions

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y
