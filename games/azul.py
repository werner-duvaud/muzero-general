import datetime
import os
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
from gym_azul.model import action_from_action_num, action_num_from_action, \
    Action, Slot, Color, Line

from .abstract_game import AbstractGame
from gym_azul.envs import AzulEnv


class MuZeroConfig:
    def __init__(self):
        # More information is available here:
        # https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        # Seed for numpy, torch and the game
        self.seed = 0
        # None will use every GPUs available
        self.max_num_gpus = None

        ### Game
        # (channel, height, width)
        self.observation_shape = (5, 10, 10)
        self.action_space = list(range(10 * 5 * 5))
        # List of players
        self.players = list(range(2))
        # Number of previous observations and previous actions
        # to add to the current observation
        self.stacked_observations = 8

        # Evaluate
        # Turn Muzero begins to play
        # (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        # Hard coded agent that MuZero faces to assess his progress.
        # It doesn't influence training.
        # None, "random" or "expert" if implemented in the Game class
        self.opponent = "expert"

        ### Self-Play
        # Number of simultaneous threads/workers self-playing
        # to feed the replay buffer
        self.num_workers = 2
        self.selfplay_on_gpu = False
        # Maximum number of moves if game is not finished before
        self.max_moves = 100
        # Number of future moves self-simulated NOTE: from paper
        self.num_simulations = 50
        # Chronological discount of the reward
        self.discount = 1
        # Number of moves before dropping the temperature given by visit_
        # softmax_temperature_fn to 0 (ie selecting the best action).
        # If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a
        # vector with a range of -support_size to support_size.
        # Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network,
        # False / "CNN" (lighter) / "resnet"
        # (See paper appendix Network Architecture)
        self.downsample = False
        # Number of blocks in the ResNet
        self.blocks = 16
        # Number of channels in the ResNet
        self.channels = 128
        # Number of channels in reward head
        self.reduced_channels_reward = 2
        # Number of channels in value head
        self.reduced_channels_value = 2
        # Number of channels in policy head
        self.reduced_channels_policy = 4
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64]

        # Fully Connected Network
        self.encoding_size = 32
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        # Define the hidden layers in the value network
        self.fc_value_layers = []
        # Define the hidden layers in the policy network
        self.fc_policy_layers = []

        ### Training
        # Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../results",
            os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-"
                "%M-%S"))
        # Save the checkpoint in results_path as model.checkpoint
        self.save_model = True
        # Total number of training steps
        # (ie weights update according to a batch)
        self.training_steps = 100_000
        # Number of parts of games to train on at each training step
        self.batch_size = 2048
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 10_000
        # Scale the value loss to avoid overfitting of the value function,
        # paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        # Train on GPU if available
        self.train_on_gpu = torch.cuda.is_available()

        # "Adam" or "SGD". Paper uses SGD
        self.optimizer = "Adam"
        # L2 weights regularization
        self.weight_decay = 1e-4
        # Used only if optimizer is SGD
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.002
        # Set it to 1 to use a constant learning rate
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 10000

        ### Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 1_000_000
        # Number of game moves to keep for every batch element
        self.num_unroll_steps = 5
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 5
        # Prioritized Replay (See paper appendix Training),
        # select in priority the elements in the replay buffer
        # which are unexpected for the network
        self.PER = True
        # How much prioritization is used,
        # 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix R
        # # Use the last model to provide a fresher, stable n-step value
        # (See paper appendix Reanalyze)eanalyse)
        self.use_last_model_value = True
        # "cpu" / "cuda"
        self.reanalyse_device = "cpu"
        # Number of GPUs to use for the reanalyse, it can be fractional,
        # don't forget to take the train worker and the
        # selfplay workers into account
        self.reanalyse_num_gpus = 0
        # Reanalyze (See paper appendix Reanalyse)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        # Number of seconds to wait after each played game
        self.self_play_delay = 0
        # Number of seconds to wait after each training step
        self.training_delay = 0
        # Desired training steps per self played step ratio.
        # Equivalent to a synchronous version, training can take much longer.
        # Set it to None to disable it
        self.ratio = 1

    def visit_softmax_temperature_fn(self, trained_steps: int) -> float:
        """
        Parameter to alter the visit count distribution to ensure that the
        action selection becomes greedier as training progresses.

        The smaller it is, the more likely the best action
        (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.env = AzulEnv(seed)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        observation, reward, done, info = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self) -> int:
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self) -> List[int]:
        """
        Should return the legal actions at each turn, if it is not available,
        it can return the whole action space.

        At each turn, the game have to be able to handle one of returned actions.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self) -> np.ndarray:
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self) -> None:
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def expert_agent(self) -> int:
        """
        Hard coded agent that MuZero faces to assess his progress in
        multiplayer games. It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def human_to_action(self) -> int:
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """

        colors: Dict[str, Color] = {
            "B": Color.BLUE,
            "Y": Color.YELLOW,
            "R": Color.RED,
            "K": Color.BLACK,
            "C": Color.CYAN
        }

        slot_choice = ""
        color_choice = ""
        line_choice = ""
        action_num = -1

        color_choices = "[" + ",".join(colors.keys()) + "]"

        while action_num not in self.legal_actions():
            print(f"Player {self.to_play()}")
            try:
                slot_choice = input("Enter the slot: [Cen, Fa{1-9}]: ")
                color_choice = input(f"Enter the color: {color_choices}: ")
                line_choice = input(f"Enter the line: [1-5]: ")

                if slot_choice == "Cen":
                    slot = 0
                else:
                    factory_number = line_choice[-1]
                    slot = int(factory_number)

                color = colors[color_choice]
                line = int(line_choice)

                action = Action(Slot(slot), Color(color), Line(line - 1))
                print(f"Action: {action}")

                action_num = action_num_from_action(action)
            except ValueError:
                print(
                    f"Could not parse {(slot_choice, color_choice, line_choice)}")

        return action_num

    def action_to_string(self, action_num: int) -> str:
        """
        Convert an action number to a string representing the action.

        Returns:
            String representing the action.
        """
        slot, color, line = action_from_action_num(action_num)
        return f"Slot: {slot}, Color: {color}, Line: {line}"
