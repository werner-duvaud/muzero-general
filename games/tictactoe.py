import datetime
import functools
import pathlib

import numpy
import torch

from config import MuZeroConfig as BaseMuZeroConfig

from games.abstract_game import AbstractGame


class MuZeroConfig(BaseMuZeroConfig):
    def __init__(self):
        # fmt: off
        # Resources
        self.seed = 0  # Seed for NumPy, PyTorch and game resets.
        self.max_num_gpus = 1  # Use at most one GPU when CUDA is available.
        self.num_threads = 1  # PyTorch CPU threads per worker.

        # Game
        self.observation_shape = (3, 3, 3)  # Observation shape: (channels, height, width).
        self.observation_scale = 1  # Divide observations by this value before network inference.
        self.reward_scale = 1  # Divide rewards during training; evaluation restores environment units.
        self.action_space = list(range(9))  # Consecutive discrete action IDs, or the number of continuous action dimensions.
        self.players = list(range(2))  # Consecutive player IDs; one player for a single-agent environment.
        self.stacked_observations = 0  # Previous observations and actions appended to the current observation.
        self.muzero_player = 0  # Player ID controlled by MuZero during multiplayer evaluation.
        self.opponent = "expert"  # Evaluation opponent: None, "random" or "expert"; does not affect training.

        # Self-play and search
        self.num_workers = 2  # Parallel self-play workers; also limits random collection workers.
        self.selfplay_batch_size = 4  # Independent games searched together per worker; 1 disables batching.
        self.selfplay_on_gpu = torch.cuda.is_available()  # Run self-play network inference on GPU when True; otherwise CPU.
        self.max_moves = 9  # Maximum environment transitions per episode.
        self.num_simulations = 100  # MCTS simulations performed for each real action.
        self.discount = 1  # Reward discount factor per environment transition.
        self.temperature_threshold = None  # Episode move count before greedy play; None uses the schedule throughout.
        self.root_dirichlet_alpha = 0.3  # Dirichlet concentration for self-play root exploration noise.
        self.root_exploration_fraction = 0.5  # Fraction of root priors replaced with Dirichlet noise during self-play.
        self.pb_c_base = 19652  # Controls how MCTS prior exploration grows with node visits.
        self.pb_c_init = 2.5  # Initial strength of MCTS prior exploration.
        self.known_bounds = (-1, 1)  # Fixed search value bounds (min, max); None estimates them during search.

        # Network
        self.network = "resnet"  # Network architecture: "resnet" or "fullyconnected".
        self.support_size = 10  # Transformed value/reward support: [-support_size, support_size].
        self.train_reward = True  # Include reward prediction in the training loss.

        # Residual network (used when network = "resnet")
        self.downsample = False  # Observation downsampling: False, "CNN" or "resnet" (See paper appendix Network Architecture).
        self.blocks = 1  # Residual blocks in the representation, dynamics and prediction networks.
        self.channels = 16  # Number of channels in the residual latent state.
        self.resnet_state_normalization = "global"  # Latent min/max normalization: "global" or per-"channel".
        self.reduced_channels_reward = 16  # Channels retained before the residual reward head.
        self.reduced_channels_value = 16  # Channels retained before the residual value head.
        self.reduced_channels_policy = 16  # Channels retained before the residual policy head.
        self.resnet_fc_reward_layers = [8]  # Hidden layer widths of the residual reward head.
        self.resnet_fc_value_layers = [8]  # Hidden layer widths of the residual value head.
        self.resnet_fc_policy_layers = [8]  # Hidden layer widths of the residual policy head.

        # Fully connected network (used when network = "fullyconnected")
        self.encoding_size = 32  # Number of features in the fully connected latent state.
        self.fc_representation_layers = []  # Hidden layer widths of the fully connected representation network.
        self.fc_dynamics_layers = [16]  # Hidden layer widths of the fully connected dynamics network.
        self.fc_reward_layers = [16]  # Hidden layer widths of the fully connected reward head.
        self.fc_value_layers = []  # Hidden layer widths of the fully connected value head.
        self.fc_policy_layers = []  # Hidden layer widths of the fully connected policy head.

        # Training
        self.results_path = (  # Output directory for checkpoints, logs and evaluation reports.
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
        )
        self.save_model = True  # Save model/replay snapshots and run metadata under results_path.
        self.training_steps = 300000  # Total gradient updates, including random-buffer warmup.
        self.temperature_schedule = [(0, 1.0)]  # Pairs of (gradient update, action-sampling temperature); zero is greedy.
        self.batch_size = 256  # Replay positions sampled for each gradient update.
        self.checkpoint_interval = 10  # Gradient updates between publishing weights to self-play workers.
        self.value_loss_weight = 0.25  # Value loss multiplier; paper recommends 0.25 (See paper appendix Reanalyze).
        self.policy_loss_weight = 1  # Multiplier on the policy prediction loss.
        self.train_on_gpu = torch.cuda.is_available()  # Use GPU for training when available; otherwise CPU.
        self.optimizer = "Adam"  # Optimizer: "Adam" or "SGD"; paper uses SGD.
        self.max_grad_norm = 0.5  # Clip the gradient norm before each update; None disables clipping.
        self.weight_decay = 0.0001  # L2 regularization strength.
        self.momentum = 0.9  # SGD momentum; unused with Adam.
        self.lr_init = 0.003  # Initial learning rate.
        self.lr_decay_rate = 1  # Exponential decay factor; 1 keeps the learning rate constant.
        self.lr_decay_steps = 10000  # Gradient updates per learning rate decay factor.

        # Replay buffer
        self.replay_buffer_size = 1000  # Maximum complete games retained in FIFO replay.
        self.num_unroll_steps = 3  # Recurrent model steps trained after each sampled state.
        self.td_steps = 9  # Reward horizon before value bootstrapping; episode end uses observed returns.
        self.PER = False  # Prioritize replay games and positions by prediction error (See paper appendix Training).
        self.PER_alpha = 0.5  # Priority exponent when PER is enabled; zero samples uniformly, paper suggests 1.

        # Initialization
        self.random_bootstrap_episodes = 0  # Uniform random episodes collected before fresh training; 0 disables it.
        self.random_bootstrap_keep_best = 100  # Highest-return random episodes retained, capped by replay_buffer_size.
        self.random_bootstrap_training_steps = 0  # Joint MuZero updates on retained random data before self-play starts.

        # Reanalysis (See paper appendix Reanalyse)
        self.reanalyse_batch_size = 32  # Maximum positions refreshed per reanalysis iteration.
        self.use_last_model_value = False  # Refresh replay value targets with the latest network (See paper appendix Reanalyze).
        self.reanalyse_policy = False  # Refresh replay policy targets with MCTS using the latest network.
        self.reanalyse_policy_ratio = 0.8  # Fraction of eligible samples that use refreshed policy targets.
        self.reanalyse_on_gpu = torch.cuda.is_available()  # Use GPU for reanalysis when available; otherwise CPU.

        # Synchronization
        self.self_play_delay = 0  # Seconds to pause after each self-play game; 0 disables the delay.
        self.training_delay = 0  # Seconds to pause after each gradient update; 0 disables the delay.
        self.ratio = 1.0  # Target gradient updates per new self-play transition; None disables pacing.

        # Evaluation
        self.evaluation_both_players = True  # Alternate player seats across seeded evaluation episodes.
        self.evaluation_episodes = 200  # Final evaluation episode count; 0 disables final evaluation.
        self.evaluation_seed = 1000000  # First seed of consecutive final evaluation episodes.
        self.evaluation_target = 0  # Required mean environment reward; None disables the pass threshold.
        self.validation_interval = 2000  # Gradient updates between validation checkpoints; 0 disables validation.
        self.validation_episodes = 40  # Episodes used to evaluate each validation checkpoint.
        self.validation_seed = 5000000  # First validation seed; keep validation and final evaluation seeds separate.
        self.checkpoint_selection = "best"  # "best" selects the highest validation mean; "last" keeps final weights.
        # fmt: on


class Game(AbstractGame):
    """Game wrapper."""

    def __init__(self, seed=None, render=False):
        self.env = TicTacToe()

    def step(self, action):
        """Apply a legal action and return the resulting transition.

        Args:
            action (int): Action ID from legal_actions.

        Returns:
            tuple: Observation in (channels, height, width) order, reward before
            config.reward_scale is applied, and an episode-ended boolean.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """Return the current player ID from the configured players list."""
        return self.env.to_play()

    def legal_actions(self):
        """Return currently executable action IDs from the configured action space."""
        return self.env.legal_actions()

    def reset(self):
        """Start a new episode and return its initial observation.

        Returns:
            array-like: Observation in (channels, height, width) order.
        """
        return self.env.reset()

    def render(self):
        """Display the game state and wait for Enter before continuing."""
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """Prompt for a legal move and return its action ID."""
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                    choice in self.legal_actions()
                    and 1 <= row
                    and 1 <= col
                    and row <= 3
                    and col <= 3
                ):
                    break
            except ValueError:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """Return an optimal minimax action for multiplayer evaluation."""
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """Return a human-readable label for an action ID.

        Args:
            action_number (int): Action ID from the configured action space.

        Returns:
            str: Description of the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class TicTacToe:
    def __init__(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1

    def step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def to_play(self):
        return 0 if self.player == 1 else 1

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def reset(self):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1
        return self.get_observation()

    def render(self):
        print(self.board[::-1])

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((3, 3), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
            self.board[0, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[2, 2] == self.player
        ):
            return True
        if (
            self.board[2, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[0, 2] == self.player
        ):
            return True

        return False

    def expert_action(self):
        """Choose uniformly among optimal moves; used only as an opponent."""
        _, actions = self.solve(tuple(self.board.flat), self.player)
        return int(numpy.random.choice(actions))

    @staticmethod
    @functools.lru_cache(maxsize=6000)
    def solve(board, player):
        """Return the exact value and optimal actions for a reachable position."""
        game = TicTacToe()
        game.board = numpy.array(board, dtype="int32").reshape(3, 3)
        game.player = -player
        if game.have_winner():
            return -1, ()
        actions = game.legal_actions()
        if not actions:
            return 0, ()
        values = {}
        for action in actions:
            child = list(board)
            child[action] = player
            values[action] = -TicTacToe.solve(tuple(child), -player)[0]
        best = max(values.values())
        return best, tuple(action for action in actions if values[action] == best)
