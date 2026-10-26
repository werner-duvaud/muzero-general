"""Self-contained MuZero defaults and a simplified Twenty-One card game.

Aces always count as one. Face cards count as ten, and the dealer draws
until reaching at least seventeen. There is no special natural-21 payout.
"""

import datetime
import pathlib

import numpy
import torch

from config import MuZeroConfig as BaseMuZeroConfig

from .abstract_game import AbstractGame


class MuZeroConfig(BaseMuZeroConfig):
    def __init__(self):
        # fmt: off
        # Resources
        self.seed = 0  # Seed for NumPy, PyTorch and game resets.
        self.max_num_gpus = None  # GPU limit; None uses all available GPUs.
        self.num_threads = 1  # PyTorch CPU threads per worker.

        # Game
        self.observation_shape = (3, 3, 3)  # Observation shape: (channels, height, width).
        self.observation_scale = 1  # Divide observations by this value before network inference.
        self.reward_scale = 0.1  # Divide rewards during training; evaluation restores environment units.
        self.action_space = list(range(2))  # Consecutive action IDs starting at zero; change the number of actions.
        self.players = list(range(1))  # Consecutive player IDs; one player for a single-agent environment.
        self.stacked_observations = 0  # Previous observations and actions appended to the current observation.
        self.muzero_player = 0  # Player ID controlled by MuZero during multiplayer evaluation.
        self.opponent = None  # Evaluation opponent: None, "random" or "expert"; does not affect training.

        # Self-play and search
        self.num_workers = 4  # Parallel self-play workers; also limits random collection workers.
        self.selfplay_on_gpu = False  # Run self-play network inference on GPU.
        self.max_moves = 21  # Maximum environment transitions per episode.
        self.num_simulations = 21  # MCTS simulations performed for each real action.
        self.discount = 1  # Reward discount factor per environment transition.
        self.temperature_threshold = None  # Episode move count before greedy play; None uses the schedule throughout.
        self.root_dirichlet_alpha = 0.25  # Dirichlet concentration for self-play root exploration noise.
        self.root_exploration_fraction = 0.25  # Fraction of root priors replaced with Dirichlet noise during self-play.
        self.pb_c_base = 19652  # Controls how MCTS prior exploration grows with node visits.
        self.pb_c_init = 1.25  # Initial strength of MCTS prior exploration.
        self.known_bounds = None  # Fixed search value bounds (min, max); None estimates them during search.

        # Network
        self.network = "resnet"  # Network architecture: "resnet" or "fullyconnected".
        self.support_size = 10  # Transformed value/reward support: [-support_size, support_size].
        self.train_reward = True  # Include reward prediction in the training loss.

        # Residual network (used when network = "resnet")
        self.downsample = False  # Observation downsampling: False, "CNN" or "resnet" (See paper appendix Network Architecture).
        self.blocks = 2  # Residual blocks in the representation, dynamics and prediction networks.
        self.channels = 32  # Number of channels in the residual latent state.
        self.resnet_state_normalization = "global"  # Latent min/max normalization: "global" or per-"channel".
        self.reduced_channels_reward = 32  # Channels retained before the residual reward head.
        self.reduced_channels_value = 32  # Channels retained before the residual value head.
        self.reduced_channels_policy = 32  # Channels retained before the residual policy head.
        self.resnet_fc_reward_layers = [16]  # Hidden layer widths of the residual reward head.
        self.resnet_fc_value_layers = [16]  # Hidden layer widths of the residual value head.
        self.resnet_fc_policy_layers = [16]  # Hidden layer widths of the residual policy head.

        # Fully connected network (used when network = "fullyconnected")
        self.encoding_size = 32  # Number of features in the fully connected latent state.
        self.fc_representation_layers = [16]  # Hidden layer widths of the fully connected representation network.
        self.fc_dynamics_layers = [16]  # Hidden layer widths of the fully connected dynamics network.
        self.fc_reward_layers = [16]  # Hidden layer widths of the fully connected reward head.
        self.fc_value_layers = [16]  # Hidden layer widths of the fully connected value head.
        self.fc_policy_layers = [16]  # Hidden layer widths of the fully connected policy head.

        # Training
        self.results_path = (  # Output directory for checkpoints, logs and evaluation reports.
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
        )
        self.save_model = True  # Save model/replay snapshots and run metadata under results_path.
        self.training_steps = 15000  # Total gradient updates, including random-buffer warmup.
        self.temperature_schedule = [(0, 1.0), (500000, 0.5), (750000, 0.25)]  # Pairs of (gradient update, action-sampling temperature); zero is greedy.
        self.batch_size = 64  # Replay positions sampled for each gradient update.
        self.checkpoint_interval = 10  # Gradient updates between publishing weights to self-play workers.
        self.value_loss_weight = 0.25  # Value loss multiplier; paper recommends 0.25 (See paper appendix Reanalyze).
        self.policy_loss_weight = 1  # Multiplier on the policy prediction loss.
        self.train_on_gpu = torch.cuda.is_available()  # Train the network on GPU rather than CPU.
        self.optimizer = "SGD"  # Optimizer: "Adam" or "SGD"; paper uses SGD.
        self.weight_decay = 0.0001  # L2 regularization strength.
        self.momentum = 0.9  # SGD momentum; unused with Adam.
        self.lr_init = 0.03  # Initial learning rate.
        self.lr_decay_rate = 0.75  # Exponential decay factor; 1 keeps the learning rate constant.
        self.lr_decay_steps = 150000  # Gradient updates per learning rate decay factor.

        # Replay buffer
        self.replay_buffer_size = 10000  # Maximum complete games retained in FIFO replay.
        self.num_unroll_steps = 20  # Recurrent model steps trained after each sampled state.
        self.td_steps = 50  # Reward horizon before value bootstrapping; episode end uses observed returns.
        self.PER = True  # Prioritize replay games and positions by prediction error (See paper appendix Training).
        self.PER_alpha = 0.5  # Priority exponent when PER is enabled; zero samples uniformly, paper suggests 1.

        # Initialization
        self.random_bootstrap_episodes = 0  # Uniform random episodes collected before fresh training; 0 disables it.
        self.random_bootstrap_keep_best = 100  # Highest-return random episodes retained, capped by replay_buffer_size.
        self.random_bootstrap_training_steps = 0  # Joint MuZero updates on retained random data before self-play starts.

        # Reanalysis (See paper appendix Reanalyse)
        self.reanalyse_batch_size = 32  # Maximum positions refreshed per reanalysis iteration.
        self.use_last_model_value = True  # Refresh replay value targets with the latest network (See paper appendix Reanalyze).
        self.reanalyse_policy = True  # Refresh replay policy targets with MCTS using the latest network.
        self.reanalyse_policy_ratio = 0.8  # Fraction of eligible samples that use refreshed policy targets.
        self.reanalyse_on_gpu = False  # Run reanalysis network inference on GPU.

        # Synchronization
        self.self_play_delay = 0  # Seconds to pause after each self-play game; 0 disables the delay.
        self.training_delay = 0  # Seconds to pause after each gradient update; 0 disables the delay.
        self.ratio = None  # Target gradient updates per new self-play transition; None disables pacing.

        # Evaluation
        self.evaluation_episodes = 100  # Final evaluation episode count; 0 disables final evaluation.
        self.evaluation_seed = 1000000  # First seed of consecutive final evaluation episodes.
        self.evaluation_target = None  # Required mean environment reward; None disables the pass threshold.
        self.validation_interval = 0  # Gradient updates between validation checkpoints; 0 disables validation.
        self.validation_episodes = 20  # Episodes used to evaluate each validation checkpoint.
        self.validation_seed = 5000000  # First validation seed; keep validation and final evaluation seeds separate.
        self.checkpoint_selection = "last"  # "best" selects the highest validation mean; "last" keeps final weights.
        # fmt: on


class Game(AbstractGame):
    """Game wrapper."""

    def __init__(self, seed=None, render=False):
        self.env = TwentyOne(seed)

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
        choice = input(
            f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """Return a human-readable label for an action ID.

        Args:
            action_number (int): Action ID from the configured action space.

        Returns:
            str: Description of the action.
        """
        actions = {
            0: "Hit",
            1: "Stand",
        }
        return f"{action_number}. {actions[action_number]}"


class TwentyOne:
    def __init__(self, seed):
        self.random = numpy.random.RandomState(seed)

        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()

    def reset(self):
        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()
        return self.get_observation()

    def step(self, action):
        """Apply a hit (action 0) or stand (action 1).

        End the episode on a bust, a stand, or reaching twenty-one, then resolve the
        dealer's play. Return the new observation, reward, and done flag. Rewards are
        zero until the episode ends.
        """
        if action == 0:
            self.player_hand += self.deal_card_value()

        done = self.is_busted() or action == 1 or self.player_hand == 21

        if done:
            self.dealer_plays()

        return self.get_observation(), self.get_reward(done), done

    def get_observation(self):
        return [
            numpy.full((3, 3), self.player_hand, dtype="float32"),
            numpy.full((3, 3), self.dealer_hand, dtype="float32"),
            numpy.full((3, 3), 0),
        ]

    def legal_actions(self):
        # 0 = hit
        # 1 = stand
        return [0, 1]

    def get_reward(self, done):
        if not done:
            return 0
        if self.player_hand <= 21 and self.dealer_hand < self.player_hand:
            return 1
        if self.player_hand <= 21 and self.dealer_hand > 21:
            return 1
        if self.player_hand > 21:
            return -1
        if self.player_hand == self.dealer_hand:
            return 0
        return -1

    def deal_card_value(self):
        card = self.random.randint(1, 14)
        if card >= 10:
            value = 10
        else:
            value = card
        return value

    def dealer_plays(self):
        if self.player_hand > 21:
            return
        while self.dealer_hand <= 16:
            self.dealer_hand += self.deal_card_value()

    def is_busted(self):
        return self.player_hand > 21

    def render(self):
        print("Dealer hand: " + str(self.dealer_hand))
        print("Player hand: " + str(self.player_hand))
