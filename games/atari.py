import datetime
import pathlib

import ale_py
import gymnasium as gym
import numpy
import torch

from config import MuZeroConfig as BaseMuZeroConfig

from .abstract_game import AbstractGame

gym.register_envs(ale_py)

try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install dependencies from requirements.txt")


class MuZeroConfig(BaseMuZeroConfig):
    def __init__(self):
        # fmt: off
        # Resources
        self.seed = 0  # Seed for NumPy, PyTorch and game resets.
        self.max_num_gpus = None  # GPU limit; None uses all available GPUs.
        self.num_threads = 1  # PyTorch CPU threads per worker.

        # Game
        self.observation_shape = (3, 96, 96)  # Observation shape: (channels, height, width).
        self.observation_scale = 255  # Divide observations by this value before network inference.
        self.reward_scale = 1  # Divide rewards during training; evaluation restores environment units.
        self.action_space = list(range(4))  # Consecutive action IDs starting at zero; change the number of actions.
        self.players = list(range(1))  # Consecutive player IDs; one player for a single-agent environment.
        self.stacked_observations = 32  # Previous observations and actions appended to the current observation.
        self.muzero_player = 0  # Player ID controlled by MuZero during multiplayer evaluation.
        self.opponent = None  # Evaluation opponent: None, "random" or "expert"; does not affect training.

        # Self-play and search
        self.num_workers = 350  # Parallel self-play workers; also limits random collection workers.
        self.selfplay_on_gpu = False  # Run self-play network inference on GPU.
        self.max_moves = 27000  # Maximum environment transitions per episode.
        self.num_simulations = 50  # MCTS simulations performed for each real action.
        self.discount = 0.997  # Reward discount factor per environment transition.
        self.temperature_threshold = None  # Episode move count before greedy play; None uses the schedule throughout.
        self.root_dirichlet_alpha = 0.25  # Dirichlet concentration for self-play root exploration noise.
        self.root_exploration_fraction = 0.25  # Fraction of root priors replaced with Dirichlet noise during self-play.
        self.pb_c_base = 19652  # Controls how MCTS prior exploration grows with node visits.
        self.pb_c_init = 1.25  # Initial strength of MCTS prior exploration.
        self.known_bounds = None  # Fixed search value bounds (min, max); None estimates them during search.

        # Network
        self.network = "resnet"  # Network architecture: "resnet" or "fullyconnected".
        self.support_size = 300  # Transformed value/reward support: [-support_size, support_size].
        self.train_reward = True  # Include reward prediction in the training loss.

        # Residual network (used when network = "resnet")
        self.downsample = "resnet"  # Observation downsampling: False, "CNN" or "resnet" (See paper appendix Network Architecture).
        self.blocks = 16  # Residual blocks in the representation, dynamics and prediction networks.
        self.channels = 256  # Number of channels in the residual latent state.
        self.resnet_state_normalization = "global"  # Latent min/max normalization: "global" or per-"channel".
        self.reduced_channels_reward = 256  # Channels retained before the residual reward head.
        self.reduced_channels_value = 256  # Channels retained before the residual value head.
        self.reduced_channels_policy = 256  # Channels retained before the residual policy head.
        self.resnet_fc_reward_layers = [256, 256]  # Hidden layer widths of the residual reward head.
        self.resnet_fc_value_layers = [256, 256]  # Hidden layer widths of the residual value head.
        self.resnet_fc_policy_layers = [256, 256]  # Hidden layer widths of the residual policy head.

        # Fully connected network (used when network = "fullyconnected")
        self.encoding_size = 10  # Number of features in the fully connected latent state.
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
        self.training_steps = 1000000  # Total gradient updates, including random-buffer warmup.
        self.temperature_schedule = [(0, 1.0), (500000, 0.5), (750000, 0.25)]  # Pairs of (gradient update, action-sampling temperature); zero is greedy.
        self.batch_size = 1024  # Replay positions sampled for each gradient update.
        self.checkpoint_interval = 1000  # Gradient updates between publishing weights to self-play workers.
        self.value_loss_weight = 0.25  # Value loss multiplier; paper recommends 0.25 (See paper appendix Reanalyze).
        self.policy_loss_weight = 1  # Multiplier on the policy prediction loss.
        self.train_on_gpu = torch.cuda.is_available()  # Train the network on GPU rather than CPU.
        self.optimizer = "SGD"  # Optimizer: "Adam" or "SGD"; paper uses SGD.
        self.weight_decay = 0.0001  # L2 regularization strength.
        self.momentum = 0.9  # SGD momentum; unused with Adam.
        self.lr_init = 0.05  # Initial learning rate.
        self.lr_decay_rate = 0.1  # Exponential decay factor; 1 keeps the learning rate constant.
        self.lr_decay_steps = 350000.0  # Gradient updates per learning rate decay factor.

        # Replay buffer
        self.replay_buffer_size = 1000000  # Maximum complete games retained in FIFO replay.
        self.num_unroll_steps = 5  # Recurrent model steps trained after each sampled state.
        self.td_steps = 10  # Reward horizon before value bootstrapping; episode end uses observed returns.
        self.PER = True  # Prioritize replay games and positions by prediction error (See paper appendix Training).
        self.PER_alpha = 1  # Priority exponent when PER is enabled; zero samples uniformly, paper suggests 1.

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
        self.env = gym.make("ALE/Breakout-v5", render_mode="human" if render else None)
        self.seed = seed

    def step(self, action):
        """Apply a legal action and return the resulting transition.

        Args:
            action (int): Action ID from legal_actions.

        Returns:
            tuple: Observation in (channels, height, width) order, reward before
            config.reward_scale is applied, and an episode-ended boolean.
        """
        observation, reward, terminated, truncated, _ = self.env.step(action)
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="uint8")
        observation = numpy.moveaxis(observation, -1, 0)
        return observation, reward, terminated or truncated

    def legal_actions(self):
        """Return currently executable action IDs from the configured action space."""
        return list(range(4))

    def reset(self):
        """Start an episode, applying the constructor seed only on the first reset.

        Returns:
            array-like: Observation in (channels, height, width) order.
        """
        observation, _ = self.env.reset(seed=self.seed)
        self.seed = None
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = numpy.asarray(observation, dtype="uint8")
        observation = numpy.moveaxis(observation, -1, 0)
        return observation

    def close(self):
        """Release environment resources."""
        self.env.close()

    def render(self):
        """Display the game state and wait for Enter before continuing."""
        self.env.render()
        input("Press enter to take a step ")
