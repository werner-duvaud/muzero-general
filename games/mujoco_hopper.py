import datetime
import pathlib

import torch

from config import MuZeroConfig as BaseMuZeroConfig

from games.continuous_game import ContinuousGame


class MuZeroConfig(BaseMuZeroConfig):
    def __init__(self):
        # fmt: off
        # Resources
        self.seed = 0  # Seed for NumPy, PyTorch and game resets.
        self.max_num_gpus = 1  # Use at most one GPU when CUDA is available.
        self.num_threads = 1  # PyTorch CPU threads per worker.

        # Game
        self.observation_shape = (1, 1, 11)  # Observation shape: (channels, height, width).
        self.observation_scale = 1  # Divide observations by this value before network inference.
        self.reward_scale = 1  # Divide rewards during training; evaluation restores environment units.
        self.action_space = 3  # Number of continuous action dimensions.
        self.players = list(range(1))  # Consecutive player IDs; one player for a single-agent environment.
        self.stacked_observations = 0  # Previous observations and actions appended to the current observation.
        self.muzero_player = 0  # Player ID controlled by MuZero during multiplayer evaluation.
        self.opponent = None  # Evaluation opponent: None, "random" or "expert"; does not affect training.

        # Self-play and search
        self.num_workers = 2  # Parallel self-play workers; also limits random collection workers.
        self.selfplay_batch_size = 4  # Independent games searched together per worker; 1 disables batching.
        self.selfplay_on_gpu = torch.cuda.is_available()  # Run self-play network inference on GPU when True; otherwise CPU.
        self.max_moves = 1000  # Maximum environment transitions per episode.
        self.num_simulations = 50  # MCTS simulations performed for each real action.
        self.discount = 0.997  # Reward discount factor per environment transition.
        self.temperature_threshold = None  # Episode move count before greedy play; None uses the schedule throughout.
        self.root_dirichlet_alpha = 0.25  # Dirichlet concentration for self-play root exploration noise.
        self.root_exploration_fraction = 0.25  # Fraction of root priors replaced with Dirichlet noise during self-play.
        self.pb_c_base = 19652  # Controls how MCTS prior exploration grows with node visits.
        self.pb_c_init = 1.25  # Initial strength of MCTS prior exploration.
        self.known_bounds = None  # Fixed search value bounds (min, max); None estimates them during search.

        # Continuous search and policy
        self.num_sampled_actions = 8  # Initial candidates at every node.
        self.pw_alpha = 0.49  # Progressive widening exponent; 0 keeps a fixed candidate count.
        self.node_prior = "uniform"  # "density" is an optional latent-density heuristic.
        self.log_std_clamp = (-5, 2)  # Bound Gaussian log standard deviations.
        self.entropy_loss_weight = 0.001  # Encourage entropy of the normalized tanh policy.

        # Network
        self.network = "fullyconnected"  # Network architecture: "resnet" or "fullyconnected".
        self.support_size = 20  # Transformed value/reward support: [-support_size, support_size].
        self.train_reward = True  # Include reward prediction in the training loss.

        # Residual network (used when network = "resnet")
        self.downsample = False  # Observation downsampling: False, "CNN" or "resnet" (See paper appendix Network Architecture).
        self.blocks = 1  # Residual blocks in the representation, dynamics and prediction networks.
        self.channels = 2  # Number of channels in the residual latent state.
        self.resnet_state_normalization = "global"  # Latent min/max normalization: "global" or per-"channel".
        self.reduced_channels_reward = 2  # Channels retained before the residual reward head.
        self.reduced_channels_value = 2  # Channels retained before the residual value head.
        self.reduced_channels_policy = 2  # Channels retained before the residual policy head.
        self.resnet_fc_reward_layers = []  # Hidden layer widths of the residual reward head.
        self.resnet_fc_value_layers = []  # Hidden layer widths of the residual value head.
        self.resnet_fc_policy_layers = []  # Hidden layer widths of the residual policy head.

        # Fully connected network (used when network = "fullyconnected")
        self.encoding_size = 64  # Number of features in the fully connected latent state.
        self.fc_representation_layers = [64, 64]  # Hidden layer widths of the fully connected representation network.
        self.fc_dynamics_layers = [64]  # Hidden layer widths of the fully connected dynamics network.
        self.fc_reward_layers = [64]  # Hidden layer widths of the fully connected reward head.
        self.fc_value_layers = [64]  # Hidden layer widths of the fully connected value head.
        self.fc_policy_layers = [64]  # Hidden layer widths of the fully connected policy head.

        # Training
        self.results_path = (  # Output directory for checkpoints, logs and evaluation reports.
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S-%f")
        )
        self.save_model = True  # Save model/replay snapshots and run metadata under results_path.
        self.training_steps = 200000  # Total gradient updates, including random-buffer warmup.
        self.temperature_schedule = [(0, 1.0), (50000, 0.5), (150000, 0.25)]  # Pairs of (gradient update, action-sampling temperature); zero is greedy.
        self.batch_size = 128  # Replay positions sampled for each gradient update.
        self.checkpoint_interval = 10  # Gradient updates between publishing weights to self-play workers.
        self.value_loss_weight = 0.25  # Value loss multiplier; paper recommends 0.25 (See paper appendix Reanalyze).
        self.policy_loss_weight = 1  # Multiplier on the policy prediction loss.
        self.train_on_gpu = torch.cuda.is_available()  # Use GPU for training when available; otherwise CPU.
        self.optimizer = "Adam"  # Optimizer: "Adam" or "SGD"; paper uses SGD.
        self.max_grad_norm = 10.0  # Clip the gradient norm before each update; None disables clipping.
        self.weight_decay = 0.0001  # L2 regularization strength.
        self.momentum = 0.9  # SGD momentum; unused with Adam.
        self.lr_init = 0.0003  # Initial learning rate.
        self.lr_decay_rate = 1  # Exponential decay factor; 1 keeps the learning rate constant.
        self.lr_decay_steps = 200000  # Gradient updates per learning rate decay factor.

        # Replay buffer
        self.replay_buffer_size = 500  # Maximum complete games retained in FIFO replay.
        self.num_unroll_steps = 10  # Recurrent model steps trained after each sampled state.
        self.td_steps = 50  # Reward horizon before value bootstrapping; episode end uses observed returns.
        self.PER = True  # Prioritize replay games and positions by prediction error (See paper appendix Training).
        self.PER_alpha = 0.5  # Priority exponent when PER is enabled; zero samples uniformly, paper suggests 1.

        # Initialization
        self.random_bootstrap_episodes = 0  # Uniform random episodes collected before fresh training; 0 disables it.
        self.random_bootstrap_keep_best = 3  # Highest-return random episodes retained, capped by replay_buffer_size.
        self.random_bootstrap_training_steps = 0  # Joint MuZero updates on retained random data before self-play starts.

        # Reanalysis (See paper appendix Reanalyse)
        self.reanalyse_batch_size = 32  # Maximum positions refreshed per reanalysis iteration.
        self.use_last_model_value = True  # Refresh replay value targets with the latest network (See paper appendix Reanalyze).
        self.reanalyse_policy = False  # Refresh replay policy targets with MCTS using the latest network.
        self.reanalyse_policy_ratio = 0.8  # Fraction of eligible samples that use refreshed policy targets.
        self.reanalyse_on_gpu = torch.cuda.is_available()  # Use GPU for reanalysis when available; otherwise CPU.

        # Synchronization
        self.self_play_delay = 0  # Seconds to pause after each self-play game; 0 disables the delay.
        self.training_delay = 0  # Seconds to pause after each gradient update; 0 disables the delay.
        self.ratio = 0.2  # Target gradient updates per new self-play transition; None disables pacing.

        # Evaluation
        self.evaluation_both_players = False  # Alternate player seats across seeded evaluation episodes.
        self.evaluation_episodes = 20  # Final evaluation episode count; 0 disables final evaluation.
        self.evaluation_seed = 1000000  # First seed of consecutive final evaluation episodes.
        self.evaluation_target = None  # Required mean environment reward; None disables the pass threshold.
        self.validation_interval = 10000  # Gradient updates between validation checkpoints; 0 disables validation.
        self.validation_episodes = 10  # Episodes used to evaluate each validation checkpoint.
        self.validation_seed = 5000000  # First validation seed; keep validation and final evaluation seeds separate.
        self.checkpoint_selection = "best"  # "best" selects the highest validation mean; "last" keeps final weights.
        # fmt: on


class Game(ContinuousGame):
    """Hopper-v5 with normalized continuous controls."""

    env_id = "Hopper-v5"
