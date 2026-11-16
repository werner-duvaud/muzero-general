import datetime
import pathlib

import numpy
import torch

from config import MuZeroConfig as BaseMuZeroConfig

from games.abstract_game import AbstractGame

# OpenSpiel adapter for deterministic games with one player or two alternating players.

try:
    import open_spiel.python.algorithms.minimax
    import pyspiel

except ImportError as error:
    raise ModuleNotFoundError(
        "You need to install open_spiel by running pip install open_spiel. For a full documentation, see: https://github.com/deepmind/open_spiel/blob/master/docs/install.md"
    ) from error

# The game you want to run. See https://github.com/deepmind/open_spiel/blob/master/docs/games.md for a list of games.
game = pyspiel.load_game("tic_tac_toe")


def get_observation_tensor_shape(spiel_game):
    """Pad vector/matrix observations to (channels, height, width)."""
    if not spiel_game.get_type().provides_observation_tensor:
        raise ValueError("The OpenSpiel adapter requires an observation tensor.")
    shape = tuple(spiel_game.observation_tensor_shape())
    if not 1 <= len(shape) <= 3 or any(size <= 0 for size in shape):
        raise ValueError(
            f"OpenSpiel observation tensors must have 1 to 3 nonempty dimensions; got {shape}."
        )
    return (1,) * (3 - len(shape)) + shape


def validate_game(spiel_game):
    """Reject game types the current MuZero search cannot represent."""
    game_type = spiel_game.get_type()
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The OpenSpiel adapter requires sequential player turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError(
            "The OpenSpiel adapter does not support chance turns (for example, "
            "backgammon dice rolls). Use a deterministic game with alternating "
            "player turns; padding its observation tensor alone is insufficient."
        )
    if spiel_game.num_players() not in (1, 2):
        raise ValueError("The OpenSpiel adapter supports only one or two players.")
    if (
        spiel_game.num_players() == 2
        and game_type.utility != pyspiel.GameType.Utility.ZERO_SUM
    ):
        raise ValueError("Two-player OpenSpiel games must have zero-sum rewards.")


class MuZeroConfig(BaseMuZeroConfig):
    def __init__(self):
        validate_game(game)
        # fmt: off
        # Resources
        self.seed = 0  # Seed for NumPy, PyTorch and game resets.
        self.max_num_gpus = 1  # Use at most one GPU when CUDA is available.
        self.num_threads = 1  # PyTorch CPU threads per worker.

        # Game
        self.observation_shape = get_observation_tensor_shape(game)  # Observation shape: (channels, height, width).
        self.observation_scale = 1  # Divide observations by this value before network inference.
        self.reward_scale = 1  # Divide rewards during training; evaluation restores environment units.
        self.action_space = list(range(game.num_distinct_actions()))  # Consecutive discrete action IDs, or the number of continuous action dimensions.
        self.players = list(range(game.num_players()))  # Consecutive player IDs; one player for a single-agent environment.
        self.stacked_observations = 0  # Previous observations and actions appended to the current observation.
        self.muzero_player = 0  # Player ID controlled by MuZero during multiplayer evaluation.
        self.opponent = "expert" if game.get_type().short_name == "tic_tac_toe" else "random"  # Evaluation opponent: None, "random" or "expert"; does not affect training.

        # Self-play and search
        self.num_workers = 2  # Parallel self-play workers; also limits random collection workers.
        self.selfplay_batch_size = 4  # Independent games searched together per worker; 1 disables batching.
        self.selfplay_on_gpu = torch.cuda.is_available()  # Run self-play network inference on GPU when True; otherwise CPU.
        self.max_moves = game.max_game_length()  # Maximum environment transitions per episode.
        self.num_simulations = 100  # MCTS simulations performed for each real action.
        self.discount = 1  # Reward discount factor per environment transition.
        self.temperature_threshold = None  # Episode move count before greedy play; None uses the schedule throughout.
        self.root_dirichlet_alpha = 0.3  # Dirichlet concentration for self-play root exploration noise.
        self.root_exploration_fraction = 0.5  # Fraction of root priors replaced with Dirichlet noise during self-play.
        self.pb_c_base = 19652  # Controls how MCTS prior exploration grows with node visits.
        self.pb_c_init = 2.5  # Initial strength of MCTS prior exploration.
        self.known_bounds = (-1, 1) if game.get_type().short_name == "tic_tac_toe" else None  # Fixed search value bounds (min, max); None estimates them during search.

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
        self.training_steps = 1000000000  # Total gradient updates, including random-buffer warmup.
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
        self.lr_init = 0.0003  # Initial learning rate.
        self.lr_decay_rate = 1  # Exponential decay factor; 1 keeps the learning rate constant.
        self.lr_decay_steps = 10000  # Gradient updates per learning rate decay factor.

        # Replay buffer
        self.replay_buffer_size = 5000  # Maximum complete games retained in FIFO replay.
        self.num_unroll_steps = 3  # Recurrent model steps trained after each sampled state.
        self.td_steps = game.max_game_length()  # Reward horizon before value bootstrapping; episode end uses observed returns.
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
        self.evaluation_both_players = game.num_players() == 2  # Alternate player seats across seeded evaluation episodes.
        self.evaluation_episodes = 200  # Final evaluation episode count; 0 disables final evaluation.
        self.evaluation_seed = 1000000  # First seed of consecutive final evaluation episodes.
        self.evaluation_target = 0 if game.get_type().short_name == "tic_tac_toe" else None  # Required mean environment reward; None disables the pass threshold.
        self.validation_interval = 2000  # Gradient updates between validation checkpoints; 0 disables validation.
        self.validation_episodes = 40  # Episodes used to evaluate each validation checkpoint.
        self.validation_seed = 5000000  # First validation seed; keep validation and final evaluation seeds separate.
        self.checkpoint_selection = "best"  # "best" selects the highest validation mean; "last" keeps final weights.
        # fmt: on


class Game(AbstractGame):
    """Game wrapper."""

    def __init__(self, seed=None, render=False):
        self.env = OpenSpiel()

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
                print("Legal Actions: ", self.legal_actions_human())
                choice = input("Enter your move: ")
                if choice in self.legal_actions_human():
                    break
            except (RuntimeError, ValueError):
                pass
            print("Wrong input, try again")

        return self.env.board.string_to_action(choice)

    def expert_agent(self):
        """Sample uniformly among optimal moves, suitable for small games."""
        actions = self.legal_actions()
        values = [
            open_spiel.python.algorithms.minimax.alpha_beta_search(
                self.env.game,
                state=self.env.board.child(action),
                maximum_depth=self.env.game.max_game_length(),
                maximizing_player_id=self.to_play(),
            )[0]
            for action in actions
        ]
        best_value = max(values)
        return int(
            numpy.random.choice(
                [
                    action
                    for action, value in zip(actions, values)
                    if value == best_value
                ]
            )
        )

    def action_to_string(self, action_number):
        """Return a human-readable label for an action ID.

        Args:
            action_number (int): Action ID from the configured action space.

        Returns:
            str: Description of the action.
        """
        return self.env.board.action_to_string(self.to_play(), action_number)

    def legal_actions_human(self):
        """Return human-readable labels for the current legal actions."""
        return self.env.human_legal_actions()


class OpenSpiel:
    def __init__(self):
        self.game = game
        validate_game(self.game)
        self.observation_shape = get_observation_tensor_shape(self.game)
        self.board = self.game.new_initial_state()
        self.player = self.board.current_player()

    def step(self, action):
        acting_player = self.player
        board = self.board.child(action)
        done = board.is_terminal()
        next_player = (acting_player + 1) % self.game.num_players()
        if not done and board.current_player() != next_player:
            raise ValueError(
                "The OpenSpiel adapter requires strictly alternating turns."
            )
        reward = board.rewards()[acting_player]
        self.board = board
        self.player = next_player
        observation = self.get_observation()
        return observation, reward, done

    def to_play(self):
        return self.player

    def legal_actions(self):
        return self.board.legal_actions()

    def reset(self):
        self.board = self.game.new_initial_state()
        self.player = self.board.current_player()
        return self.get_observation()

    def render(self):
        print(self.board)

    def get_observation(self):
        return numpy.array(self.board.observation_tensor(self.player)).reshape(
            self.observation_shape
        )

    def human_legal_actions(self):
        return [self.board.action_to_string(x) for x in self.board.legal_actions()]
