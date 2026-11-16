"""Common config behavior; each game declares all of its own settings explicitly."""

import json
import math
import pathlib


class MuZeroConfig:
    """Shared units and utilities for every game's self-contained configuration.

    ``reward_scale`` divides environment rewards only when recording training data.
    Evaluation reports the original environment returns. ``observation_scale``
    normalizes network inputs. Temperature schedules contain (training step,
    temperature) pairs. Evaluation uses consecutive seeds starting at
    ``evaluation_seed`` and compares their mean return with ``evaluation_target``;
    None disables the comparison. Zero evaluation episodes disables evaluation.
    Validation uses its own seeds to select checkpoints; final evaluation seeds
    are used only after training. ``checkpoint_selection="best"`` restores the
    snapshot with the highest mean validation return.

    Bootstrap episodes use independent uniform random legal actions. The best
    requested number initialize replay, followed by ordinary MuZero updates.
    """

    def visit_softmax_temperature_fn(self, trained_steps):
        """Return the scheduled temperature at a gradient update count."""
        temperature = self.temperature_schedule[0][1]
        for start, value in self.temperature_schedule:
            if trained_steps < start:
                break
            temperature = value
        return temperature

    def to_dict(self):
        """Return JSON-compatible settings with paths as strings and tuples as lists."""
        # JSON round-tripping also converts tuples to portable lists.
        return json.loads(json.dumps(vars(self), default=self._json_default))

    @staticmethod
    def _json_default(value):
        """Serialize a path or raise TypeError for unsupported config values."""
        if isinstance(value, pathlib.Path):
            return str(value)
        raise TypeError(f"Cannot serialize config value: {value!r}")

    def update(self, overrides):
        """Apply known field overrides, normalize path/tuple fields, and validate.

        Updates happen in place; validation errors do not roll back changes.

        Args:
            overrides (dict): Existing config fields and their replacement values.

        Raises:
            ValueError: An unknown field or invalid setting is supplied.
        """
        unknown = set(overrides) - set(vars(self))
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        for key, value in overrides.items():
            if key == "results_path":
                value = pathlib.Path(value)
            elif key in ("observation_shape", "known_bounds") and value is not None:
                value = tuple(value)
            setattr(self, key, value)
        self.validate()

    def validate(self):
        """Validate shared settings and reject incompatible options."""
        for name in (
            "num_workers",
            "selfplay_batch_size",
            "num_threads",
            "max_moves",
            "num_simulations",
            "training_steps",
            "batch_size",
            "checkpoint_interval",
            "replay_buffer_size",
            "num_unroll_steps",
            "td_steps",
            "random_bootstrap_keep_best",
            "reanalyse_batch_size",
            "validation_episodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "seed",
            "evaluation_seed",
            "evaluation_episodes",
            "random_bootstrap_episodes",
            "random_bootstrap_training_steps",
            "validation_seed",
            "validation_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("observation_scale", "reward_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if len(self.observation_shape) != 3:
            raise ValueError("observation_shape must have three dimensions")
        if (
            isinstance(self.action_space, bool)
            or not isinstance(self.action_space, int)
            or self.action_space < 1
        ):
            raise ValueError(
                "action_space must be a positive number of continuous dimensions"
            )
        if self.players != [0]:
            raise ValueError("Continuous environments require players = [0]")
        if self.network not in ("fullyconnected", "resnet"):
            raise ValueError("network must be fullyconnected or resnet")
        if self.node_prior not in ("uniform", "density"):
            raise ValueError("node_prior must be uniform or density")
        if (
            isinstance(self.num_sampled_actions, bool)
            or not isinstance(self.num_sampled_actions, int)
            or self.num_sampled_actions < 1
        ):
            raise ValueError("num_sampled_actions must be a positive integer")
        if not math.isfinite(self.pw_alpha) or not 0 <= self.pw_alpha <= 1:
            raise ValueError("pw_alpha must be between zero and one")
        if (
            len(self.log_std_clamp) != 2
            or not all(math.isfinite(x) for x in self.log_std_clamp)
            or not -20 <= self.log_std_clamp[0] < self.log_std_clamp[1] <= 5
        ):
            raise ValueError("log_std_clamp must satisfy -20 <= lower < upper <= 5")
        if not math.isfinite(self.entropy_loss_weight) or self.entropy_loss_weight < 0:
            raise ValueError("entropy_loss_weight must be finite and nonnegative")
        if self.stacked_observations < 0 or not isinstance(
            self.stacked_observations, int
        ):
            raise ValueError("stacked_observations must be a nonnegative integer")
        if self.muzero_player not in self.players:
            raise ValueError("muzero_player must be in players")
        both_players = self.evaluation_both_players
        if not isinstance(both_players, bool):
            raise TypeError("evaluation_both_players must be a boolean")
        if both_players and (
            len(self.players) != 2
            or self.evaluation_episodes % 2
            or (self.validation_interval and self.validation_episodes % 2)
        ):
            raise ValueError(
                "Evaluating both players requires two players and even episode counts"
            )
        max_grad_norm = self.max_grad_norm
        if max_grad_norm is not None and (
            isinstance(max_grad_norm, bool)
            or not math.isfinite(max_grad_norm)
            or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive and finite or None")
        if self.ratio is not None and self.ratio <= 0:
            raise ValueError("ratio must be positive or None")
        if not self.temperature_schedule or self.temperature_schedule[0][0] != 0:
            raise ValueError("temperature_schedule must start at step zero")
        starts = [step for step, _ in self.temperature_schedule]
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError("temperature_schedule steps must be strictly increasing")
        if any(
            not math.isfinite(value) or value < 0
            for _, value in self.temperature_schedule
        ):
            raise ValueError("temperatures must be nonnegative and finite")
        if self.evaluation_target is not None and not math.isfinite(
            self.evaluation_target
        ):
            raise ValueError("evaluation_target must be finite or None")
        if self.resnet_state_normalization not in ("global", "channel"):
            raise ValueError("resnet_state_normalization must be 'global' or 'channel'")
        if self.checkpoint_selection not in ("last", "best"):
            raise ValueError("checkpoint_selection must be 'last' or 'best'")
        if self.checkpoint_selection == "best" and not self.validation_interval:
            raise ValueError(
                "Selecting the best checkpoint requires a positive validation_interval"
            )
        if not math.isfinite(self.policy_loss_weight) or self.policy_loss_weight <= 0:
            raise ValueError("policy_loss_weight must be positive and finite")
        if self.validation_interval and max(
            self.validation_seed, self.evaluation_seed
        ) < min(
            self.validation_seed + self.validation_episodes,
            self.evaluation_seed + self.evaluation_episodes,
        ):
            raise ValueError("Validation and final evaluation seeds must not overlap")
