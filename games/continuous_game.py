"""Gymnasium adapters for finite, vector-valued Box action spaces."""

import gymnasium as gym
import numpy

from games.abstract_game import AbstractGame


class ContinuousGame(AbstractGame):
    env_id = None

    def __init__(self, seed=None, render=False):
        self.env = gym.make(self.env_id, render_mode="human" if render else None)
        self.seed = seed
        space = self.env.action_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            self.env.close()
            raise ValueError("Expected a vector Box action space")
        if not numpy.isfinite(space.low).all() or not numpy.isfinite(space.high).all():
            self.env.close()
            raise ValueError("Continuous action bounds must be finite")

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(
            self.environment_action(action)
        )
        return (
            self.observation(observation),
            float(reward),
            bool(terminated or truncated),
        )

    def legal_actions(self):
        """All finite action vectors are valid; search samples its candidates."""
        return None

    def reset(self):
        observation, _ = self.env.reset(seed=self.seed)
        self.seed = None
        return self.observation(observation)

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def human_to_action(self):
        """Read normalized controls and convert them to policy coordinates."""
        value = numpy.asarray(
            input("Normalized controls (-1 to 1), space separated: ").split(),
            dtype="float32",
        )
        if (
            value.shape != self.env.action_space.shape
            or not numpy.isfinite(value).all()
            or (numpy.abs(value) > 1).any()
        ):
            raise ValueError("Invalid normalized control vector")
        return numpy.arctanh(numpy.clip(value, -1 + 1e-6, 1 - 1e-6))

    def action_to_string(self, action):
        return str(self.environment_action(getattr(action, "value", action)))

    def environment_action(self, action):
        """Map pre-tanh Gaussian coordinates onto the full environment bounds."""
        action = numpy.asarray(action, dtype="float32")
        space = self.env.action_space
        if action.shape != space.shape or not numpy.isfinite(action).all():
            raise ValueError(f"Expected a finite action vector of shape {space.shape}")
        normalized = numpy.tanh(action)
        control = space.low + (normalized + 1) * 0.5 * (space.high - space.low)
        return numpy.clip(control, space.low, space.high).astype(space.dtype)

    @staticmethod
    def observation(value):
        result = numpy.asarray(value, dtype="float32").reshape(1, 1, -1)
        if not numpy.isfinite(result).all():
            raise ValueError("Environment returned a nonfinite observation")
        return result
