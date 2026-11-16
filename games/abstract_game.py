from abc import ABC, abstractmethod


class AbstractGame(ABC):
    """Inherit this class for MuZero to play your game.

    Observations must match config.observation_shape: (channels, height, width).
    Vectors use the same 3D layout, e.g. (1, 1, 4) for inverted pendulum.
    """

    @abstractmethod
    def __init__(self, seed=None, render=False):
        """Initialize a game, optionally using a seed and enabling visual rendering.

        Args:
            seed (int, optional): Seed for environments with stochastic behavior.
            render (bool): Enable visual rendering where supported.
        """
        pass

    @abstractmethod
    def step(self, action):
        """Apply a legal action and return the resulting transition.

        Args:
            action (array-like): Finite pre-tanh continuous action vector.

        Returns:
            tuple: Observation in (channels, height, width) order, reward before
            config.reward_scale is applied, and an episode-ended boolean.
        """
        pass

    def to_play(self):
        """Return the current player ID; the single-player default is zero."""
        return 0

    @abstractmethod
    def legal_actions(self):
        """Return None for unrestricted continuous actions."""
        pass

    @abstractmethod
    def reset(self):
        """Start a new episode and return its initial observation.

        Returns:
            array-like: Observation in (channels, height, width) order.
        """
        pass

    def close(self):
        """Release environment resources; the default implementation does nothing."""
        pass

    @abstractmethod
    def render(self):
        """Display the game state using the adapter's rendering interface."""
        pass

    def human_to_action(self):
        """Read a continuous action vector; implemented by the game adapter."""
        raise NotImplementedError

    def expert_agent(self):
        """Return a heuristic opponent action for multiplayer evaluation.

        Raises:
            NotImplementedError: The adapter does not provide an expert opponent.
        """
        raise NotImplementedError

    def action_to_string(self, action_number):
        """Return a human-readable label for an action ID.

        Args:
            action_number (int): Action ID from the configured action space.

        Returns:
            str: Description of the action.
        """
        return str(action_number)
