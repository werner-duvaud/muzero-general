import matplotlib.pyplot as plt
import numpy
import seaborn
import torch

import models
from game_history import GameHistory
from self_play import MCTS, Node, SelfPlay, next_player


class DiagnoseModel:
    """Tools to understand the learned model.

    Args:
        checkpoint (dict): Checkpoint containing network weights.
        config (MuZeroConfig): Configuration matching the checkpoint architecture.
    """

    def __init__(self, checkpoint, config):
        self.config = config

        # Initialize the network.
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(checkpoint["weights"])
        self.model.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )  # on GPU if available since the DataParallel objects in MuZeroNetwork requires that.
        self.model.eval()

    @torch.no_grad()
    def get_virtual_trajectory_from_obs(
        self, observation, horizon, plot=True, to_play=0
    ):
        """Play a game using the learned model instead of the environment.

        Run MCTS with root exploration noise at each imagined state, choosing the
        most visited action and advancing with recurrent inference. Imagined states
        use the full configured action space.

        Args:
            observation (array-like): Initial, unstacked game observation.
            horizon (int): Number of imagined actions.
            plot (bool): Display trajectory diagnostic plots.
            to_play (int): Player at the initial state.

        Returns:
            TrajectoryInfo: Actions, rewards, and search diagnostics.
        """
        trajectory_info = TrajectoryInfo("Virtual trajectory", self.config)
        history = GameHistory()
        history.observation_history.append(observation)
        observation = history.get_stacked_observations(
            -1,
            self.config.stacked_observations,
            len(self.config.action_space),
            getattr(self.config, "observation_scale", 1),
        )
        root, mcts_info = MCTS(self.config).run(
            self.model, observation, self.config.action_space, to_play, True
        )
        trajectory_info.store_info(root, mcts_info, None, numpy.nan)

        virtual_to_play = to_play
        for i in range(horizon):
            action = SelfPlay.select_action(root, 0)

            # Players play turn by turn.
            virtual_to_play = next_player(virtual_to_play, self.config.players)

            # Generate new root.
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                root.hidden_state,
                torch.tensor([[action]]).to(root.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            root = Node(0)
            root.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            root, mcts_info = MCTS(self.config).run(
                self.model, None, self.config.action_space, virtual_to_play, True, root
            )
            trajectory_info.store_info(
                root, mcts_info, action, reward, new_prior_root_value=value
            )

        if plot:
            trajectory_info.plot_trajectory()

        return trajectory_info

    @torch.no_grad()
    def compare_virtual_with_real_trajectories(
        self, first_obs, game, horizon, plot=True
    ):
        """Compare a model trajectory with the same actions in the real game.

        The game must already be at first_obs. Real-state searches record diagnostics;
        the executed actions come from the virtual trajectory. Stop at episode end,
        an illegal action, or the horizon. Rewards use training units. Search graphs
        are saved as PDFs even when plot is False.

        Args:
            first_obs (array-like): Initial, unstacked game observation.
            game (AbstractGame): Environment positioned at first_obs.
            horizon (int): Maximum number of actions to compare.
            plot (bool): Display trajectory plots and open search graphs.

        Returns:
            tuple: Virtual TrajectoryInfo, real TrajectoryInfo, and the index of the
            first illegal virtual action, or None if no illegal action is encountered.
        """
        virtual_trajectory_info = self.get_virtual_trajectory_from_obs(
            first_obs, horizon, False
        )
        real_trajectory_info = TrajectoryInfo("Real trajectory", self.config)
        history = GameHistory()
        history.observation_history.append(first_obs)
        stacked_observation = history.get_stacked_observations(
            -1,
            self.config.stacked_observations,
            len(self.config.action_space),
            getattr(self.config, "observation_scale", 1),
        )
        trajectory_divergence_index = None
        real_trajectory_end_reason = "Reached horizon"

        # Illegal moves are masked at the root.
        root, mcts_info = MCTS(self.config).run(
            self.model,
            stacked_observation,
            game.legal_actions(),
            game.to_play(),
            True,
        )
        self.plot_mcts(root, plot)
        real_trajectory_info.store_info(root, mcts_info, None, numpy.nan)
        for i, action in enumerate(virtual_trajectory_info.action_history):
            # Follow virtual trajectory until it reaches an illegal move in the real env.
            if action not in game.legal_actions():
                trajectory_divergence_index = i
                real_trajectory_end_reason = (
                    f"Virtual trajectory reached an illegal move at timestep {i}."
                )
                break

            observation, reward, done = game.step(action)
            reward /= self.config.reward_scale
            history.action_history.append(action)
            history.observation_history.append(observation)
            if done:
                real_trajectory_info.action_history.append(action)
                real_trajectory_info.reward_history.append(reward)
                real_trajectory_end_reason = "Real trajectory reached Done"
                break
            stacked_observation = history.get_stacked_observations(
                -1,
                self.config.stacked_observations,
                len(self.config.action_space),
                getattr(self.config, "observation_scale", 1),
            )
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observation,
                game.legal_actions(),
                game.to_play(),
                True,
            )
            real_trajectory_info.store_info(root, mcts_info, action, reward)

        if plot:
            virtual_trajectory_info.plot_trajectory()
            real_trajectory_info.plot_trajectory()
            print(real_trajectory_end_reason)

        return (
            virtual_trajectory_info,
            real_trajectory_info,
            trajectory_divergence_index,
        )

    def close_all(self):
        """Close all Matplotlib figures."""
        plt.close("all")

    def plot_mcts(self, root, plot=True):
        """Save visited search nodes to mcts.pdf in the current directory.

        Args:
            root (Node): Root of the search tree.
            plot (bool): Open the rendered graph in a viewer.

        Returns:
            graphviz.Digraph or None: Rendered graph, or None when the Python
            graphviz package is unavailable.
        """
        try:
            from graphviz import Digraph
        except ModuleNotFoundError:
            print("Please install graphviz to get the MCTS plot.")
            return None

        graph = Digraph(comment="MCTS", engine="neato")
        graph.attr("graph", rankdir="LR", splines="true", overlap="false")
        id = 0

        def traverse(node, action, parent_id, best):
            nonlocal id
            node_id = id
            graph.node(
                str(node_id),
                label=f"Action: {action}\nValue: {node.value():.2f}\nVisit count: {node.visit_count}\nPrior: {node.prior:.2f}\nReward: {node.reward:.2f}",
                color="orange" if best else "black",
            )
            id += 1
            if parent_id is not None:
                graph.edge(str(parent_id), str(node_id), constraint="false")

            if len(node.children) != 0:
                best_visit_count = max(
                    [child.visit_count for child in node.children.values()]
                )
            else:
                best_visit_count = False
            for action, child in node.children.items():
                if child.visit_count != 0:
                    traverse(
                        child,
                        action,
                        node_id,
                        (
                            True
                            if best_visit_count
                            and child.visit_count == best_visit_count
                            else False
                        ),
                    )

        traverse(root, None, None, True)
        graph.node(str(0), color="red")
        # print(graph.source)
        graph.render("mcts", view=plot, cleanup=True, format="pdf")
        return graph


class TrajectoryInfo:
    """Store and display trajectory rewards and search information."""

    def __init__(self, title, config):
        self.title = title + ": "
        self.config = config
        self.action_history = []
        self.reward_history = []
        self.prior_policies = []
        self.policies_after_planning = []
        # Not implemented, need to store them in every nodes of the mcts.
        self.prior_values = []
        self.values_after_planning = [[numpy.nan] * len(self.config.action_space)]
        self.prior_root_value = []
        self.root_value_after_planning = []
        self.prior_rewards = [[numpy.nan] * len(self.config.action_space)]
        self.mcts_depth = []

    def store_info(self, root, mcts_info, action, reward, new_prior_root_value=None):
        """Append one step of action, reward, and search diagnostics."""
        if action is not None:
            self.action_history.append(action)
        if reward is not None:
            self.reward_history.append(reward)
        self.prior_policies.append(
            [
                (
                    root.children[action].prior
                    if action in root.children.keys()
                    else numpy.nan
                )
                for action in self.config.action_space
            ]
        )
        self.policies_after_planning.append(
            [
                (
                    root.children[action].visit_count / self.config.num_simulations
                    if action in root.children.keys()
                    else numpy.nan
                )
                for action in self.config.action_space
            ]
        )
        self.values_after_planning.append(
            [
                (
                    root.children[action].value()
                    if action in root.children.keys()
                    else numpy.nan
                )
                for action in self.config.action_space
            ]
        )
        self.prior_root_value.append(
            mcts_info["root_predicted_value"]
            if new_prior_root_value is None
            else new_prior_root_value
        )
        self.root_value_after_planning.append(root.value())
        self.prior_rewards.append(
            [
                (
                    root.children[action].reward
                    if action in root.children.keys()
                    else numpy.nan
                )
                for action in self.config.action_space
            ]
        )
        self.mcts_depth.append(mcts_info["max_tree_depth"])

    def plot_trajectory(self):
        """Print trajectory diagnostics and display their heatmaps without blocking."""
        name = "Prior policies"
        print(name, self.prior_policies, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.prior_policies,
            mask=numpy.isnan(self.prior_policies),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        name = "Policies after planning"
        print(name, self.policies_after_planning, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.policies_after_planning,
            mask=numpy.isnan(self.policies_after_planning),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        if 0 < len(self.action_history):
            name = "Action history"
            print(name, self.action_history, "\n")
            plt.figure(self.title + name)
            # ax = seaborn.lineplot(x=list(range(len(self.action_history))), y=self.action_history)
            ax = seaborn.heatmap(
                numpy.transpose([self.action_history]),
                mask=numpy.isnan(numpy.transpose([self.action_history])),
                xticklabels=False,
                annot=True,
            )
            ax.set(ylabel="Timestep")
            ax.set_title(name)

        name = "Values after planning"
        print(name, self.values_after_planning, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.values_after_planning,
            mask=numpy.isnan(self.values_after_planning),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        name = "Prior root value"
        print(name, self.prior_root_value, "\n")
        plt.figure(self.title + name)
        # ax = seaborn.lineplot(x=list(range(len(self.prior_root_value))), y=self.prior_root_value)
        ax = seaborn.heatmap(
            numpy.transpose([self.prior_root_value]),
            mask=numpy.isnan(numpy.transpose([self.prior_root_value])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        name = "Root value after planning"
        print(name, self.root_value_after_planning, "\n")
        plt.figure(self.title + name)
        # ax = seaborn.lineplot(x=list(range(len(self.root_value_after_planning))), y=self.root_value_after_planning)
        ax = seaborn.heatmap(
            numpy.transpose([self.root_value_after_planning]),
            mask=numpy.isnan(numpy.transpose([self.root_value_after_planning])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        name = "Prior rewards"
        print(name, self.prior_rewards, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.prior_rewards, mask=numpy.isnan(self.prior_rewards), annot=True
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        if 0 < len(self.reward_history):
            name = "Reward history"
            print(name, self.reward_history, "\n")
            plt.figure(self.title + name)
            # ax = seaborn.lineplot(x=list(range(len(self.reward_history))), y=self.reward_history)
            ax = seaborn.heatmap(
                numpy.transpose([self.reward_history]),
                mask=numpy.isnan(numpy.transpose([self.reward_history])),
                xticklabels=False,
                annot=True,
            )
            ax.set(ylabel="Timestep")
            ax.set_title(name)

        name = "MCTS depth"
        print(name, self.mcts_depth, "\n")
        plt.figure(self.title + name)
        # ax = seaborn.lineplot(x=list(range(len(self.mcts_depth))), y=self.mcts_depth)
        ax = seaborn.heatmap(
            numpy.transpose([self.mcts_depth]),
            mask=numpy.isnan(numpy.transpose([self.mcts_depth])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        plt.show(block=False)
