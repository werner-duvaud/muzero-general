import numpy


class GameHistory:
    """Trajectory data with explicit state and transition indexing.

    For ``N`` environment transitions:

    * observations and players have length ``N + 1`` and describe states;
    * actions and rewards have length ``N`` and describe outgoing transitions;
    * search policies and root values have length ``N`` and describe the
      non-terminal state from which each action was selected.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.legal_actions_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        self.reanalysed_child_visits = None
        # Prioritized experience replay metadata.
        self.priorities = None
        self.game_priority = None

    @staticmethod
    def policy_from_root(root, action_space):
        """Return visit fractions in action_space order, or zeros without visits."""
        if root is None:
            return [0.0 for _ in action_space]
        sum_visits = sum(child.visit_count for child in root.children.values())
        if sum_visits == 0:
            return [0.0 for _ in action_space]
        return [
            (
                root.children[action].visit_count / sum_visits
                if action in root.children
                else 0.0
            )
            for action in action_space
        ]

    def store_search_statistics(self, root, action_space, legal_actions):
        """Append search policy, root value, and legal actions for the current state."""
        self.child_visits.append(self.policy_from_root(root, action_space))
        self.root_values.append(root.value() if root is not None else None)
        self.legal_actions_history.append(list(legal_actions))

    def compute_target_value(
        self, index, td_steps, discount, use_reanalysed_value=False
    ):
        """Compute an n-step value from the perspective of state index.

        Use a root-value bootstrap when available, optionally from reanalysis. If
        lookahead reaches episode end, use discounted observed returns. States past
        the last transition have zero value.
        """
        transition_count = len(self.action_history)
        if index >= transition_count:
            return 0.0

        bootstrap_index = index + td_steps
        if bootstrap_index >= transition_count:
            # Completed trajectories keep their observed rewards. Cache full
            # returns so Monte Carlo training does not repeatedly sum a long tail.
            key = (discount, transition_count)
            if getattr(self, "_returns_cache_key", None) != key:
                returns = numpy.zeros(transition_count + 1, dtype="float64")
                for position in range(transition_count - 1, -1, -1):
                    sign = (
                        1
                        if self.to_play_history[position]
                        == self.to_play_history[position + 1]
                        else -1
                    )
                    returns[position] = (
                        self.reward_history[position]
                        + discount * sign * returns[position + 1]
                    )
                self._returns_cache_key = key
                self._returns_cache = returns
            return float(self._returns_cache[index])
        if bootstrap_index < len(self.root_values):
            root_values = self.root_values
            if (
                use_reanalysed_value
                and self.reanalysed_predicted_root_values is not None
            ):
                root_values = self.reanalysed_predicted_root_values
            bootstrap_value = root_values[bootstrap_index]
            if self.to_play_history[bootstrap_index] != self.to_play_history[index]:
                bootstrap_value = -bootstrap_value
            value = bootstrap_value * discount**td_steps
        else:
            value = 0.0

        reward_end = min(bootstrap_index, transition_count)
        for offset, transition_index in enumerate(range(index, reward_end)):
            reward = self.reward_history[transition_index]
            if self.to_play_history[transition_index] != self.to_play_history[index]:
                reward = -reward
            value += reward * discount**offset
        return value

    def make_target(
        self,
        state_index,
        num_unroll_steps,
        td_steps,
        discount,
        action_space,
        use_reanalysed_value=False,
        use_reanalysed_policy=False,
        train_reward=True,
    ):
        """Build aligned targets and validity masks for one sampled state.

        Prediction zero is the initial inference at state_index. Prediction k > 0
        follows action state_index + k - 1, so its reward belongs to that transition
        and its value/policy belong to state state_index + k.

        Root rewards are masked. Absorbing states beyond episode end have zero value
        and reward targets; their policies are masked. Reanalysis can replace stored
        root values and policies when requested.

        Returns:
            tuple: Values, rewards, policies, actions, value masks, reward masks, and
            policy masks. All lists have num_unroll_steps + 1 entries except actions,
            which has num_unroll_steps entries.
        """
        transition_count = len(self.action_history)
        zero_policy = [0.0 for _ in action_space]
        policy_history = (
            self.reanalysed_child_visits
            if use_reanalysed_policy and self.reanalysed_child_visits is not None
            else self.child_visits
        )

        values, rewards, policies = [], [], []
        value_masks, reward_masks, policy_masks = [], [], []
        actions = []

        for prediction_step in range(num_unroll_steps + 1):
            current_state = state_index + prediction_step

            if current_state < transition_count:
                values.append(
                    self.compute_target_value(
                        current_state,
                        td_steps,
                        discount,
                        use_reanalysed_value,
                    )
                )
                policies.append(policy_history[current_state])
                value_masks.append(1.0)
                policy_masks.append(1.0)
            else:
                # MuZero can search beyond episode end. Train the resulting
                # absorbing states to keep predicting zero future return.
                values.append(0.0)
                policies.append(zero_policy)
                value_masks.append(1.0)
                policy_masks.append(0.0)

            if prediction_step == 0:
                rewards.append(0.0)
                reward_masks.append(0.0)
                continue

            transition_index = current_state - 1
            if transition_index < transition_count:
                actions.append(self.action_history[transition_index])
                rewards.append(self.reward_history[transition_index])
                reward_masks.append(1.0 if train_reward else 0.0)
            else:
                # Every action in an absorbing state has zero reward and value.
                actions.append(numpy.random.choice(action_space))
                rewards.append(0.0)
                reward_masks.append(1.0 if train_reward else 0.0)

        return (
            values,
            rewards,
            policies,
            actions,
            value_masks,
            reward_masks,
            policy_masks,
        )

    def get_stacked_observations(
        self,
        index,
        num_stacked_observations,
        action_space_size,
        observation_scale=1,
        output=None,
    ):
        """Stack the current observation with past observations and action planes.

        Current channels come first, followed by past states from newest to oldest.
        Missing history is zero padded. Observation channels are divided by
        observation_scale; action planes contain action / action_space_size.

        Args:
            index (int): State index, wrapped modulo observation count.
            num_stacked_observations (int): Number of past states to append.
            action_space_size (int): Number of configured actions.
            observation_scale (float): Divisor for observation channels.
            output (numpy.ndarray, optional): Float32 destination array with the
                required stacked shape; filled and returned instead of allocating.

        Returns:
            numpy.ndarray: Float32 data with shape
            (C * (num_stacked_observations + 1) + num_stacked_observations, H, W).
        """
        index = index % len(self.observation_history)
        observation = numpy.asarray(self.observation_history[index])
        channels = observation.shape[0]
        shape = (
            channels * (num_stacked_observations + 1) + num_stacked_observations,
            *observation.shape[1:],
        )
        stacked_observations = (
            numpy.zeros(shape, dtype=numpy.float32) if output is None else output
        )
        if output is not None:
            stacked_observations.fill(0)
        stacked_observations[:channels] = observation
        stacked_observations[:channels] /= observation_scale

        channel = channels
        for past_index in range(index - 1, index - num_stacked_observations - 1, -1):
            if past_index >= 0:
                stacked_observations[channel : channel + channels] = (
                    self.observation_history[past_index]
                )
                stacked_observations[channel : channel + channels] /= observation_scale
                stacked_observations[channel + channels] = (
                    self.action_history[past_index] / action_space_size
                )
            channel += channels + 1
        return stacked_observations
