import torch


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super(FullyConnectedNetwork, self).__init__()
        sizes_list = layer_sizes.copy()
        sizes_list.insert(0, input_size)
        layers = []
        if 1 < len(sizes_list):
            for i in range(len(sizes_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(sizes_list[i], sizes_list[i + 1]),
                        torch.nn.ReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(sizes_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MuZeroNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_size,
        stacked_observations,
        action_space_size,
        encoding_size,
        hidden_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = FullyConnectedNetwork(
            observation_size * (stacked_observations + 1), [], encoding_size
        )

        self.dynamics_encoded_state_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size, hidden_layers, encoding_size
        )
        self.dynamics_reward_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size,
            hidden_layers,
            self.full_support_size,
        )

        self.prediction_policy_network = FullyConnectedNetwork(
            encoding_size, [], self.action_space_size
        )
        self.prediction_value_network = FullyConnectedNetwork(
            encoding_size, [], self.full_support_size,
        )

    def prediction(self, encoded_state):
        policy_logit = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logit, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        encoded_state_diff = encoded_state - encoded_state.min(1, keepdim=True)[0]
        encoded_state_normalized = (
            encoded_state_diff / encoded_state_diff.max(1, keepdim=True)[0]
        )
        return encoded_state

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        next_encoded_state_diff = (
            next_encoded_state - next_encoded_state.min(1, keepdim=True)[0]
        )
        next_encoded_state_normalized = (
            next_encoded_state_diff / next_encoded_state_diff.max(1, keepdim=True)[0]
        )

        reward = self.dynamics_reward_network(x)
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logit, value = self.prediction(encoded_state)
        return (
            value,
            torch.zeros(len(observation), self.full_support_size).to(
                observation.device
            ),
            policy_logit,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logit, value = self.prediction(next_encoded_state)
        return value, reward, policy_logit, next_encoded_state

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)
