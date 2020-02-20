import torch


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.hidden_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.pooling_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.support_size,
            )
        else:
            raise ValueError(
                'The network parameter should be "fullyconnected" or "resnet"'
            )


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
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
            observation_shape[0]
            * observation_shape[1]
            * observation_shape[2]
            * (stacked_observations + 1),
            [],
            encoding_size,
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
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state == 0] = 1
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

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
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state == 0] = 1
        next_encoded_state_normalized = (
            encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

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


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self, observation_shape, stacked_observations, num_blocks, num_channels
    ):
        super(RepresentationNetwork, self).__init__()
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1), num_channels
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        return out


class DynamicNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        pooling_size,
        fc_reward_layers,
        full_support_size,
    ):
        super(DynamicNetwork, self).__init__()
        self.num_channels = num_channels
        self.observation_shape = observation_shape
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.pool = torch.nn.AvgPool2d(pooling_size, stride=2)
        self.fc = FullyConnectedNetwork(
            (num_channels - 1)
            * (observation_shape[1] // 2)
            * (observation_shape[2] // 2),
            fc_reward_layers,
            full_support_size,
            activation=None,
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        out = self.pool(out)
        out = out.view(
            -1,
            (self.num_channels - 1)
            * (self.observation_shape[1] // 2)
            * (self.observation_shape[2] // 2),
        )
        reward = self.fc(out)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        num_channels,
        pooling_size,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
    ):
        super(PredictionNetwork, self).__init__()
        self.num_channels = num_channels
        self.observation_shape = observation_shape
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.pool = torch.nn.AvgPool2d(pooling_size, stride=2)
        self.fc_value = FullyConnectedNetwork(
            num_channels * (observation_shape[1] // 2) * (observation_shape[2] // 2),
            fc_value_layers,
            full_support_size,
            activation=None,
        )
        self.fc_policy = FullyConnectedNetwork(
            num_channels * (observation_shape[1] // 2) * (observation_shape[2] // 2),
            fc_policy_layers,
            action_space_size,
            activation=None,
        )

    def forward(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.pool(out)
        out = out.view(
            -1,
            self.num_channels
            * (self.observation_shape[1] // 2)
            * (self.observation_shape[2] // 2),
        )
        value = self.fc_value(out)
        policy = self.fc_policy(out)
        return policy, value


class MuZeroResidualNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        pooling_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = RepresentationNetwork(
            observation_shape, stacked_observations, num_blocks, num_channels
        )

        self.dynamics_network = DynamicNetwork(
            observation_shape,
            num_blocks,
            num_channels + 1,
            pooling_size,
            fc_reward_layers,
            self.full_support_size,
        )

        self.prediction_network = PredictionNetwork(
            observation_shape,
            action_space_size,
            num_blocks,
            num_channels,
            pooling_size,
            fc_value_layers,
            fc_policy_layers,
            self.full_support_size,
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state == 0] = 1
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (action.shape[0], 1, encoded_state.shape[2], encoded_state.shape[3])
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state == 0] = 1
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
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


########### End ResNet ###########
##################################


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
