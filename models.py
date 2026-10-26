import math
from abc import ABC, abstractmethod

import torch

_SUPPORTS = {}


class DataParallel(torch.nn.DataParallel):
    """Keep CPU workers usable on hosts with multiple visible GPUs."""

    def forward(self, *inputs, **kwargs):
        if next(self.module.parameters()).device.type == "cpu":
            return self.module(*inputs, **kwargs)
        return super().forward(*inputs, **kwargs)


def parallelize(module):
    """Wrap a module in CPU-compatible DataParallel when multiple GPUs are visible."""
    return DataParallel(module) if torch.cuda.device_count() > 1 else module


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
                config.train_reward,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
                config.train_reward,
                getattr(config, "resnet_state_normalization", "channel"),
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    """Copy a dictionary, moving tensor values and tensors in nested dicts to CPU.

    Other value types, including lists and tuples, are left unchanged.
    """
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


def scale_gradient(tensor, scale):
    """Return the same tensor value while multiplying its backward gradient."""
    return tensor * scale + tensor.detach() * (1 - scale)


def zero_support_logits(batch_size, full_support_size, device):
    """Return logits concentrated on the central, zero-valued support category."""
    logits = torch.full((batch_size, full_support_size), -1e9, device=device)
    logits[:, full_support_size // 2] = 0.0
    return logits


class AbstractNetwork(ABC, torch.nn.Module):
    """Base class for MuZero networks.

    Inference returns value logits, reward logits, policy logits, and latent
    state. Initial inference encodes a zero reward.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        """Return value, zero reward, policy logits, and state for observations."""
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        """Return value, reward, policy logits, and next state after actions."""
        pass

    def get_weights(self):
        """Return CPU state-dict tensors with DataParallel prefixes removed."""
        return {
            key.replace(".module.", "."): value.cpu()
            for key, value in self.state_dict().items()
        }

    def set_weights(self, weights):
        """Load canonical weights into wrapped or unwrapped network modules."""
        self.load_state_dict(
            {key: weights[key.replace(".module.", ".")] for key in self.state_dict()}
        )


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
        train_reward,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.train_reward = train_reward

        self.representation_network = parallelize(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = parallelize(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        if train_reward:
            self.dynamics_reward_network = parallelize(
                mlp(encoding_size, fc_reward_layers, self.full_support_size)
            )

        self.prediction_policy_network = parallelize(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = parallelize(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        """Return policy logits and value logits for a batch of latent states."""
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        """Encode stacked observations and normalize their latent states."""
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training).
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        """Return normalized next states and reward logits for the supplied actions."""
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture).
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = (
            self.dynamics_reward_network(next_encoded_state)
            if self.train_reward
            else zero_support_logits(len(action), self.full_support_size, action.device)
        )

        # Scale encoded state between [0, 1] (See paper appendix Training).
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        """Return value, zero reward, policy logits, and state for observations."""
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency.
        reward = zero_support_logits(
            len(observation), self.full_support_size, observation.device
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        """Return value, reward, policy logits, and next state after actions."""
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    """Construct a bias-free 3-by-3 convolution with one-cell padding."""
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture).
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        action_space_size,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        train_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels + action_space_size, num_channels)
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.full_support_size = full_support_size
        self.train_reward = train_reward
        if train_reward:
            self.conv1x1_reward = torch.nn.Conv2d(
                num_channels, reduced_channels_reward, 1
            )
            self.block_output_size_reward = block_output_size_reward
            self.fc = mlp(
                self.block_output_size_reward,
                fc_reward_layers,
                full_support_size,
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        if self.train_reward:
            x = self.conv1x1_reward(x)
            x = x.view(-1, self.block_output_size_reward)
            reward = self.fc(x)
        else:
            reward = zero_support_logits(len(x), self.full_support_size, x.device)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
        train_reward,
        state_normalization="channel",
    ):
        super().__init__()
        self.state_normalization = state_normalization
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = parallelize(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = parallelize(
            DynamicsNetwork(
                num_blocks,
                num_channels,
                action_space_size,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
                train_reward,
            )
        )

        self.prediction_network = parallelize(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state):
        """Return policy logits and value logits for a batch of latent states."""
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        """Encode stacked observations and normalize their latent states."""
        encoded_state = self.representation_network(observation)

        return self.normalize_state(encoded_state)

    def dynamics(self, encoded_state, action):
        """Return normalized next states and reward logits for the supplied actions."""
        # Tile a one-hot action vector into planes (See paper appendix Network Architecture).
        action_one_hot = (
            torch.zeros(
                (
                    encoded_state.shape[0],
                    self.action_space_size,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(
            1,
            action.long()[:, :, None, None].expand(
                -1, -1, encoded_state.shape[2], encoded_state.shape[3]
            ),
            1.0,
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        return self.normalize_state(next_encoded_state), reward

    def normalize_state(self, state):
        """Min-max normalize latent states using global or per-channel bounds.

        Global bounds span all features of each sample; channel bounds span each
        channel's spatial positions. Stabilize ranges smaller than 1e-5.
        """
        # Scale encoded state between [0, 1] (See paper appendix Training).
        # The channel mode preserves inference for checkpoints trained before this option.
        dimensions = 2 if self.state_normalization == "channel" else 1
        flattened = state.flatten(dimensions)
        minimum = flattened.min(-1, keepdim=True)[0]
        maximum = flattened.max(-1, keepdim=True)[0]
        shape = (*state.shape[:dimensions], *((1,) * (state.ndim - dimensions)))
        minimum, maximum = minimum.reshape(shape), maximum.reshape(shape)
        scale = maximum - minimum
        scale[scale < 1e-5] += 1e-5
        return (state - minimum) / scale

    def initial_inference(self, observation):
        """Return value, zero reward, policy logits, and state for observations."""
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency.
        reward = zero_support_logits(
            len(observation), self.full_support_size, observation.device
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        """Return value, reward, policy logits, and next state after actions."""
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    """Build linear layers with configurable hidden and output activations."""
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """Decode categorical logits and invert MuZero's signed square-root transform.

    See paper appendix Network Architecture.

    Args:
        logits (torch.Tensor): Shape (batch, 2 * support_size + 1).
        support_size (int): Radius of the integer support centered at zero.

    Returns:
        torch.Tensor: Decoded scalars with shape (batch, 1).
    """
    # Decode to a scalar.
    probabilities = torch.softmax(logits, dim=1)
    key = (support_size, probabilities.device, probabilities.dtype)
    support = _SUPPORTS.get(key)
    if support is None:
        with torch.inference_mode(False):
            support = torch.arange(
                -support_size,
                support_size + 1,
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
        _SUPPORTS[key] = support
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593).
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """Transform scalars and interpolate onto MuZero's categorical support.

    Apply the signed square-root transform with a linear term, clip to the
    support, and assign probability to adjacent integer bins.
    See paper appendix Network Architecture.

    Args:
        x (torch.Tensor): Scalars with shape (batch, prediction_steps).
        support_size (int): Radius of the integer support centered at zero.

    Returns:
        torch.Tensor: Probabilities with shape
        (batch, prediction_steps, 2 * support_size + 1).
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593).
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector.
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = x.new_zeros(x.shape[0], x.shape[1], 2 * support_size + 1)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
