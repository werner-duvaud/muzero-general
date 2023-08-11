import math
from abc import ABC, abstractmethod

import torch

from models import *

class SimplifiedMuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return SimplifiedMuZeroFullyConnectedNetwork(
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
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )
class SimplifiedMuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(self,
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
                 ):
        super().__init__()
        # 动作空间大小
        self.action_space_size = action_space_size
        #为什么是2*support_size +1
        self.full_support_size = 2 * support_size + 1
        representation_input_size = observation_shape[0] * observation_shape[1] * observation_shape[2] * (stacked_observations + 1)\
                                    + stacked_observations * observation_shape[1] * observation_shape[2]

        # 改进方法：
        #   1. input size = encoding _size
        #   2. input 后边加上 action space
        self.representation_network = torch.nn.DataParallel(
            mlp(
                representation_input_size,
                fc_representation_layers,
                encoding_size
            )
        )

        self.dynamic_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size +self.action_space_size,
                fc_dynamics_layers,
                encoding_size
            )
        )

        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_polic_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )

        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encode_state):
        policy_logits = self.prediction_polic_network(encode_state)
        value = self.prediction_value_network(encode_state)
        return policy_logits, value

    # 将encoded_stated标准化
    def encoded_stated_normalized(self, encoded_state):
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5  # 防止为0，出现NAN
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state

        return encoded_state_normalized

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )

        return self.encoded_stated_normalized(encoded_state)

    # dynamic同representation的唯一不同就是前者需要将encoded_state和action合并在一起作为输入，而representation不需要绑定action
    def dynamics(self, encoded_state, action):
        action_one_hot = (torch.zeros((action.shape[0], self.action_space_size)).to(action.device).float())
        action_one_hot.scatter(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamic_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        next_encoded_state_normalized = self.encoded_stated_normalized(next_encoded_state)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)

        # reward的样子为[[0,0,...,0,1,0,...,0,0]，...]。即中间值为1，其余全为0，然后重复于observation行数相同的次数
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (value, reward, policy_logits, encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state
