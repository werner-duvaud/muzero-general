import math
from abc import ABC, abstractmethod

import torch

from models import support_to_scalar, scalar_to_support, mlp, AbstractNetwork, conv3x3, RepresentationNetwork, DynamicsNetwork, PredictionNetwork

class MuZeroNetwork_2net:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork_2net(
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
            print("resnet")
            return MuZeroResidualNetwork_2net(
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
class MuZeroFullyConnectedNetwork_2net(AbstractNetwork):
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
    ):
        super().__init__()
        print(self.__class__.__name__)
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        # support_size 表示的应该是一个选择的范围【-support_size, support_size】.最后+1是因为range最后不包含最后的数

        representation_input_size = observation_shape[0] * observation_shape[1] * observation_shape[2] * (
                    stacked_observations + 1) \
                                    + stacked_observations * observation_shape[1] * observation_shape[2]

        # 输出等于输入，即编码维度等于输入维度
        encoding_size = representation_input_size

        # self.representation_network = torch.nn.DataParallel(
        #     # mlp(
        #     #     representation_input_size,
        #     #     fc_representation_layers,
        #     #     encoding_size,
        #     # )
        #     mlp(
        #         representation_input_size + self.action_space_size,
        #         fc_representation_layers,
        #         encoding_size,
        #     )
        # )

        #dynamics的输入是encoding_size+action_space_size
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size) #最后的输出为full_support_size，因为范围是[-support_size, support_size]
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size) #输出action的概率
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size) #最后的输出为full_support_size，因为范围是[-support_size, support_size]
        )


    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
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
        observation = observation.view(observation.shape[0], -1)
        action_zeros = (torch.zeros((observation.shape[0], self.action_space_size)).to(observation.device).float())
        x = torch.cat((observation, action_zeros), dim=1)

        # encoded_state = self.representation_network(x)
        encoded_state = self.dynamics_encoded_state_network(x)

        # encoded_state = self.representation_network(
        #     observation.view(observation.shape[0], -1)
        # )

        return self.encoded_stated_normalized(encoded_state)


    # dynamic同representation的唯一不同就是前者需要将encoded_state和action合并在一起作为输入，而representation不需要绑定action
    def dynamics(self, encoded_state, action):
        action_one_hot = (torch.zeros((action.shape[0], self.action_space_size)).to(action.device).float())
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)
        next_encoded_state_normalized = self.encoded_stated_normalized(next_encoded_state)

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency 一致性奖励等于 0
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        # reward的样子为[[0,0,...,0,1,0,...,0,0]，...]。即中间值为1，其余全为0，然后重复于observation行数相同的次数

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

class MuZeroResidualNetwork_2net(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations, # stacken_observations表示先去观察的数量，用在那些需要历史信息的游戏里。如果不需要历史观察，该值为0
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
    ):
        super().__init__()
        num_channels = observation_shape[1]
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

        # observations_shape存放的时观察值的维度形状，第0维时观察的当前和历史维度，后面几维是观察值
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

        # self.representation_network = torch.nn.DataParallel(
        #     RepresentationNetwork(
        #         observation_shape,
        #         stacked_observations,
        #         num_blocks,
        #         num_channels,
        #         downsample,
        #     )
        # )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
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
        # print("encoded_state shape is : " , encoded_state.shape)
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    # def representation(self, observation):
    #     # print("observation shape is : ", observation.shape)
    #     encoded_state = self.representation_network(observation)
    #
    #     # Scale encoded state between [0, 1] (See appendix paper Training)
    #     min_encoded_state = (
    #         encoded_state.view(
    #             -1,
    #             encoded_state.shape[1],
    #             encoded_state.shape[2] * encoded_state.shape[3],
    #         )
    #         .min(2, keepdim=True)[0]
    #         .unsqueeze(-1)
    #     )
    #     max_encoded_state = (
    #         encoded_state.view(
    #             -1,
    #             encoded_state.shape[1],
    #             encoded_state.shape[2] * encoded_state.shape[3],
    #         )
    #         .max(2, keepdim=True)[0]
    #         .unsqueeze(-1)
    #     )
    #     scale_encoded_state = max_encoded_state - min_encoded_state
    #     scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
    #     encoded_state_normalized = (
    #         encoded_state - min_encoded_state
    #     ) / scale_encoded_state
    #     return encoded_state_normalized

    def representation(self, encoded_state):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(encoded_state.device)
            .float()
        )
        # action_one_hot = (
        #         action[:, :, None, None] * action_one_hot / self.action_space_size
        # )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, _ = self.dynamics_network(x) # 第二个参数是reward，在表示网络不需要它

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
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
                                                next_encoded_state - min_next_encoded_state
                                        ) / scale_next_encoded_state
        return next_encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
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
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        # action = torch.tensor([[0]]).to(observation.device)
        # encoded_state = self.dynamics(observation, action)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0) # 将support_size位置设为1
                .repeat(len(observation), 1) # 根据observation的长度复制，保证reward的维度于observation的一致，即之前的observation也赋值
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state
