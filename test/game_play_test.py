from self_play import MCTS, GameHistory
from games.simple_grid import MuZeroConfig, Game
# from games.tictactoe import MuZeroConfig, Game
import models

import numpy
import torch

import math
import time
import copy

class MySelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, model, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        # self.model = models.MuZeroNetwork(self.config)
        # self.model.set_weights(initial_checkpoint["weights"])
        self.model = model
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.trained_steps = initial_checkpoint["training_step"]
        self.terminate = False

    #play game 运行
    # 合法的actions是固定的，由游戏文件提供(在本函数中，可以看到调用legal_actions函数没有使用env，这表面现游戏环境于的改变于动作无关)。
    # 运行步骤：
    #   1. 创建GameHistory用来存储数据
    #   2. 检查游戏是否结束或者到底最大移动次数
    #   3. 获取stacked observation（因为有些游戏需要考虑之前的历史数据和移动轨迹)
    #   4. 运行MCTS搜索下一步的action
    #   5. 调用游戏函数step（action），获取下一步action之后的observation、reward和done
    #   6. 持续运行2-5步直到结束
    #   7. 返回GameHistory
    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation) # 添加reset之后的observation
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False
        game_id = None

        if render:
            self.game.render()

        game_id = self.game.to_play()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ): # 游戏没有结束且运行步数小于最大移动步长
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )
                # index是-1，game_history 会在创建时添加reset的observation，因此其长度为1.index取模（%）之后时1
                # config.stacked_observationis是存储之前的observation的数量，如果不要之前的信息，可以设为0，这样就不会存储之前的信息

                # 一下的if-else部分主要是为了选择一个动作
                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(), # to_play返回当期玩游戏的玩家ID，默认是0
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    ) # 根据temperature选择动作

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action( #选择对手动作，分为随机，human和expert三种
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action) # 运行游戏

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation) #添加到observation的队列。取数据是使用stacked_observation函数，从后往前取
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_id, game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    # 根据访问次数分布和温度选择操作。 温度通过配置中的visit_softmax_Temperature函数动态改变。
    # 公式为 c^(1/t)。可以看到：
    #   t越小，1/t于接近于无穷大，值大的c就越容易被选中。
    #   t越大,1/t->0。c^0=1。则所有的访问次数变为相同的1，难以区分大小，因此就会相当于随机选择
    #   特殊地，当t=0时，使用random完全随机选择，当t=+∞,使用argmax选择最大的
    @staticmethod # 静态方法修饰符，类似于static关键字
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action

class PlayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer) # initial_buffer默认为{}
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

    def save_game(self, game_history):
        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = None

        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=True): #将force_uniform 设置为True，强制安装平均分布选取
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None

        game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games):
        selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
        game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None

        position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        # if next(iter(self.buffer)) <= game_id:
        #     self.buffer[game_id] = game_history

        self.buffer[game_id] = game_history

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
                if game_history.to_play_history[bootstrap_index]
                == game_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )

            value = last_step_value * self.config.discount**self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                reward
                if game_history.to_play_history[index]
                == game_history.to_play_history[index + i]
                else -reward
            ) * self.config.discount**i

        return value

    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(game_history.reward_history[current_index])
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions

class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        # self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        # if initial_checkpoint["optimizer_state"] is not None:
        #     print("Loading optimizer...\n")
        #     self.optimizer.load_state_dict(
        #         copy.deepcopy(initial_checkpoint["optimizer_state"])
        #     )

    # # update weights 与 continuous update weights 的区别
    # #   1. update weights 是实际计算更新network的权重
    # #   2. continuous update weights 从replay buffer 里获取数据batch， 并将batch传递给update weights，使update weights完成参数更新
    # def continuous_update_weights(self, play_buffer, terminate): # terminate是用来记录replay buffer等其它程序是否终止的，跟game的状态无关
    #     next_batch = play_buffer.get_batch()
    #     # Training loop
    #     while self.training_step < self.config.training_steps and not terminate:
    #         index_batch, batch = next_batch
    #         next_batch = play_buffer.get_batch()
    #         self.update_lr()
    #         (
    #             priorities,
    #             total_loss,
    #             value_loss,
    #             reward_loss,
    #             policy_loss,
    #         ) = self.update_weights(batch)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        observation_batch = (
            torch.tensor(numpy.array(observation_batch)).float().to(device)
        )
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, i]
            )
            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            models.support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.PER_alpha
        )

        for i in range(1, len(predictions)):
            value, reward, policy_logits = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                models.support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                numpy.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss

        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss

if __name__ == "__main__":
    config = MuZeroConfig()

    checkpoint = {
        "weights": None,
        "optimizer_state": None,
        "total_reward": 0,
        "muzero_reward": 0,
        "opponent_reward": 0,
        "episode_length": 0,
        "mean_value": 0,
        "training_step": 0,
        "lr": 0,
        "total_loss": 0,
        "value_loss": 0,
        "reward_loss": 0,
        "policy_loss": 0,
        "num_played_games": 0,
        "num_played_steps": 0,
        "num_reanalysed_games": 0,
        "terminate": False,
    }

    trainer = Trainer(checkpoint, config)
    selfplay = MySelfPlay(trainer.model, checkpoint, Game, config, config.seed)
    buffer = {}
    play_buffer = PlayBuffer(checkpoint, buffer, config)
    for i in range(config.training_steps):
        game_id, game_history = selfplay.play_game(selfplay.config.visit_softmax_temperature_fn(0), selfplay.config.temperature_threshold, False, "self",0)

        # print(game_id)
        # print(game_history.action_history)
        # print(game_history.reward_history)
        # print(game_history.to_play_history)
        # # print(game_history.observation_history)
        # print("child visits", game_history.child_visits)
        # print(game_history.root_values) # root value指的是root节点的UCB值

        # buffer[game_id] = game_history

        play_buffer.update_game_history(game_id, game_history)

        for i in range(10):
            index_batch, batch = play_buffer.get_batch()
            # print(batch[1])
            trainer.update_lr()
            trainer.update_weights(batch)

    selfplay.close_game()


