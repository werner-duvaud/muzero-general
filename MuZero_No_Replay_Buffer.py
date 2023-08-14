import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
# import simplifiedMuZero.without_rb.models_without_replay_buffer as models
import models
# import replay_buffer
# import simplifiedMuZero.without_rb.self_play_without_replay_buffer as self_play
import shared_storage
# import simplifiedMuZero.without_rb.trainer_without_replay_buffer as trainer
from self_play import MCTS, GameHistory
from muzero import load_model_menu, CPUActor

# training_step是一个全局变量，用来存储现有的运行次数，不要超过游戏config里的training_steps，如30000次

class GamePlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.trained_steps = initial_checkpoint["training_step"]
        self.terminate = False

    def continuous_self_play(self, test_mode=False):
    # def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while self.trained_steps < self.config.training_steps and not self.terminate: # 如果当前的训练步数低于训练总步数，并且没有终止的话，继续进行训练
            # 此处不要用set——weights，因为现在移除了replay_buffer，不需要shared_storage了
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights"))) # 从shared_storage中获取当前的参数

            if not test_mode:
                # game_history = self.play_game(
                #     self.config.visit_softmax_temperature_fn(
                #         trained_steps=ray.get(
                #             shared_storage.get_info.remote("training_step")
                #         )
                #     ),
                #     self.config.temperature_threshold,
                #     False,
                #     "self",
                #     0,
                # )
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        self.trained_steps
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                # replay_buffer.save_game.remote(game_history, shared_storage)
                return game_history

            else:
                # Take the best action (no exploration) in test mode # 在测试模式下采取最佳行动（无探索）
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    # play game 与continuous self play 的区别：
    #   1. play game 是实际运行游戏，游戏的结果存在game history里，不向replay buffer里写
    #   2. continuous self play 调用play game，把获取到的game history 异步写进 replay buffer
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

        if render:
            self.game.render()

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

        return game_history

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

class Trainer_without_Replay_Buffer:
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
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
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

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    # update weights 与 continuous update weights 的区别
    #   1. update weights 是实际计算更新network的权重
    #   2. continuous update weights 从replay buffer 里获取数据batch， 并将batch传递给update weights，使update weights完成参数更新
    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

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
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
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
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 此处才算一次迭代完成，training step加1
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

class MuZero_No_Replay_Buffer:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        config (dict, MuZeroConfig, optional): Override the default config of the game.

        split_resources_in (int, optional): Split the GPU usage when using concurent muzero instances.

    Example:
        >>> muzero = MuZero_No_Replay_Buffer("cartpole")
        >>> muzero.train()
        >>> muzero.test(render=True)
    """

    def __init__(self, game_name, config=None, split_resources_in=1):
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            print("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err

        # Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Manage GPUs
        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)

        ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
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
        self.replay_buffer = {}

        # cpu_actor = CPUActor.remote()
        # cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        # 移除ray
        cpu_actor = CPUActor()
        cpu_weights = cpu_actor.get_initial_weights(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        # self.training_worker = trainer.Trainer.options(
        #     num_cpus=0,
        #     num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        # ).remote(self.checkpoint, self.config)
        #
        # self.shared_storage_worker = shared_storage.SharedStorage.remote(
        #     self.checkpoint,
        #     self.config,
        # )
        # self.shared_storage_worker.set_info.remote("terminate", False)
        #
        # self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        #     self.checkpoint, self.replay_buffer, self.config
        # )

        # 初始化权重
        self.training_worker = Trainer_without_Replay_Buffer(self.checkpoint, self.config)

        # #使用最后一个模型提供更新鲜、稳定的n步值（参见论文附录Reanalyze）
        # if self.config.use_last_model_value:
        #     self.reanalyse_worker = replay_buffer.Reanalyse.options(
        #         num_cpus=0,
        #         num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
        #     ).remote(self.checkpoint, self.config)
        #
        # self.self_play_workers = [
        #     self_play.SelfPlay.options(
        #         num_cpus=0,
        #         num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
        #     ).remote(
        #         self.checkpoint,
        #         self.Game,
        #         self.config,
        #         self.config.seed + seed,
        #     )
        #     for seed in range(self.config.num_workers)
        # ]
        #
        # # 这里调用continuous类的函数，主要是continuous函数会调用replay_buffer，
        #
        # # Launch workers
        # # 此处调用worker进行self play，把结果存在replay_buffer里
        # [
        #     self_play_worker.continuous_self_play.remote(
        #         self.shared_storage_worker, self.replay_buffer_worker
        #     )
        #     for self_play_worker in self.self_play_workers
        # ]

        # # 此处使用trainer，从replay buffer里按batch抽取数据，进行网络训练和更新
        # self.training_worker.continuous_update_weights.remote(
        #     self.replay_buffer_worker, self.shared_storage_worker
        # )
        self.training_worker.continuous_update_weights(self.replay_buffer_worker, self.shared_storage_worker)

        # # 使用最后一个模型提供更新鲜、稳定的n步值（参见论文附录Reanalyze）
        # if self.config.use_last_model_value:
        #     self.reanalyse_worker.reanalyse.remote(
        #         self.replay_buffer_worker, self.shared_storage_worker
        #     )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )

    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.Game,
            self.config,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            self.summary,
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))
                writer.add_scalar(
                    "1.Total_reward/1.Total_reward",
                    info["total_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/2.Mean_value",
                    info["mean_value"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/3.Episode_length",
                    info["episode_length"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/4.MuZero_reward",
                    info["muzero_reward"],
                    counter,
                )
                writer.add_scalar(
                    "1.Total_reward/5.Opponent_reward",
                    info["opponent_reward"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/1.Self_played_games",
                    info["num_played_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/2.Training_steps", info["training_step"], counter
                )
                writer.add_scalar(
                    "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
                )
                writer.add_scalar(
                    "2.Workers/4.Reanalysed_games",
                    info["num_reanalysed_games"],
                    counter,
                )
                writer.add_scalar(
                    "2.Workers/5.Training_steps_per_self_played_step_ratio",
                    info["training_step"] / max(1, info["num_played_steps"]),
                    counter,
                )
                writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
                writer.add_scalar(
                    "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
                )
                writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
                writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
                writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
                print(
                    f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            path = self.config.results_path / "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer games to disk at {path}")
            # 此处是将replay buffer的结果写入文件保持
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(path, "wb"),
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        render,
                        opponent,
                        muzero_player,
                    )
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            # pickle用来存储和导入文件，其作用是将对象转换为字符串或者将字符串转换为对象
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            # 此处更新replay buffer的值
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        game = self.Game(self.config.seed)
        obs = game.reset()
        dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
        dm.compare_virtual_with_real_trajectories(obs, game, horizon)
        input("Press enter to close all plots")
        dm.close_all()


# @ray.remote(num_cpus=0, num_gpus=0)
# class CPUActor:
#     # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
#     def __init__(self):
#         pass
#
#     def get_initial_weights(self, config):
#         model = models.MuZeroNetwork(config)
#         weigths = model.get_weights()
#         summary = str(model).replace("\n", " \n\n")
#         return weigths, summary


def hyperparameter_search(
    game_name, parametrization, budget, parallel_experiments, num_tests
):
    """
    Search for hyperparameters by launching parallel experiments.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

        parametrization : Nevergrad parametrization, please refer to nevergrad documentation.

        budget (int): Number of experiments to launch in total.

        parallel_experiments (int): Number of experiments to launch in parallel.

        num_tests (int): Number of games to average for evaluating an experiment.
    """
    optimizer = nevergrad.optimizers.OnePlusOne(
        parametrization=parametrization, budget=budget
    )

    running_experiments = []
    best_training = None
    try:
        # Launch initial experiments
        for i in range(parallel_experiments):
            if 0 < budget:
                param = optimizer.ask()
                print(f"Launching new experiment: {param.value}")
                muzero = MuZero_No_Replay_Buffer(game_name, param.value, parallel_experiments)
                muzero.param = param
                muzero.train(False)
                running_experiments.append(muzero)
                budget -= 1

        while 0 < budget or any(running_experiments):
            for i, experiment in enumerate(running_experiments):
                if experiment and experiment.config.training_steps <= ray.get(
                    experiment.shared_storage_worker.get_info.remote("training_step")
                ):
                    experiment.terminate_workers()
                    result = experiment.test(False, num_tests=num_tests)
                    if not best_training or best_training["result"] < result:
                        best_training = {
                            "result": result,
                            "config": experiment.config,
                            "checkpoint": experiment.checkpoint,
                        }
                    print(f"Parameters: {experiment.param.value}")
                    print(f"Result: {result}")
                    optimizer.tell(experiment.param, -result)

                    if 0 < budget:
                        param = optimizer.ask()
                        print(f"Launching new experiment: {param.value}")
                        muzero = MuZero_No_Replay_Buffer(game_name, param.value, parallel_experiments)
                        muzero.param = param
                        muzero.train(False)
                        running_experiments[i] = muzero
                        budget -= 1
                    else:
                        running_experiments[i] = None

    except KeyboardInterrupt:
        for experiment in running_experiments:
            if isinstance(experiment, MuZero_No_Replay_Buffer):
                experiment.terminate_workers()

    recommendation = optimizer.provide_recommendation()
    print("Best hyperparameters:")
    print(recommendation.value)
    if best_training:
        # Save best training weights (but it's not the recommended weights)
        best_training["config"].results_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            best_training["checkpoint"],
            best_training["config"].results_path / "model.checkpoint",
        )
        # Save the recommended hyperparameters
        text_file = open(
            best_training["config"].results_path / "best_parameters.txt",
            "w",
        )
        text_file.write(str(recommendation.value))
        text_file.close()
    return recommendation.value


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py cartpole
        muzero = MuZero_No_Replay_Buffer(sys.argv[1])
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        muzero = MuZero_No_Replay_Buffer(sys.argv[1], config)
        muzero.train()
    else:
        print("\nWelcome to MuZero! Here's a list of games:")
        # Let user pick a game
        games = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
            if filename.name != "abstract_game.py"
        ]
        for i in range(len(games)):
            print(f"{i}. {games[i]}")
        choice = input("Enter a number to choose the game: ")
        valid_inputs = [str(i) for i in range(len(games))]
        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")

        # Initialize MuZero
        choice = int(choice)
        game_name = games[choice]
        muzero = MuZero_No_Replay_Buffer(game_name)

        while True:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            print()
            for i in range(len(options)):
                print(f"{i}. {options[i]}")

            choice = input("Enter a number to choose an action: ")
            valid_inputs = [str(i) for i in range(len(options))]
            while choice not in valid_inputs:
                choice = input("Invalid input, enter a number listed above: ")
            choice = int(choice)
            if choice == 0:
                start_time = time.time()
                muzero.train()
                end_time = time.time()
                print("耗时: {:.2f}秒".format(end_time - start_time))
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZero_No_Replay_Buffer(game_name, best_hyperparameters)
            else:
                break
            print("\nDone")

    ray.shutdown()
