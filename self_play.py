import math
import time

import numpy
import ray
import torch

import models


@ray.remote
class SelfPlay:
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

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ): # 如果当前的训练步数低于训练总步数，并且没有终止的话，继续进行训练
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights"))) # 从shared_storage中获取当前的参数

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
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
        game_history.to_play_history.append(self.game.to_play()) # to_play_history是用来存放玩家id的

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


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    # run函数运行流程：
    #   1. 获取root节点
    #       (1)如果由指定节点这将root赋值为该节点；
    #       (2)如果没有，则
    #           i. 创建新的节点Node(0)
    #           ii. 使用initial_inference函数通过observation获取相应的reward，hidden state，legal actions等数据
    #           iii. 将ii中获取的数据赋值到创建的root节点中取
    #           PS. 可以看到，在（1）的情况下不需要调用initial_inference函数
    #   2. 检查是否需要添加探索噪音
    #   3. 开始循环模拟游戏，模拟的次数由num simulation决定
    #       （1） 将初始节点node设置为root，并将节点node加入search tree中
    #       （2） 检查该节点是否已经扩展，如果已经扩展，则通过ucb值来选择子节点expand. 并将node 设置为选中的节点。并将节点node加入search tree中
    #       （3） 重复2，直到找到expanded为false的node为止
    #       （4） 选择search_tree[-2]为parent(因为最后一个是node)
    #       （5） 运行recurrent_inference函数，获得reward，hidden state，legal actions等数据
    #       （6） 扩展node,即为node创建子节点，使node展开。
    #       （7） 反向传播算法，对路径上的所有访问次数+1，value值加reward
    #       PS: 可以看到，通过不停的模拟，节点被一层层的扩展（每次模拟扩展一个节点）。
    #   4. 返回扩展过后的节点树root，以便之后的程序根据它选择动作action
    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with: #检查有没有提供Node,如果有，则指定；如果没有，则自己创建一个
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            ) # observation转tensor，外面包一层形成一个batch。 Observation的长度由参数stacked_observation配置，主要存储之前的previous。不要之前privious的配置为0
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations): # 开始模拟游戏
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            # expanded根据node的子节点个数判断是否已经扩展了，如果没有子节点，说明没被扩展
            while node.expanded(): #这个循环一直在搜索没有expand的子节点。如果子节点已经expand了，则通过select_child选择下一个
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats) #选取ucb最大的一个action，如果有多个action得分相同，随机选取一个
                search_path.append(node) #把节点添加到搜索队列

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # 在搜索树内部，我们使用动态函数来获取给定动作的下一个hidden_state和previous hidden state
            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2] # 选择倒数第二个节点，因为当前的node是-1，则-2是它的parent
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            # expand一层节点，actions是动作列表，policy_logits是rewards列表
            # 通过该函数，在该节点扩展一层节点
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    # MCTS 的select child和之前SelfPlay的select action逻辑是不一样的
    #   1. select child是根据UCB选取的，select action是根据各个动作的visit count和temperature选取的
    #   2. select child 选择的对象是Node,Node是由当前的state执行action后生成的新Node形成的。select action单纯的是选action
    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice( # 随机选择ucb值等于最大ucb的动作（因为可能有多个动作的值都达到了最大的ucb,如果只有一个，那么就会选取这个)
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats): #该函数只进行一步查询，不进行多步
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base # pc_c_base由配置文件决定
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior # prior 之前的p_value
        # 公式 pb_c = (log((N+C+1)/C)+init ) * sqrt(N/(VC+1))
        # prior_score = pbc * prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize( # 括号里的是Q值，Q=E[r+r*Q'。此处在对其进行正则化
                child.reward
                + self.config.discount # 衰减系数， 之后乘以子节点的值
                * (child.value() if len(self.config.players) == 1 else -child.value()) # 根据players的个数，如果大于1，则子节点必定是对手，因此子节点的取负。
            )
        else:
            value_score = 0

        return prior_score + value_score # 先前的分数加上Q值就是新的UCB值

    # 反向传播算法
    # 对路径上的所有访问次数+1，value值加reward
    def backpropagate(self, search_path, value, to_play, min_max_stats): # MCTS反向传播，visit count加1
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0 #visit count默认是0，只有经过反向传播之后才能变成增加
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        # expand一层节点，actions是动作列表，policy_logits是rewards列表
        # 通过该函数，在该节点扩展一层节点
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)} # 列出所有的合法动作及对于的value值
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ): #根据索引index获取observation序列
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy() #分为两部分，一部分是当前（current）观察值，一部分是之前的(previous)观察值
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate( # np.concatenate将第一个参数的list组合起来，方法是依次拆开每个元素，拼接
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate( # 向stoacked_observtions添加内容
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf") # 最大是-∞
        self.minimum = float("inf") # 最小是+∞
        # 跟类一定要update至少两次才能产生正确的范围。第一次更新掉max<min的情况，但是max=min;之后更新使其产生范围

    def update(self, value): # 更新max和min,方法时对比大小，大的更新为上限，小的更新为下限
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value): #对value规范化，公式为(x-a)/(a-b) 当x∈[a,b]时
        if self.maximum > self.minimum: # 如果最大大于最小，说明至少更新了两次（第一次更新掉max<min的情况，但是max=min;之后更新使其产生范围）
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value # 如果范围没有更新，就直接返回value
