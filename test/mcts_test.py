import models
from self_play import MCTS, GameHistory, Node, MinMaxStats
from games.tictactoe import MuZeroConfig, Game

import torch
import numpy
import math

class MCTS1:
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
        print(override_root_with)
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

config = MuZeroConfig()
game = Game(config.seed)

game_history = GameHistory()

observation = game.reset()

game_history.action_history.append(0)
game_history.observation_history.append(observation)  # 添加reset之后的observation
game_history.reward_history.append(0)
game_history.to_play_history.append(game.to_play())

stacked_observations = game_history.get_stacked_observations( -1, config.stacked_observations, len(config.action_space))

done = False

model = models.MuZeroNetwork(config)

root, mcts_info = MCTS1(config).run(model, stacked_observations, game.legal_actions(), game.to_play(), True)

print(root)

game.close()