import numpy
import torch
from self_play import GameHistory, MCTS
class GamePlay:
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