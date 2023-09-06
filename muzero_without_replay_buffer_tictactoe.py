from self_play import MCTS, GameHistory
from games.tictactoe import MuZeroConfig, Game
# from games.tictactoe import MuZeroConfig, Game
import models

import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

import math
import time
import copy

from simplifiedMuZero.without_rb.game_play import GamePlay
from simplifiedMuZero.without_rb.play_buffer import PlayBuffer
from simplifiedMuZero.without_rb.trainer import Trainer

def logging_loop(config, checkpoint, writer, training_steps):
    # writer = SummaryWriter(config.results_path)

    # print(
    #     "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
    # )

    # Save hyperparameters to TensorBoard
    hp_table = [
        f"| {key} | {value} |" for key, value in config.__dict__.items()
    ]
    writer.add_text(
        "Hyperparameters",
        "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    )
    # # Save model representation
    # writer.add_text(
    #     "Model summary",
    #     str(model).replace("\n", " \n\n") # self.summary, 换成其它的
    # )
    # Loop for updating the training performance
    counter = training_steps

    try:
        if True:
        # while checkpoint["training_step"] < config.training_steps:
            writer.add_scalar(
                "1.Total_reward/1.Total_reward",
                checkpoint["total_reward"],
                counter,
            )
            writer.add_scalar(
                "1.Total_reward/2.Mean_value",
                checkpoint["mean_value"],
                counter,
            )
            writer.add_scalar(
                "1.Total_reward/3.Episode_length",
                checkpoint["episode_length"],
                counter,
            )
            writer.add_scalar(
                "1.Total_reward/4.MuZero_reward",
                checkpoint["muzero_reward"],
                counter,
            )
            writer.add_scalar(
                "1.Total_reward/5.Opponent_reward",
                checkpoint["opponent_reward"],
                counter,
            )
            writer.add_scalar(
                "2.Workers/1.Self_played_games",
                checkpoint["num_played_games"],
                counter,
            )
            writer.add_scalar(
                "2.Workers/2.Training_steps", checkpoint["training_step"], counter
            )
            writer.add_scalar(
                "2.Workers/3.Self_played_steps", checkpoint["num_played_steps"], counter
            )
            writer.add_scalar(
                "2.Workers/4.Reanalysed_games",
                checkpoint["num_reanalysed_games"],
                counter,
            )
            writer.add_scalar(
                "2.Workers/5.Training_steps_per_self_played_step_ratio",
                checkpoint["training_step"] / max(1, checkpoint["num_played_steps"]),
                counter,
            )
            writer.add_scalar("2.Workers/6.Learning_rate", checkpoint["lr"], counter)
            writer.add_scalar(
                "3.Loss/1.Total_weighted_loss", checkpoint["total_loss"], counter
            )
            writer.add_scalar("3.Loss/Value_loss", checkpoint["value_loss"], counter)
            writer.add_scalar("3.Loss/Reward_loss", checkpoint["reward_loss"], counter)
            writer.add_scalar("3.Loss/Policy_loss", checkpoint["policy_loss"], counter)
            print(
                f'Last test reward: {checkpoint["total_reward"]:.2f}. Training step: {checkpoint["training_step"]}/{config.training_steps}. Played games: {checkpoint["num_played_games"]}. Loss: {checkpoint["total_loss"]:.2f}',
                end="\r",
            )
            counter += 1
            # time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    # if config.save_model:
    #     # Persist replay buffer to disk
    #     path = config.results_path / "replay_buffer.pkl"
    #     print(f"\n\nPersisting replay buffer games to disk at {path}")
    #     pickle.dump(
    #         {
    #             "buffer": buffer,
    #             "num_played_games": checkpoint["num_played_games"],
    #             "num_played_steps": checkpoint["num_played_steps"],
    #             "num_reanalysed_games": checkpoint["num_reanalysed_games"],
    #         },
    #         open(path, "wb"),
    #     )

def update_gameplay_checkpoint(config, checkpoint, game_history):
    checkpoint["episode_length"] = len(game_history.action_history) - 1
    checkpoint["total_reward"] = sum(game_history.reward_history)
    checkpoint["mean_value"] = numpy.mean( [value for value in game_history.root_values if value])

    if 1 < len(config.players):
        checkpoint["muzero_reward"] = sum(
                    reward
                    for i, reward in enumerate(game_history.reward_history)
                    if game_history.to_play_history[i - 1]
                    == config.muzero_player
                )
        checkpoint["opponent_reward"] = sum(
                    reward
                    for i, reward in enumerate(game_history.reward_history)
                    if game_history.to_play_history[i - 1]
                    != config.muzero_player
                )

def save_checkpoint(config, checkpoint, path=None): #将模型存储在文件中
    if not path:
        path = config.results_path / "model.checkpoint"

    torch.save(checkpoint, path)

def train(log_in_tensorboard=True):
    config = MuZeroConfig()
    config.results_path /= "muzero_without_rb"

    if log_in_tensorboard or config.save_model:
        config.results_path.mkdir(parents=True, exist_ok=True)

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

    trainer = Trainer(models.MuZeroNetwork, checkpoint, config)
    selfplay = GamePlay(trainer.model, checkpoint, Game, config, config.seed)
    buffer = {}
    play_buffer = PlayBuffer(checkpoint, buffer, config)

    step = 1 # 间隔，即每次模拟后训练多少次
    max_steps = int(config.training_steps/step)
    # max_steps = 2000

    writer = SummaryWriter(config.results_path)

    for episode in range(max_steps):
        game_id, game_history = selfplay.play_game(selfplay.config.visit_softmax_temperature_fn(0), selfplay.config.temperature_threshold, False, "self",0)

        # print(game_id)
        # print(game_history.action_history)
        # print(game_history.reward_history)
        # print(game_history.to_play_history)
        # # print(game_history.observation_history)
        # print("child visits", game_history.child_visits)
        # print(game_history.root_values) # root value指的是root节点的UCB值

        play_buffer.update_game_history(game_id, game_history)
        update_gameplay_checkpoint(config, checkpoint, game_history)

        for i in range(step):
            index_batch, batch = play_buffer.get_batch()
            # print(batch[1])
            trainer.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = trainer.update_weights(batch)


            training_step = episode * step + i
            if training_step % config.checkpoint_interval == 0:
                checkpoint["weights"] = copy.deepcopy(trainer.model.get_weights())
                checkpoint["optimizer_state"] =copy.deepcopy(models.dict_to_cpu(trainer.optimizer.state_dict()) )

                if config.save_model:
                    save_checkpoint(config, checkpoint)
            checkpoint["training_step"] = training_step
            checkpoint["lr"] = trainer.optimizer.param_groups[0]["lr"]
            checkpoint["total_loss"] = total_loss
            checkpoint["value_loss"] = value_loss
            checkpoint["reward_loss"] = reward_loss
            checkpoint["policy_loss"] = policy_loss

        # print(training_step)
        # if training_step % 500 == 0:
        # if training_step % config.checkpoint_interval == 0:
        #     # print(training_step)
        #     logging_loop(config, checkpoint, writer)

        logging_loop(config, checkpoint, writer, training_step)


    writer.close()

    selfplay.close_game()

if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print("耗时: {:.2f}秒".format(end_time - start_time))