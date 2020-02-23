import time

import numpy
import ray
import torch

import models


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_weights, config):
        self.config = config
        self.training_step = 0

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_weights)
        self.model.to(torch.device(config.training_device))
        self.model.train()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            # momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def continuous_update_weights(self, replay_buffer, shared_storage_worker):
        # Wait for the replay buffer to be filled
        while ray.get(replay_buffer.get_self_play_count.remote()) < 1:
            time.sleep(0.1)

        # Training loop
        while True:
            batch = ray.get(replay_buffer.get_batch.remote())
            total_loss, value_loss, reward_loss, policy_loss = self.update_weights(
                batch
            )

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage_worker.set_weights.remote(self.model.get_weights())
            shared_storage_worker.set_infos.remote("training_step", self.training_step)
            shared_storage_worker.set_infos.remote("total_loss", total_loss)
            shared_storage_worker.set_infos.remote("value_loss", value_loss)
            shared_storage_worker.set_infos.remote("reward_loss", reward_loss)
            shared_storage_worker.set_infos.remote("policy_loss", policy_loss)

            if self.config.training_delay:
                time.sleep(self.config.training_delay)

    def update_weights(self, batch):
        """
        Perform one training step.
        """
        self.update_lr()

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
        ) = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).float().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)

        target_value = self.scalar_to_support(target_value, self.config.support_size)
        target_reward = self.scalar_to_support(target_reward, self.config.support_size)

        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]
        for action_i in range(self.config.num_unroll_steps):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, action_i]
            )
            predictions.append((value, reward, policy_logits))

        # Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        for i, prediction in enumerate(predictions):
            value, reward, policy_logits = prediction
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
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        loss = (value_loss + reward_loss + policy_loss).mean()

        # Scale gradient by number of unroll steps (See paper Training appendix)
        loss.register_hook(lambda grad: grad / self.config.num_unroll_steps)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
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
    def scalar_to_support(x, support_size):
        """
        Transform a scalar to a categorical representation with (2 * support_size + 1) categories
        See paper appendix Network Architecture
        """
        # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
        x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

        # Encode on a vector
        x = torch.clamp(x, -support_size, support_size)
        floor = x.floor()
        prob = x - floor
        logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
        logits.scatter_(
            2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
        )
        indexes = floor + support_size + 1
        prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
        indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
        logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
        return logits

    @staticmethod
    def loss_function(
        value, reward, policy_logits, target_value, target_reward, target_policy
    ):
        # Cross-entropy had a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1).mean()
        reward_loss = (
            (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1).mean()
        )
        policy_loss = (
            (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1).mean()
        )
        return value_loss, reward_loss, policy_loss
