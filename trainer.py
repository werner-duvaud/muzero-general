import copy
import time

import numpy
import ray
import torch

import models


@ray.remote
class Trainer:
    """Ray actor optimizing the joint MuZero loss and publishing checkpoint state."""

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed.
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.set_num_threads(self.config.num_threads)

        # Initialize the network.
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]
        self.training_step_offset = initial_checkpoint.get("training_step_offset", 0)
        self.played_steps_offset = initial_checkpoint.get("played_steps_offset", 0)

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer.
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

    def continuous_update_weights(self, replay_buffer, shared_storage):
        """Train from replay until the update budget or a stop request is reached.

        Wait for initial data, prefetch batches, update priorities when enabled, and
        pace updates using the configured ratio and delay.
        """
        # Wait for the replay buffer to be filled.
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            status = ray.get(
                shared_storage.get_info.remote(["training_step", "terminate"])
            )
            if (
                status["training_step"] >= self.config.training_steps
                or status["terminate"]
            ):
                return
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
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933).
                replay_buffer.update_priorities.remote(priorities, index_batch)

            self.publish_training_info(
                shared_storage, (total_loss, value_loss, reward_loss, policy_loss)
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    max(0, self.training_step - self.training_step_offset)
                    / max(
                        1,
                        ray.get(shared_storage.get_info.remote("num_played_steps"))
                        - self.played_steps_offset,
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def warmup(self, replay_buffer, shared_storage, target_step):
        """Fit original replay targets before self-play and reanalysis begin.

        Use the ordinary joint loss without training/self-play ratio waits, stopping
        at target_step or a termination request.
        """
        while self.training_step < target_step:
            if ray.get(shared_storage.get_info.remote("terminate")):
                return
            indices, batch = ray.get(replay_buffer.get_batch.remote(False))
            self.update_lr()
            priorities, *losses = self.update_weights(batch)
            if self.config.PER:
                ray.get(replay_buffer.update_priorities.remote(priorities, indices))
            publish = self.publish_training_info(
                shared_storage,
                losses,
                force_checkpoint=self.training_step == target_step,
            )
            if self.training_step == target_step:
                ray.get(publish)
            if self.training_step % 1000 == 0 or self.training_step == target_step:
                print(
                    f"Warmup {self.training_step}/{target_step}: "
                    f"loss={losses[0]:.3f}, policy_loss={losses[3]:.3f}",
                    flush=True,
                )

    def get_checkpoint(self):
        """Return a consistent snapshot after preceding actor tasks have finished."""
        return {
            "training_step": self.training_step,
            "weights_version": self.training_step,
            "weights": copy.deepcopy(self.model.get_weights()),
            "optimizer_state": copy.deepcopy(
                models.dict_to_cpu(self.optimizer.state_dict())
            ),
        }

    def publish_checkpoint(self, shared_storage):
        """Flush a snapshot after this actor's preceding storage updates."""
        ray.get(shared_storage.set_info.remote(self.get_checkpoint()))

    def publish_training_info(self, shared_storage, losses, force_checkpoint=False):
        """Publish metrics and periodically synchronize weights and optimizer state.

        Return the shared-storage update reference. force_checkpoint also triggers a
        weight/optimizer snapshot and disk save when save_model is enabled.
        """
        save_checkpoint = self.config.save_model and (
            force_checkpoint
            or self.training_step % 1000 == 0
            or self.training_step == self.config.training_steps
        )
        checkpoint = (
            save_checkpoint
            or force_checkpoint
            or self.training_step % self.config.checkpoint_interval == 0
            or self.training_step == self.config.training_steps
        )
        total_loss, value_loss, reward_loss, policy_loss = losses
        training_info = {
            "training_step": self.training_step,
            "lr": self.optimizer.param_groups[0]["lr"],
            "total_loss": total_loss,
            "value_loss": value_loss,
            "reward_loss": reward_loss,
            "policy_loss": policy_loss,
        }
        if checkpoint:
            training_info.update(self.get_checkpoint())
        published = shared_storage.set_info.remote(training_info)
        if save_checkpoint:
            shared_storage.save_checkpoint.remote()
        return published

    def unroll_model(self, observation_batch, action_batch):
        """Predict the root and recurrent steps for a training batch.

        For residual models, pool prediction-head normalization statistics across
        unroll positions. Apply recurrent gradient scaling.

        Returns:
            list[tuple]: Value, reward, and policy logits for the root and each
            recurrent prediction, in temporal order.
        """
        if self.config.network == "resnet":
            # Normalize the shared prediction head over the same mixture of
            # root and recurrent states during training and inference.
            hidden_state = self.model.representation(observation_batch)
            states = [hidden_state]
            rewards = [
                models.zero_support_logits(
                    len(observation_batch),
                    self.model.full_support_size,
                    observation_batch.device,
                )
            ]
            for i in range(action_batch.shape[1]):
                next_state, reward = self.model.dynamics(
                    hidden_state, action_batch[:, i]
                )
                states.append(next_state)
                rewards.append(reward)
                # Own-head gradients use the unscaled state; only later
                # dynamics receive the paper's factor of one half (See paper appendix Training).
                hidden_state = models.scale_gradient(next_state, 0.5)
            # Keep a sample's root and unroll states together when scattering
            # the prediction batch across GPUs.
            inputs = torch.stack(states, dim=1).flatten(0, 1)
            policy_logits, value = self.model.prediction(inputs)
            predictions = list(
                zip(
                    value.reshape(len(observation_batch), len(states), -1).unbind(1),
                    rewards,
                    policy_logits.reshape(
                        len(observation_batch), len(states), -1
                    ).unbind(1),
                )
            )
        else:
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch
            )
            predictions = [(value, reward, policy_logits)]
            for i in range(action_batch.shape[1]):
                value, reward, policy_logits, hidden_state = (
                    self.model.recurrent_inference(hidden_state, action_batch[:, i])
                )
                # Scale the gradient at the start of the dynamics function (See paper appendix Training).
                hidden_state = models.scale_gradient(hidden_state, 0.5)
                predictions.append((value, reward, policy_logits))
        return predictions

    def update_weights(self, batch):
        """Apply one joint-loss optimizer update and increment the training step.

        Returns:
            tuple: Replay priorities, total loss, value loss, reward loss, and policy
            loss. Loss values are Python floats; priorities cover sampled positions
            and their unrolls.
        """

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            value_mask_batch,
            reward_mask_batch,
            policy_mask_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay.
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.as_tensor(
                weight_batch.copy(), dtype=torch.float32, device=device
            )
        observation_batch = torch.as_tensor(
            numpy.asarray(observation_batch), dtype=torch.float32, device=device
        )
        action_batch = torch.as_tensor(
            action_batch, dtype=torch.long, device=device
        ).unsqueeze(-1)
        target_value = torch.as_tensor(target_value, dtype=torch.float32, device=device)
        target_reward = torch.as_tensor(
            target_reward, dtype=torch.float32, device=device
        )
        target_policy = torch.as_tensor(
            target_policy, dtype=torch.float32, device=device
        )
        value_mask_batch = torch.as_tensor(
            value_mask_batch, dtype=torch.float32, device=device
        )
        reward_mask_batch = torch.as_tensor(
            reward_mask_batch, dtype=torch.float32, device=device
        )
        policy_mask_batch = torch.as_tensor(
            policy_mask_batch, dtype=torch.float32, device=device
        )
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # *_mask_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(
            target_reward, self.config.support_size
        )
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        predictions = self.unroll_model(observation_batch, action_batch)
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        value_loss = torch.zeros(observation_batch.shape[0], device=device)
        reward_loss = torch.zeros(observation_batch.shape[0], device=device)
        policy_loss = torch.zeros(observation_batch.shape[0], device=device)
        # The initial inference has full weight; every recurrent prediction has
        # a 1 / num_unroll_steps gradient scale (See paper appendix Training).
        recurrent_scale = max(1, action_batch.shape[1])
        value, reward, policy_logits = predictions[0]
        (
            current_value_loss,
            current_reward_loss,
            current_policy_loss,
        ) = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
        )
        value_loss += current_value_loss * value_mask_batch[:, 0]
        reward_loss += current_reward_loss * reward_mask_batch[:, 0]
        policy_loss += current_policy_loss * policy_mask_batch[:, 0]
        # Compute priorities for the prioritized replay (See paper appendix Training).
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

            value_loss += current_value_loss * value_mask_batch[:, i] / recurrent_scale
            reward_loss += (
                current_reward_loss * reward_mask_batch[:, i] / recurrent_scale
            )
            policy_loss += (
                current_policy_loss * policy_mask_batch[:, i] / recurrent_scale
            )

            # Compute priorities for the prioritized replay (See paper appendix Training).
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
            ) * value_mask_batch[:, i].detach().cpu().numpy()

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze).
        loss = (
            value_loss * self.config.value_loss_weight
            + reward_loss
            + policy_loss * self.config.policy_loss_weight
        )
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights.
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize.
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
        """Set the exponentially decayed learning rate for the current training step."""
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
        """Return per-sample value, reward, and policy categorical cross-entropies."""
        # Cross-entropy seems to have a better convergence than MSE.
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(
            1
        )
        return value_loss, reward_loss, policy_loss
