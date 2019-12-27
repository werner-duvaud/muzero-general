import ray
import torch


class Network(torch.nn.Module):
    def __init__(self, input_size, action_space_n, encoding_size, hidden_size):
        super().__init__()
        self.action_space_n = action_space_n

        self.representation_network = FullyConnectedNetwork(
            input_size, [], encoding_size
        )

        self.dynamics_state_network = FullyConnectedNetwork(
            encoding_size + self.action_space_n, [hidden_size], encoding_size
        )
        self.dynamics_reward_network = FullyConnectedNetwork(
            encoding_size + self.action_space_n, [hidden_size], 1
        )

        self.prediction_actor_network = FullyConnectedNetwork(
            encoding_size, [], self.action_space_n, activation=None
        )
        self.prediction_value_network = FullyConnectedNetwork(
            encoding_size, [], 1, activation=None
        )

    def prediction(self, state):
        actor_logit = self.prediction_actor_network(state)
        value = self.prediction_value_network(state)
        return actor_logit, value

    def representation(self, observation):
        return self.representation_network(observation)

    def dynamics(self, state, action):
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_n))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self.dynamics_state_network(x)
        reward = self.dynamics_reward_network(x)
        return next_state, reward

    def initial_inference(self, observation):
        state = self.representation(observation)
        actor_logit, value = self.prediction(state)
        return (
            value,
            torch.zeros(len(observation)).to(observation.device),
            actor_logit,
            state,
        )

    def recurrent_inference(self, hidden_state, action):
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)
        return value, reward, actor_logit, state


def update_weights(optimizer, model, batch, config):
    observation_batch, action_batch, target_reward, target_value, target_policy = batch

    observation_batch = torch.tensor(observation_batch).float().to(config.device)
    action_batch = torch.tensor(action_batch).float().to(config.device).unsqueeze(-1)
    target_value = torch.tensor(target_value).float().to(config.device)
    target_reward = torch.tensor(target_reward).float().to(config.device)
    target_policy = torch.tensor(target_policy).float().to(config.device)

    value, reward, policy_logits, hidden_state = model.initial_inference(
        observation_batch
    )
    predictions = [(value, reward, policy_logits)]
    for action_i in range(config.num_unroll_steps):
        value, reward, policy_logits, hidden_state = model.recurrent_inference(
            hidden_state, action_batch[:, action_i]
        )
        predictions.append((value, reward, policy_logits))

    loss = 0
    for i, prediction in enumerate(predictions):
        value, reward, policy_logits = prediction
        loss += loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            target_value[:, i],
            target_reward[:, i],
            target_policy[:, i, :],
        )

    # Scale gradient by number of unroll steps (See paper Training appendix)
    loss = loss.mean() / config.num_unroll_steps

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def loss_function(
    value, reward, policy_logits, target_value, target_reward, target_policy
):
    # TODO: paper promotes cross entropy instead of MSE
    value_loss = torch.nn.MSELoss(reduction="none")(value, target_value)
    reward_loss = torch.nn.MSELoss(reduction="none")(reward, target_reward)
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy).sum(1)
    return value_loss + reward_loss + policy_loss


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(
        self, input_size, layers_sizes, output_size, activation=torch.nn.Tanh()
    ):
        super(FullyConnectedNetwork, self).__init__()
        layers_sizes.insert(0, input_size)
        layers = []
        if 1 < len(layers_sizes):
            for i in range(len(layers_sizes) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(layers_sizes[i], layers_sizes[i + 1]),
                        torch.nn.ReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(layers_sizes[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@ray.remote
class SharedStorage:
    def __init__(self, model):
        self.training_step = 0
        self.model = model

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        return self.model.load_state_dict(weights)

    def set_training_step(self, training_step):
        self.training_step = training_step

    def get_training_step(self):
        return self.training_step
