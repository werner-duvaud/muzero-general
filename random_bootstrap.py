"""Initialize replay from the best episodes collected with random legal actions."""

import copy
import heapq

import numpy
import ray

from evaluation import write_json
from game_history import GameHistory


def play_random_game(game, config, rng):
    """Record an episode with an independent uniform legal action at each step."""
    return play_game(game, config, lambda observation, actions: rng.choice(actions))


def play_game(game, config, select_action):
    """Record a behavior-policy episode and observed discounted return targets.

    Args:
        game (AbstractGame): Game to reset and play up to config.max_moves.
        config (MuZeroConfig): Reward scaling, discount, players, and action space.
        select_action (callable): Callback receiving the current unstacked
            observation and legal actions, and returning an action to execute.

    Returns:
        GameHistory: Trajectory with scaled rewards, discounted observed values,
        uniform initial priorities, and one-hot executed-action policy targets.
        In multiplayer games, opponent actions have zero policy targets.
    """
    history = GameHistory()
    history.observation_history.append(game.reset())
    history.to_play_history.append(game.to_play())
    done = False
    while not done and len(history.action_history) < config.max_moves:
        legal_actions = list(game.legal_actions())
        action = select_action(history.observation_history[-1], legal_actions)
        observation, reward, done = game.step(action)
        history.legal_actions_history.append(legal_actions)
        history.action_history.append(action)
        history.reward_history.append(reward / getattr(config, "reward_scale", 1))
        history.observation_history.append(observation)
        history.to_play_history.append(game.to_play())
        imitate_player = (
            len(config.players) == 1
            or history.to_play_history[-2] == config.muzero_player
        )
        # Opponent actions in a selected loss are not policy demonstrations.
        history.child_visits.append(
            [
                float(imitate_player and candidate == action)
                for candidate in config.action_space
            ]
        )

    value = 0.0
    history.root_values = [0.0] * len(history.action_history)
    for position in reversed(range(len(history.action_history))):
        if history.to_play_history[position] != history.to_play_history[position + 1]:
            value = -value
        value = history.reward_history[position] + config.discount * value
        history.root_values[position] = value
    history.observation_history = numpy.asarray(history.observation_history)
    # Sample demonstrations uniformly until the trainer updates their errors.
    history.priorities = numpy.ones(len(history.action_history), dtype="float32")
    history.game_priority = 1.0
    return history


def retain_best(heap, history, keep_best, sequence, player=None):
    """Keep the highest-return histories in a bounded min-heap, in place.

    Score from player's perspective, defaulting to the initial player. Higher
    sequence numbers win equal-score ties. Copy retained read-only Ray arrays
    to avoid keeping discarded batch data alive.
    """
    if player is None:
        player = history.to_play_history[0]
    score = sum(
        reward if history.to_play_history[index] == player else -reward
        for index, reward in enumerate(history.reward_history)
    )
    entry = (score, sequence, history)
    replacing = len(heap) >= keep_best
    if replacing and not (entry[:2] > heap[0][:2]):
        return
    # Ray arrays can pin the entire returned batch. Own only retained data.
    if (
        isinstance(history.observation_history, numpy.ndarray)
        and not history.observation_history.flags.writeable
    ):
        entry = (score, sequence, copy.deepcopy(history))
    if replacing:
        heapq.heapreplace(heap, entry)
    else:
        heapq.heappush(heap, entry)


@ray.remote(num_cpus=1)
def collect_random_episodes(Game, config, num_episodes, keep_best, seed):
    """Collect seeded random episodes without a network or MCTS.

    Return a bounded heap of (return, sequence, GameHistory) entries, ranked from
    config.muzero_player's perspective in training reward units.
    """
    numpy.random.seed(seed)
    rng = numpy.random.default_rng(seed)
    game = Game(seed)
    best = []
    try:
        for episode in range(num_episodes):
            history = play_random_game(game, config, rng)
            retain_best(best, history, keep_best, episode, config.muzero_player)
    finally:
        game.close()
    return best


def bootstrap_replay_buffer(Game, config):
    """Collect random episodes in CPU workers and retain the best returns.

    Keep at most random_bootstrap_keep_best episodes, bounded by the number
    collected and replay capacity. Save a collection summary when enabled.

    Returns:
        dict: Integer IDs mapped to GameHistory objects, inserted from lower to
        higher return for FIFO eviction. Empty when random collection is disabled.
    """
    episodes = getattr(config, "random_bootstrap_episodes", 0)
    if not isinstance(episodes, int) or episodes < 0:
        raise ValueError("random_bootstrap_episodes must be a nonnegative integer")
    if not episodes:
        return {}
    keep_best = config.random_bootstrap_keep_best
    if not isinstance(keep_best, int) or keep_best < 1:
        raise ValueError("random_bootstrap_keep_best must be a positive integer")
    if min(config.num_workers, config.max_moves, config.replay_buffer_size) < 1:
        raise ValueError(
            "Random replay bootstrap needs workers, moves and buffer space"
        )
    keep_best = min(keep_best, episodes, config.replay_buffer_size)

    print(f"Collecting {episodes} bootstrap episodes; keeping the best {keep_best}.")
    # Small batches provide progress and avoid storing all random trajectories
    # in Ray's object store. Chunk seeds and tie breaks are deterministic.
    batch_size = 100
    pending = {}
    submitted = completed = 0
    best = []
    try:
        while completed < episodes:
            while submitted < episodes and len(pending) < config.num_workers:
                count = min(batch_size, episodes - submitted)
                task = collect_random_episodes.remote(
                    Game,
                    config,
                    count,
                    keep_best,
                    config.seed + submitted,
                )
                pending[task] = (submitted, count)
                submitted += count
            ready, _ = ray.wait(list(pending), num_returns=1)
            task = ready[0]
            start, count = pending.pop(task)
            for _, sequence, history in ray.get(task):
                retain_best(
                    best, history, keep_best, start + sequence, config.muzero_player
                )
            completed += count
            print(f"Random bootstrap: {completed}/{episodes} episodes", end="\r")
    finally:
        for task in pending:
            ray.cancel(task)

    returns = [entry[0] * config.reward_scale for entry in best]
    print(
        f"\nRetained {len(best)} episodes; environment returns "
        f"{min(returns):.2f} to {max(returns):.2f}, mean {numpy.mean(returns):.2f}.",
        flush=True,
    )
    if getattr(config, "save_model", False):
        write_json(
            config.results_path / "bootstrap-replay.json",
            {
                "sampled_episodes": episodes,
                "retained_episodes": len(best),
                "action_sampling": "Independent uniform legal random actions.",
                "episodes": [
                    {
                        "sequence": int(sequence),
                        "reward": float(score * config.reward_scale),
                        "length": len(history.action_history),
                    }
                    for score, sequence, history in sorted(best)
                ],
            },
        )
    # Insert lower-return demonstrations first so normal FIFO eviction removes
    # them before the strongest demonstrations as self-play fills the buffer.
    return {index: entry[2] for index, entry in enumerate(sorted(best))}
