"""Record portable run metadata and seeded evaluation reports for every game."""

import copy
import hashlib
import importlib.metadata
import pathlib
import platform
import subprocess

import numpy
import ray


@ray.remote(num_cpus=1)
def evaluate_episode_group(checkpoint, Game, config, seeds):
    """Evaluate one seed group with frozen weights and CPU inference.

    Return one episode result per seed, closing the game even if evaluation fails.
    The seed group must be nonempty.
    """
    from self_play import SelfPlay

    settings = copy.deepcopy(config)
    settings.selfplay_on_gpu = False
    worker = SelfPlay.__ray_metadata__.modified_class(
        checkpoint, Game, settings, seeds[0]
    )
    try:
        return [worker.evaluate_episode(seed) for seed in seeds]
    finally:
        worker.close_game()


class Validation:
    """Evaluate frozen snapshots independently of ongoing self-play and training."""

    def __init__(self, Game, config):
        self.Game = Game
        self.config = config
        self.tasks = []
        self.episodes = []
        self.history = []
        self.best_checkpoint = None
        self.best_path = None
        self.best_score = -numpy.inf

    def start(self, checkpoint):
        """Launch validation of a frozen checkpoint on configured validation seeds.

        Raise RuntimeError if another validation is still running.
        """
        if self.tasks:
            raise RuntimeError("A validation is already running")
        self.snapshot = copy.deepcopy(checkpoint)
        # Optimizer and weights are published together at weights_version.
        self.snapshot["training_step"] = self.snapshot["weights_version"]
        checkpoint = ray.put(self.snapshot)
        seeds = list(
            range(
                self.config.validation_seed,
                self.config.validation_seed + self.config.validation_episodes,
            )
        )
        workers = min(self.config.num_workers, len(seeds))
        self.episodes = []
        self.tasks = [
            evaluate_episode_group.remote(
                checkpoint, self.Game, self.config, seeds[i::workers]
            )
            for i in range(workers)
        ]

    def poll(self, wait=False):
        """Collect completed tasks and return a report once validation finishes.

        Args:
            wait (bool): Wait for all pending episodes instead of polling.

        Returns:
            dict or None: Completed validation report, or None when idle or still
            running. Completed reports update validation.json and the best snapshot;
            best weights are saved when save_model is enabled.
        """
        if not self.tasks:
            return None
        ready, _ = ray.wait(
            self.tasks,
            num_returns=len(self.tasks),
            timeout=None if wait else 0,
        )
        for task in ready:
            self.episodes.extend(ray.get(task))
            self.tasks.remove(task)
        if self.tasks:
            return None
        episodes = sorted(self.episodes, key=lambda episode: episode["seed"])
        report = summarize_evaluation(episodes, self.config.evaluation_target)
        report["training_step"] = self.snapshot["training_step"]
        self.history.append(report)
        write_json(self.config.results_path / "validation.json", self.history)
        if report["mean_reward"] > self.best_score:
            self.best_score = report["mean_reward"]
            self.best_checkpoint = self.snapshot
            if self.config.save_model:
                import torch

                self.best_path = self.config.results_path / "model-best.checkpoint"
                temporary = self.best_path.with_suffix(".checkpoint.tmp")
                torch.save(self.best_checkpoint, temporary)
                temporary.replace(self.best_path)
        print(
            f"\nValidation at {report['training_step']}: "
            f"mean environment reward {report['mean_reward']:.2f}; "
            f"best {self.best_score:.2f}",
            flush=True,
        )
        return report

    def cancel(self):
        """Cancel pending Ray validation tasks and clear their handles."""
        for task in self.tasks:
            ray.cancel(task, force=True)
        self.tasks = []


def write_json(path, data):
    """Atomically write finite JSON data, creating parent directories as needed."""
    import json

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    """Return a file's SHA-256 hex digest without loading it all into memory."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata():
    """Return Python/package versions and available Git and source fingerprints."""
    versions = {}
    for package in ("torch", "numpy", "ray", "gymnasium", "ale-py", "Box2D"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = {"python": platform.python_version(), "packages": versions}
    root = pathlib.Path(__file__).resolve().parent
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "*.py", "requirements.txt"],
            cwd=root,
        )
        metadata["tracked_code_dirty"] = bool(changes)
        metadata["tracked_diff_sha256"] = hashlib.sha256(changes).hexdigest()
        # Include untracked modules too, so local runs can be identified precisely.
        metadata["source_sha256"] = {
            str(path.relative_to(root)): file_sha256(path)
            for pattern in ("*.py", "games/*.py")
            for path in sorted(root.glob(pattern))
        }
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def summarize_evaluation(episodes, target):
    """Summarize episode returns and compare their mean with an optional target.

    Args:
        episodes (list[dict]): Nonempty episode results with finite reward values.
        target (float, optional): Mean return required to pass; None skips comparison.

    Returns:
        dict: Original episodes, return statistics, and target_met (None if disabled).

    Raises:
        ValueError: Episodes are empty or rewards contain nonfinite values.
    """
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    rewards = [episode["reward"] for episode in episodes]
    if not numpy.isfinite(rewards).all():
        raise ValueError("Evaluation returned nonfinite rewards")
    mean = float(numpy.mean(rewards))
    return {
        "episode_count": len(episodes),
        "mean_reward": mean,
        "std_reward": float(numpy.std(rewards)),
        "median_reward": float(numpy.median(rewards)),
        "minimum_reward": float(numpy.min(rewards)),
        "maximum_reward": float(numpy.max(rewards)),
        "target_reward": target,
        "target_met": None if target is None else mean >= target,
        "episodes": episodes,
    }
