# callbacks.py
import os
import numpy as np
from typing import Any, Dict, Optional
from stable_baselines3.common.callbacks import BaseCallback
from config import MODEL_DIR
from utils import logger

class ProgressBarCallback(BaseCallback):
    """
    A callback for updating a progress bar during training.

    Attributes:
        total_timesteps (int): Total timesteps for the training process.
        progress_bar (Any): A progress bar widget (e.g., from Tkinter) that has a 'master.after' method.
    """
    def __init__(self, total_timesteps: int, progress_bar: Any, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps: int = total_timesteps
        self.progress_bar: Any = progress_bar

    def _on_step(self) -> bool:
        """
        Update the progress bar after each step.

        Returns:
            bool: Always True to continue training.
        """
        progress: int = int(100 * self.num_timesteps / self.total_timesteps)
        
        def update() -> None:
            self.progress_bar['value'] = progress
        
        # Schedule the update on the main thread
        self.progress_bar.master.after(0, update)
        return True

class EarlyStoppingCheckpointCallback(BaseCallback):
    """
    Callback that monitors the average reward over a set frequency of steps,
    saves a checkpoint when an improvement is detected, and stops training
    early if no improvement is seen for a specified number of steps (patience).

    Attributes:
        patience (int): Number of steps to wait without improvement before stopping.
        check_freq (int): Frequency (in steps) at which to check the reward.
        save_path (str): Directory path where model checkpoints are saved.
    """
    def __init__(self, patience: int, check_freq: int, save_path: str = MODEL_DIR, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.patience: int = patience
        self.check_freq: int = check_freq
        self.save_path: str = save_path
        self.best_mean_reward: float = -np.inf
        self.no_improve_steps: int = 0

    def _on_step(self) -> bool:
        """
        Check if the model's performance has improved every `check_freq` steps.
        If performance improves, save a checkpoint; otherwise, increment a counter.
        Stops training if no improvement is detected for more steps than `patience`.

        Returns:
            bool: True to continue training, False to stop.
        """
        if self.n_calls % self.check_freq == 0:
            # Retrieve the 'infos' dictionary from the local variables
            infos: Optional[Any] = self.locals.get("infos", [{}])
            if infos:
                # Compute the mean reward from the provided info dictionaries
                mean_reward: float = np.mean([info.get("reward", 0) for info in infos])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improve_steps = 0
                    checkpoint_file: str = os.path.join(self.save_path, f"model_checkpoint_{self.num_timesteps}.zip")
                    self.model.save(checkpoint_file)
                    logger.info(f"Checkpoint saved: {checkpoint_file} with mean reward: {mean_reward:.2f}")
                else:
                    self.no_improve_steps += self.check_freq
                    logger.info(f"No improvement detected for {self.no_improve_steps} steps (best: {self.best_mean_reward:.2f}).")
                    if self.no_improve_steps > self.patience:
                        logger.info("Early stopping triggered due to lack of improvement.")
                        return False
        return True