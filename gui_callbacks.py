# gui_callbacks.py
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any
import queue

class GuiLogCallback(BaseCallback):
    """
    Custom callback that logs training metrics by placing messages into a thread-safe queue.
    """
    def __init__(self, log_queue: queue.Queue, verbose: int = 0):
        super().__init__(verbose)
        self.log_queue = log_queue

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])
        message = f"Step: {self.num_timesteps}, Info: {info}\n"
        self.log_queue.put(message)
        return True