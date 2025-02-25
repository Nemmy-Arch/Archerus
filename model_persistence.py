# model_persistence.py
import os
from tkinter import messagebox
from typing import Optional

from stable_baselines3 import SAC
from config import MODEL_DIR
from utils import logger
from training import trained_env  # only if you need 'trained_env'

MODEL_DIR = r"F:\Project Archerus\v3\saved_models"

def save_model(model, total_timesteps: int, selected_symbol: str, timeframe_str: str) -> None:
    """
    Save a trained model to disk with a filename based on the trading symbol, timeframe, and total timesteps.
    """
    import training  # so we can fallback to training.trained_model if needed

    # If model is None, fallback to the global 'trained_model'
    if model is None:
        model = training.trained_model

    if model is None:
        messagebox.showerror("Error", "No trained model to save!")
        logger.error("❌ No trained model available to save!")
        return

    logger.info(f"✅ Saving model: id={id(model)}")

    model_filename = f"SAC_{selected_symbol}_{timeframe_str}_{total_timesteps}.zip"
    model_path = os.path.join(MODEL_DIR, model_filename)

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        model.save(model_path)
        messagebox.showinfo("Success", f"Model saved: {model_path}")
        logger.info(f"✅ Model saved: {model_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {str(e)}")
        logger.error(f"❌ Failed to save model: {str(e)}")


def load_model(file_path: str) -> Optional[SAC]:
    """
    Load a trained model from disk WITHOUT attaching an environment.
    This is useful if you only need model.predict().
    """
    import training  # Use the training module's global variable

    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"Model file not found: {file_path}")
        logger.error(f"Model file not found: {file_path}")
        return None

    try:
        model = SAC.load(file_path)
        training.trained_model = model
        messagebox.showinfo("Success", f"Model loaded successfully from {file_path}")
        logger.info(f"Model loaded: {file_path}, training.trained_model id={id(training.trained_model)}")
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Could not load model: {str(e)}")
        logger.error(f"Could not load model: {str(e)}")
        return None