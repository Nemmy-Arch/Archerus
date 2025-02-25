import threading
from datetime import datetime, timedelta
import time
from typing import Any, Callable, List, Optional
import tkinter as tk  # for tk.NORMAL
from tkinter import messagebox

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import logger
from data import download_full_historical_data, resample_candlesticks
from indicators import compute_crypto_indicators, feature_cols
from env import TradingEnv
from callbacks import ProgressBarCallback, EarlyStoppingCheckpointCallback
from config import MODEL_DIR
from presets import TIMEFRAME_PRESETS
from gui import user_chosen_timeframe

# Global variables for sharing state
trained_model: Optional[SAC] = None
trained_env: Any = None
selected_symbol: Optional[str] = None
selected_symbol_name: Optional[str] = None
TOTAL_TIMESTEPS: Optional[int] = None

def set_api_key(api_entry: Any, secret_entry: Any) -> None:
    """
    Retrieve API credentials from GUI entries and assign them to config.
    """
    from config import API_KEY, SECRET_KEY, PUBLIC_HEADERS
    API_KEY = api_entry.get().strip()
    SECRET_KEY = secret_entry.get().strip()
    PUBLIC_HEADERS = {"api-key": API_KEY, "Content-Type": "application/json"}
    logger.info(f"API Key set to: {API_KEY}")

def determine_final_training_steps(symbol: str, df_hist, interval: str = "1m") -> int:
    """
    Dynamically determine the training steps based on dataset length and volatility,
    but we override it later with user-selected steps.
    """
    base_steps = 50000
    dataset_length = len(df_hist)
    price_volatility = df_hist['Close'].pct_change().std() * 100
    length_factor = max(1, dataset_length / 500)
    volatility_factor = max(0.5, min(price_volatility / 5, 2))
    total_steps = int(base_steps * length_factor * volatility_factor)
    total_steps = min(max(total_steps, 20000), 15000000)
    logger.info(f"Calculated Training Steps: {total_steps}")
    return total_steps

# --- Binary Search Helpers --- #

def has_data_for_day(symbol: str, interval: str, day: datetime) -> bool:
    """
    Check if data exists for a single day by downloading from day 00:00 to day+1 00:00.
    """
    start_str = day.strftime("%Y-%m-%d 00:00:00")
    end_str = (day + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    df = download_full_historical_data(symbol, start_str, end_str, interval=interval)
    return not df.empty

def find_first_listing_date(symbol: str, interval: str, low: datetime, high: datetime,
                            tolerance: timedelta = timedelta(days=1)) -> Optional[datetime]:
    """
    Use binary search to find the earliest day for which data exists.
    """
    if high - low <= tolerance:
        if has_data_for_day(symbol, interval, low):
            return low
        else:
            return None

    mid = low + (high - low) / 2
    logger.info(f"Binary search checking between {low} and {mid}")
    if has_data_for_day(symbol, interval, mid):
        return find_first_listing_date(symbol, interval, low, mid, tolerance)
    else:
        return find_first_listing_date(symbol, interval, mid, high, tolerance)

def train_model(
    api_entry: Any,
    secret_entry: Any,
    coins_listbox: Any,
    progress_bar: Any,
    log_queue: Any,
    training_steps_value: str,
    live_btn: Any,
    save_model_btn: Any
) -> None:
    """
    Launch training in a background thread to keep GUI responsive.
    We also pass the 'save_model_btn' to re-enable it when training completes.
    """
    train_thread = threading.Thread(
        target=train_model_thread,
        args=(
            api_entry, secret_entry, coins_listbox, progress_bar,
            log_queue, training_steps_value, live_btn, save_model_btn
        ),
        daemon=True
    )
    train_thread.start()

def train_model_thread(
    api_entry: Any,
    secret_entry: Any,
    coins_listbox: Any,
    progress_bar: Any,
    log_queue: Any,
    training_steps_value: str,
    live_btn: Any,
    save_model_btn: Any
) -> None:
    global trained_model, trained_env, selected_symbol, selected_symbol_name, TOTAL_TIMESTEPS
    set_api_key(api_entry, secret_entry)

    # NEW: import user_chosen_timeframe from gui
    from gui import user_chosen_timeframe

    # Check if timeframe is selected
    if not user_chosen_timeframe:
        messagebox.showerror("Error", "No timeframe selected.")
        return

    # Load the preset for that timeframe
    preset = TIMEFRAME_PRESETS.get(user_chosen_timeframe, None)
    if not preset:
        messagebox.showerror("Error", f"No preset found for timeframe: {user_chosen_timeframe}")
        return

    env_params = preset["env_params"]          # e.g. time_penalty, hold_penalty
    training_params = preset["training_params"] # e.g. learning_rate, batch_size, ent_coef

    selection = coins_listbox.curselection()
    if not selection:
        messagebox.showerror("Error", "Please select a trading pair from the list.")
        return

    index = selection[0]
    selected_symbol = coins_listbox.get(index).strip()
    selected_symbol_name = selected_symbol
    logger.info(f"Selected trading pair: {selected_symbol_name}")

    interval = "1m"
    end_date = datetime.now()
    
    # 1. Determine the earliest listing date
    low_date = datetime(2000, 1, 1)
    first_listing_date = find_first_listing_date(selected_symbol, interval, low_date, end_date)
    if first_listing_date is None:
        messagebox.showerror("Error", "No historical data found for training.")
        return

    actual_start_date = first_listing_date.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Coin first listed on: {actual_start_date}")
    
    # 2. Download the full historical data
    df_hist = download_full_historical_data(
        selected_symbol,
        actual_start_date,
        end_date.strftime("%Y-%m-%d %H:%M:%S"),
        interval=interval
    )
    if df_hist.empty:
        messagebox.showerror("Error", "No historical data available for training after determining the start date.")
        return

    logger.info(f"Using {len(df_hist)} candles for training.")
    
    # 3. Preprocess data: resample and compute indicators (including new ones)
    df_hist = resample_candlesticks(df_hist, 1)
    df_hist = compute_crypto_indicators(df_hist)
    logger.info(f"Computed indicators for training. Features used: {feature_cols}")
    
    # 4. Override calculated steps with user selection
    TOTAL_TIMESTEPS = int(training_steps_value)
    logger.info(f"Using user-selected Training Steps: {TOTAL_TIMESTEPS}")

    # NEW: pass env_params to your environment constructor
    def make_env() -> TradingEnv:
        return TradingEnv(df_hist.copy(), feature_cols, env_params=env_params)  # CHANGED

    env_fns: List[Callable[[], TradingEnv]] = [make_env for _ in range(8)]
    trained_env = SubprocVecEnv(env_fns)
    logger.info("Training environment ready.")
    
    # 6. Set up callbacks
    progress_callback = ProgressBarCallback(TOTAL_TIMESTEPS, progress_bar)
    checkpoint_callback = EarlyStoppingCheckpointCallback(
        patience=200000,  # Very large patience
        check_freq=1000,
        save_path=MODEL_DIR
    )
    
    from gui_callbacks import GuiLogCallback
    gui_log_callback = GuiLogCallback(log_queue)

    # 7. Determine device (CPU or GPU)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    logger.info(f"Training on: {device}")
    
    # 8. Create and train the model using training_params for hyperparameters
    model = SAC(
        "MlpPolicy",
        trained_env,
        verbose=1,
        learning_rate=training_params["learning_rate"],  # CHANGED
        ent_coef=training_params["ent_coef"],            # CHANGED
        batch_size=training_params["batch_size"],        # CHANGED
        device=device
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[progress_callback, checkpoint_callback, gui_log_callback],
        reset_num_timesteps=False
    )
    
    # 9. Assign model to the global reference
    global train_model
    trained_model = model
    logger.info(f"Training complete! trained_model id={id(trained_model)}")
    messagebox.showinfo("Success", "Training complete!")
    
    # 10. Enable the Live Recommendation button & Save Model button after training
    live_btn.after(0, lambda: live_btn.config(state=tk.NORMAL))
    save_model_btn.after(0, lambda: save_model_btn.config(state=tk.NORMAL))
