# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Any
import queue
user_chosen_timeframe = None

import training
from account import get_trading_pairs
from model_persistence import save_model, load_model
from recommender import get_recommendation_message, feature_cols
from simulation import LiveSimulation
from utils import logger

def start_gui() -> None:
    """
    Start the main GUI application.
    """
    root = tk.Tk()
    root.title("Archerus Deathcharger")
    root.geometry("1000x800+100+100")
    root.grid_columnconfigure(0, weight=1)
    
    expandable_rows = [3, 4, 6, 7, 10]  
    for row in expandable_rows:
        root.grid_rowconfigure(row, weight=1)
    
    # --- API Credentials Frame --- #
    api_frame = ttk.LabelFrame(root, text="API Credentials")
    api_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
    ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, padx=5, pady=5)
    api_entry = ttk.Entry(api_frame, width=50)
    api_entry.grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(api_frame, text="Secret Key:").grid(row=0, column=2, padx=5, pady=5)
    secret_entry = ttk.Entry(api_frame, width=50, show="*")
    secret_entry.grid(row=0, column=3, padx=5, pady=5)
    
    # --- Trading Time Frame Frame --- #
    window_frame = ttk.LabelFrame(root, text="Trading Time Frame")
    window_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
    window_label = ttk.Label(window_frame, text="Selected Time Frame: 15m (15 minutes)")
    window_label.grid(row=0, column=0, columnspan=15, padx=5, pady=5)

    timeframe_options = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mon"]
    timeframe_map = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080, "1mon": 43200
    }
    
    def set_trading_window_time(tf_str: str) -> None:
        global user_chosen_timeframe
        user_chosen_timeframe = tf_str
        minutes = timeframe_map[tf_str]
        window_label.config(text=f"Selected Time Frame: {tf_str} ({minutes} minutes)")

    # Create a button for each timeframe option
    for i, tf in enumerate(timeframe_options):
        btn = ttk.Button(
            window_frame,
            text=tf,
            command=lambda tf=tf: set_trading_window_time(tf)
        )
        btn.grid(row=1, column=i, padx=2, pady=2)

    
    def on_symbol_select(event):
        selection = coins_listbox.curselection()
        if not selection:
            training.selected_symbol = None
            return
        index = selection[0]
        training.selected_symbol = coins_listbox.get(index).strip()
        logger.info(f"Selected symbol: {training.selected_symbol}")
    
    # --- Trading Pairs Frame --- #
    coins_frame = ttk.LabelFrame(root, text="Trading Pairs")
    coins_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
    coins_frame.grid_columnconfigure(0, weight=1)
    
    coins_listbox = tk.Listbox(coins_frame, height=10)
    coins_listbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    coins_scrollbar = ttk.Scrollbar(coins_frame, orient="vertical", command=coins_listbox.yview)
    coins_scrollbar.grid(row=0, column=1, sticky="ns")
    coins_listbox.config(yscrollcommand=coins_scrollbar.set)
    coins_listbox.bind('<<ListboxSelect>>', on_symbol_select)
    search_frame = ttk.Frame(coins_frame)
    search_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
    ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=2)
    search_entry = ttk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    # Cache the full list for filtering
    full_trading_pairs = []

    def filter_trading_pairs(event):
        search_term = search_entry.get().lower()
        filtered_pairs = [pair for pair in full_trading_pairs if search_term in pair.lower()]
        coins_listbox.delete(0, tk.END)
        for pair in filtered_pairs:
            coins_listbox.insert(tk.END, pair)

    search_entry.bind("<KeyRelease>", filter_trading_pairs)

        
    def load_trading_pairs_ui() -> None:
        global full_trading_pairs
        pairs = get_trading_pairs()
        full_trading_pairs = [pair.get("symbol") for pair in pairs if pair.get("symbol")]
        coins_listbox.delete(0, tk.END)
        for symbol in full_trading_pairs:
            coins_listbox.insert(tk.END, symbol)
    
    load_coins_btn = ttk.Button(root, text="Load Trading Pairs", command=load_trading_pairs_ui)
    load_coins_btn.grid(row=2, column=0, padx=5, pady=5)
    
    # --- Training Log Frame --- #
    log_frame = ttk.LabelFrame(root, text="Training Log")
    log_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
    log_frame.grid_columnconfigure(0, weight=1)
    log_text = tk.Text(log_frame, height=8)
    log_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
    log_scrollbar.grid(row=0, column=1, sticky="ns")
    log_text.config(yscrollcommand=log_scrollbar.set)
    
    # --- Live Recommendation Frame --- #
    live_frame = ttk.LabelFrame(root, text="Live Recommendation")
    live_frame.grid(row=6, column=0, sticky="nsew", padx=10, pady=5)
    live_frame.grid_columnconfigure(0, weight=1)
    live_text = ScrolledText(live_frame, height=10, width=80)
    live_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    price_label = ttk.Label(live_frame, text="Live Price: N/A")
    price_label.grid(row=1, column=0, padx=5, pady=5)
    
    # --- Simulation Trading Frame --- #
    simulation_frame = ttk.LabelFrame(root, text="Live Simulation Trading (Paper Trading)")
    simulation_frame.grid(row=7, column=0, sticky="nsew", padx=10, pady=5)
    simulation_frame.grid_columnconfigure(0, weight=1)
    sim_text = ScrolledText(simulation_frame, height=10, width=80)
    sim_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # --- Model Activity Log Frame --- #
    model_activity_frame = ttk.LabelFrame(root, text="Model Activity Log")
    model_activity_frame.grid(row=10, column=0, sticky="nsew", padx=10, pady=5)
    model_activity_frame.grid_columnconfigure(0, weight=1)
    model_activity_text = ScrolledText(model_activity_frame, height=10, width=80)
    model_activity_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # --- Buttons Frame --- #
    buttons_frame = ttk.Frame(root)
    buttons_frame.grid(row=9, column=0, sticky="ew", padx=10, pady=5)
    buttons_frame.config(height=50)
    buttons_frame.grid_propagate(False)
    
    # Create a sub-frame for training steps
    steps_subframe = ttk.LabelFrame(buttons_frame, text="Training Steps")
    steps_subframe.grid(row=0, column=4, padx=10, pady=5, sticky="w")
    training_steps_var = tk.StringVar(value="1500000")
    step_options = [
        ("Basic (1.5M)", "1500000"),
        ("Average (5M)", "5000000"),
        ("Comprehensive (15M)", "15000000"),
        ("Override (50M)", "50000000")
    ]
    for i, (label, value) in enumerate(step_options):
        rb = ttk.Radiobutton(steps_subframe, text=label, variable=training_steps_var, value=value)
        rb.grid(row=0, column=i, padx=5, pady=5)
    
    def get_live_recommendation() -> None:
        if training.trained_model is None or training.selected_symbol is None:
            messagebox.showerror("Error", "No trained model available.")
            return
        
        # NEW: If you want to log or enforce the loaded timeframe
        if getattr(training, "selected_timeframe", None):
            from config import TRADING_WINDOW
            timeframe_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "8h": 480}
            tf_str = training.selected_timeframe
            if tf_str in timeframe_map:
                TRADING_WINDOW = timeframe_map[tf_str]
                logger.info(f"Live recommendation locked to timeframe {tf_str} -> {TRADING_WINDOW} minutes")
            else:
                logger.info("Loaded model timeframe is unknown or not in timeframe_map.")

        message, decision = get_recommendation_message(training.trained_model, feature_cols, training.selected_symbol)
        live_text.delete("1.0", tk.END)
        live_text.insert(tk.END, message)
        live_text.see(tk.END)


        
    
    def load_model_and_enable() -> None:
        file_path = filedialog.askopenfilename(
            initialdir="saved_models", title="Select Model",
            filetypes=[("Stable Baselines Model", "*.zip")]
        )
        if not file_path:
            return
        model = load_model(file_path)
        if model is not None:
            # NEW: Parse timeframe from filename
            import os
            basename = os.path.basename(file_path)
            parts = basename.split("_")  # e.g. ["SAC", "BTCUSDT", "15m", "1500000.zip"]
            if len(parts) >= 3:
                loaded_timeframe = parts[2]  # "15m"
                training.selected_timeframe = loaded_timeframe
                logger.info(f"Parsed timeframe from filename: {loaded_timeframe}")
            else:
                training.selected_timeframe = None

            logger.info(f"Loaded model, training.trained_model id={id(training.trained_model)}")
            live_btn.config(state=tk.NORMAL)
            save_model_btn.config(state=tk.NORMAL)

    
    def on_save_model():
        from gui import user_chosen_timeframe 
        logger.info(f"Inside on_save_model, training.trained_model id={id(training.trained_model)}")
        save_model(
        training.trained_model,      # model
        0,                           # total timesteps (or whatever you want)
        training.selected_symbol if training.selected_symbol else "UNKNOWN",
        user_chosen_timeframe if user_chosen_timeframe else "UNKNOWN"
    )
    
    load_model_btn = ttk.Button(
        buttons_frame, text="Load Model",
        command=load_model_and_enable
    )
    load_model_btn.grid(row=0, column=0, padx=5, pady=5)
    
    # Initially disable Save Model, will enable after training or loading
    save_model_btn = ttk.Button(
        buttons_frame, text="Save Model",
        command=on_save_model,
        state=tk.DISABLED
    )
    save_model_btn.grid(row=0, column=1, padx=5, pady=5)
    
    live_btn = ttk.Button(
        buttons_frame, text="Get Live Recommendation",
        command=get_live_recommendation,
        state=tk.DISABLED
    )
    live_btn.grid(row=0, column=3, padx=5, pady=5)
    
    # Create a thread-safe queue for log messages
    log_queue = queue.Queue()
    
    def on_train_model():
        logger.info("Train Model button clicked")
        save_model_btn.config(state=tk.DISABLED)
        training.train_model(
            api_entry, secret_entry, coins_listbox, progress_bar,
            log_queue, training_steps_var.get(), live_btn, save_model_btn
        )
    
    train_btn = ttk.Button(
        buttons_frame, text="Train Model",
        command=on_train_model
    )
    train_btn.grid(row=0, column=2, padx=5, pady=5)
    
    # Training Progress
    progress_frame = ttk.LabelFrame(root, text="Training Progress")
    progress_frame.grid(row=8, column=0, sticky="ew", padx=10, pady=5)
    progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
    progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    
    def poll_log_queue() -> None:
        try:
            while True:
                message = log_queue.get_nowait()
                log_text.insert("end", message)
                log_text.see("end")
        except Exception:
            pass
        root.after(100, poll_log_queue)
    poll_log_queue()
    
    root.update_idletasks()
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.after(0, lambda: root.attributes('-topmost', False))
    root.focus_force()
    
    root.mainloop()

if __name__ == "__main__":
    start_gui()