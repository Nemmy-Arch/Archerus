# presets.py

TIMEFRAME_PRESETS = {
    "5m": {
        "env_params": {
            # For high-frequency 5m data, use stronger penalties and a full day (288 bars) per episode.
            "time_penalty": 0.0010,
            "hold_penalty": 0.0100,
            "episode_max_bars": 288  # ~1 day of 5m bars
        },
        "training_params": {
            # Aggressive learning to adapt quickly in a noisy, high-frequency setting.
            "learning_rate": 3e-4,
            "batch_size": 256,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.5
    },
    "15m": {
        "env_params": {
            # Slightly lower penalties for a lower-frequency timeframe, with a day spanning 96 bars.
            "time_penalty": 0.0005,
            "hold_penalty": 0.0050,
            "episode_max_bars": 96  # ~1 day of 15m bars
        },
        "training_params": {
            # A modestly reduced learning rate given smoother price action.
            "learning_rate": 2e-4,
            "batch_size": 256,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 2.0
    },
    "30m": {
        "env_params": {
            # Scaling down penalties further, with 48 bars per day.
            "time_penalty": 0.0003,
            "hold_penalty": 0.0030,
            "episode_max_bars": 48  # ~1 day of 30m bars
        },
        "training_params": {
            # A slower learning rate to capture smoother trends.
            "learning_rate": 1.5e-4,
            "batch_size": 256,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 2.5
    },
    "1h": {
        "env_params": {
            # For hourly data, episodes are shorter (24 bars per day) with even gentler penalties.
            "time_penalty": 0.00015,
            "hold_penalty": 0.0015,
            "episode_max_bars": 24  # ~1 day of 1h bars
        },
        "training_params": {
            # Reduced batch size and learning rate to reflect lower data frequency.
            "learning_rate": 1e-4,
            "batch_size": 128,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 3.0
    },
    "4h": {
        "env_params": {
            # 4h bars yield only 6 points per day; penalties are scaled down accordingly.
            "time_penalty": 0.00005,
            "hold_penalty": 0.0005,
            "episode_max_bars": 6  # ~1 day of 4h bars
        },
        "training_params": {
            # Even slower learning, with a smaller batch size to accommodate the reduced sample size.
            "learning_rate": 5e-5,
            "batch_size": 64,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 4.0
    },
    "8h": {
        "env_params": {
            # With only 3 data points per day, penalties are minimal.
            "time_penalty": 0.000025,
            "hold_penalty": 0.00025,
            "episode_max_bars": 3  # ~1 day of 8h bars
        },
        "training_params": {
            # The slowest learning regime, tuned for very low-frequency data.
            "learning_rate": 2.5e-5,
            "batch_size": 32,
            "ent_coef": "auto_0.1"
        },
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 5.0
    }
}