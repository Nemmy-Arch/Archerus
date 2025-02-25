import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Any, Dict
from utils import logger

class TradingEnv(gym.Env):
    """
    A custom trading environment for a reinforcement learning model.
    
    Observations: Feature vector based on technical indicators.
    Actions: Continuous action in a Box space (interpreted as buy, sell, or hold).
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, df, feature_columns: list, env_params: Dict[str, Any] = None):
        """
        Initialize the Trading Environment.

        Args:
            df (pd.DataFrame): DataFrame containing historical trading data with required features.
            feature_columns (list): List of columns to be used as observation features.
            env_params (dict, optional): Dictionary of environment parameters (time_penalty, hold_penalty, etc.).
        """
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(len(feature_columns),), dtype=np.float32)
        
        # -- Existing Trading Parameters --
        self.initial_balance: float = 100000.0
        self.risk_fraction: float = 0.02
        self.alpha: float = 0.95
        self.waiting_reward: float = 0.01
        self.reversal_penalty: float = 5
        self.trade_freq_penalty: float = 2
        self.trade_freq_window: int = 10
        self.regime_bonus: float = 1
        self.regime_penalty: float = 1
        self.sharpe_threshold: float = 0.5
        self.terminal_penalty: float = 5

        # NEW: Additional environment parameters from env_params
        self.env_params = env_params or {}
        self.time_penalty = self.env_params.get("time_penalty", 0.0)
        self.hold_penalty = self.env_params.get("hold_penalty", 0.0)
        self.episode_max_bars = self.env_params.get("episode_max_bars", len(self.df))

        self.reset()

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options (unused).

        Returns:
            Tuple[np.ndarray, dict]: Initial observation and an empty info dictionary.
        """
        if seed is not None:
            np.random.seed(seed)
        self.current_step: int = 0
        self.balance: float = self.initial_balance
        self.trades: list = []
        self.trade_history: list = []
        self.last_trade_action: int = None
        self.last_trade_step: int = -100
        self.prev_reward: float = 0.0
        self.max_balance: float = self.initial_balance

        return self._next_observation(), {}

    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation from the dataset.

        Returns:
            np.ndarray: A vector of features.
        """
        try:
            obs = self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)
        except Exception as e:
            logger.error(f"Error accessing observation at step {self.current_step}: {e}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if np.isnan(obs).any():
            logger.error(f"NaN detected in observation at step {self.current_step}: {obs}")
            obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take an action in the environment and update the state.

        Args:
            action (np.ndarray): The action array with shape (1,).

        Returns:
            Tuple containing:
                - next observation (np.ndarray),
                - reward (float),
                - terminated flag (bool),
                - truncated flag (bool),
                - info dictionary (dict).
        """
        # Validate action shape
        if action.shape != (1,):
            logger.error("Action must be a numpy array with shape (1,)")
            action = np.array([0.0], dtype=np.float32)
        
        act = action[0]
        # Define trade actions: 2 (buy), 1 (hold), 0 (sell)
        if act > 0.1:
            trade_action = 2
        elif act < -0.1:
            trade_action = 0
        else:
            trade_action = 1

        current_price = self.df.loc[self.current_step, 'Close']

        # Check if we have reached the end of data or the episode_max_bars
        if self.current_step >= len(self.df) - 2 or self.current_step >= self.episode_max_bars:
            obs = self._next_observation()
            volatility = np.std(self.trade_history) if len(self.trade_history) > 1 and np.std(self.trade_history) > 0 else 1.0
            terminal_reward = (self.balance - self.initial_balance) / volatility
            sharpe = np.mean(self.trade_history) / volatility if len(self.trade_history) > 1 else 1.0
            if sharpe < self.sharpe_threshold:
                terminal_reward -= self.terminal_penalty
            new_reward = self.alpha * self.prev_reward + (1 - self.alpha) * terminal_reward
            self.prev_reward = new_reward
            return obs, new_reward, True, False, {}

        next_price = self.df.loc[self.current_step + 1, 'Close']
        # Compute base reward based on action taken
        if trade_action == 2:  # Buy
            raw_reward = (next_price - current_price) / current_price * self.risk_fraction * self.balance
        elif trade_action == 0:  # Sell
            raw_reward = (current_price - next_price) / current_price * self.risk_fraction * self.balance
        else:  # Hold
            raw_reward = self.waiting_reward

        # If agent is in a position, apply a hold penalty from env_params (optional)
        if trade_action != 1:
            raw_reward -= self.hold_penalty

        # Update balance and record trade if a transaction occurred
        if trade_action != 1:
            price_change = (next_price - current_price) / current_price if trade_action == 2 else (current_price - next_price) / current_price
            self.balance += self.risk_fraction * self.balance * price_change
            self.trades.append({
                'step': self.current_step,
                'action': trade_action,
                'entry': current_price,
                'exit': next_price,
                'reward': raw_reward
            })
            self.trade_history.append(raw_reward)
            # Apply reversal penalty if switching direction within a short period
            if (self.last_trade_action is not None and 
                trade_action != self.last_trade_action and 
                (self.current_step - self.last_trade_step) < 5):
                raw_reward -= self.reversal_penalty
            self.last_trade_action = trade_action
            self.last_trade_step = self.current_step

        # Penalize for excessive trading frequency within a short window
        recent_trades = [t for t in self.trades if self.current_step - t['step'] < self.trade_freq_window]
        if len(recent_trades) > 3:
            raw_reward -= self.trade_freq_penalty * (len(recent_trades) - 3)

        # Adjust reward based on moving average regime (sma_50 vs sma_200)
        sma_50 = self.df.loc[self.current_step, 'sma_50']
        sma_200 = self.df.loc[self.current_step, 'sma_200']
        if sma_50 > sma_200:
            if trade_action == 2:
                raw_reward += self.regime_bonus
            elif trade_action == 0:
                raw_reward -= self.regime_penalty
        elif sma_50 < sma_200:
            if trade_action == 0:
                raw_reward += self.regime_bonus
            elif trade_action == 2:
                raw_reward -= self.regime_penalty

        # --- Additional Reward Adjustment Using New Indicators ---
        # These adjustments use MACD, ADX, and CCI (if available in the feature set)
        # You can fine-tune the bonus/penalty multipliers as needed.
        if all(col in self.feature_columns for col in ['macd', 'macd_signal', 'adx', 'cci']):
            macd = self.df.loc[self.current_step, 'macd']
            macd_signal = self.df.loc[self.current_step, 'macd_signal']
            adx = self.df.loc[self.current_step, 'adx']
            cci_val = self.df.loc[self.current_step, 'cci']
            
            # MACD bonus: add a small bonus if MACD is above its signal
            macd_bonus = 0.0
            if macd > macd_signal:
                macd_bonus = 0.005 * self.balance
            else:
                macd_bonus = -0.005 * self.balance
            
            # ADX bonus: if ADX is above 25, add bonus for trending market
            adx_bonus = 0.0
            if adx > 25:
                adx_bonus = 0.005 * self.balance
            
            # CCI bonus: add bonus if CCI is strongly bullish (>100) or subtract if strongly bearish (<-100)
            cci_bonus = 0.0
            if cci_val > 100:
                cci_bonus = 0.005 * self.balance
            elif cci_val < -100:
                cci_bonus = -0.005 * self.balance
            
            raw_reward += (macd_bonus + adx_bonus + cci_bonus)
            logger.info(f"Additional rewards: MACD {macd_bonus:.2f}, ADX {adx_bonus:.2f}, CCI {cci_bonus:.2f}")
        
        # Smooth reward with a running average
        new_reward = self.alpha * self.prev_reward + (1 - self.alpha) * raw_reward
        self.prev_reward = new_reward

        self.max_balance = max(self.max_balance, self.balance)
        self.current_step += 1
        obs = self._next_observation() if self.current_step < len(self.df) else np.zeros(self.observation_space.shape, dtype=np.float32)
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return obs, new_reward, terminated, truncated, {}

    def render(self, mode: str = 'human'):
        """
        Render the environment state.

        Args:
            mode (str): Render mode (currently only 'human' is supported).
        """
        print(f"Step: {self.current_step}, Balance: {self.balance}")
