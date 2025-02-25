# indicators.py
import pandas as pd
import numpy as np
from typing import Tuple
from utils import logger

# Updated feature columns including new indicators for training.
feature_cols = [
    'Open', 'High', 'Low', 'Close',
    'sma_50', 'sma_200', 'atr', 'rsi', 'bb_middle', 'bb_upper', 'bb_lower',
    'cci', 'obv', 'adx',
    'macd', 'macd_signal', 'macd_hist',
    'stoch_k', 'stoch_d',
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b'
]

def compute_crypto_indicators(
    df: pd.DataFrame, 
    sma_short: int = 50, 
    sma_long: int = 200,
    atr_window: int = 14, 
    rsi_window: int = 14, 
    bb_window: int = 20, 
    bb_std_multiplier: int = 2,
    cci_period: int = 20,
    adx_period: int = 14,
    stoch_period: int = 14,
    stoch_smooth: int = 3
) -> pd.DataFrame:
    """
    Compute technical indicators for cryptocurrency data.

    Indicators calculated:
      - SMA: 50 and 200 periods.
      - ATR: Average True Range.
      - RSI: Relative Strength Index.
      - Bollinger Bands.
      - CCI: Commodity Channel Index.
      - OBV: On Balance Volume.
      - ADX: Average Directional Index.
      - MACD: with signal and histogram.
      - Stochastic Oscillator: %K and %D.
      - Ichimoku Cloud: Tenkan-sen, Kijun-sen, Senkou Span A and B.
    """
    if len(df) < max(sma_long, atr_window, rsi_window, bb_window, cci_period, adx_period, stoch_period, 52):
        logger.warning("Not enough data points to compute reliable indicators.")

    # --- Price-based Indicators ---
    df['sma_50'] = df['Close'].rolling(window=sma_short, min_periods=1).mean()
    df['sma_200'] = df['Close'].rolling(window=sma_long, min_periods=1).mean()

    # --- ATR Calculation ---
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=atr_window, min_periods=1).mean()

    # --- RSI Calculation ---
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_window, min_periods=rsi_window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_window, min_periods=rsi_window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    df['bb_middle'] = df['Close'].rolling(window=bb_window, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=bb_window, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + bb_std_multiplier * bb_std
    df['bb_lower'] = df['bb_middle'] - bb_std_multiplier * bb_std

    # --- CCI (Commodity Channel Index) ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=cci_period, min_periods=cci_period).mean()
    mad = tp.rolling(window=cci_period, min_periods=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (tp - sma_tp) / (0.015 * mad)

    # --- OBV (On Balance Volume) ---
    df['obv'] = 0
    df['obv'] = np.where(df['Close'] > df['Close'].shift(), df['Volume'],
                         np.where(df['Close'] < df['Close'].shift(), -df['Volume'], 0))
    df['obv'] = df['obv'].cumsum()

    # --- ADX (Average Directional Index) ---
    # Calculate directional movements
    df['up_move'] = df['High'] - df['High'].shift()
    df['down_move'] = df['Low'].shift() - df['Low']
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    # Smooth the TR, +dm, and -dm using rolling sum over the adx_period
    tr_smooth = tr.rolling(window=adx_period, min_periods=adx_period).sum()
    plus_dm_smooth = df['+dm'].rolling(window=adx_period, min_periods=adx_period).sum()
    minus_dm_smooth = df['-dm'].rolling(window=adx_period, min_periods=adx_period).sum()
    df['+di'] = 100 * (plus_dm_smooth / tr_smooth)
    df['-di'] = 100 * (minus_dm_smooth / tr_smooth)
    df['dx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'] + 1e-8))
    df['adx'] = df['dx'].rolling(window=adx_period, min_periods=adx_period).mean()

    # --- MACD Calculation ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Stochastic Oscillator ---
    low_min = df['Low'].rolling(window=stoch_period, min_periods=1).min()
    high_max = df['High'].rolling(window=stoch_period, min_periods=1).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-8))
    df['stoch_d'] = df['stoch_k'].rolling(window=stoch_smooth, min_periods=1).mean()

    # --- Ichimoku Cloud ---
    # Tenkan-sen (Conversion Line)
    period9_high = df['High'].rolling(window=9, min_periods=1).max()
    period9_low = df['Low'].rolling(window=9, min_periods=1).min()
    df['tenkan_sen'] = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line)
    period26_high = df['High'].rolling(window=26, min_periods=1).max()
    period26_low = df['Low'].rolling(window=26, min_periods=1).min()
    df['kijun_sen'] = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    period52_high = df['High'].rolling(window=52, min_periods=1).max()
    period52_low = df['Low'].rolling(window=52, min_periods=1).min()
    df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Ensure all required feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Drop intermediate calculation columns if desired
    drop_cols = ['up_move', 'down_move', '+dm', '-dm', '+di', '-di', 'dx']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    recent_data = df.tail(window)
    support = recent_data['Low'].min()
    resistance = recent_data['High'].max()
    return support, resistance

def fibonacci_levels(support: float, resistance: float) -> Tuple[float, float]:
    diff = resistance - support
    fib_382 = resistance - 0.382 * diff
    fib_618 = resistance - 0.618 * diff
    return fib_382, fib_618
