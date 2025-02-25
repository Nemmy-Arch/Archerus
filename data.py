# data.py
import os
import time
import pickle
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Optional

from utils import get_signed_headers, session, logger
from config import BASE_TRAINING_TIMEFRAME, HIST_CACHE_DIR

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string into a datetime object.

    Supports formats: "%Y-%m-%d %H:%M:%S" and "%Y-%m-%d".

    Args:
        date_str (str): Date string to parse.

    Returns:
        datetime: Parsed datetime object.

    Raises:
        ValueError: If the date string doesn't match supported formats.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Date string does not match supported formats.")

def download_kline_data(symbol: str, start_date: str, end_date: str, interval: Optional[str] = None) -> pd.DataFrame:
    """
    Download kline (candlestick) data from the API.

    Args:
        symbol (str): Trading symbol.
        start_date (str): Start date as string.
        end_date (str): End date as string.
        interval (Optional[str]): Time interval. Defaults to BASE_TRAINING_TIMEFRAME.

    Returns:
        pd.DataFrame: DataFrame containing the kline data, or an empty DataFrame on error.
    """
    url = "https://fapi.bitunix.com/api/v1/futures/market/kline"
    if interval is None:
        interval = f"{BASE_TRAINING_TIMEFRAME}m"

    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    params = {
        "symbol": symbol,
        "startTime": int(start_dt.timestamp() * 1000),
        "endTime": int(end_dt.timestamp() * 1000),
        "interval": interval
    }
    query_str = urlencode(params)
    headers = get_signed_headers(query_params=query_str, request_path="/api/v1/futures/market/kline")
    
    try:
        response = session.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching kline data: {e}")
        return pd.DataFrame()

    if data.get("code") != 0:
        logger.error("Error downloading kline data: " + data.get("msg", ""))
        return pd.DataFrame()
    
    kline_list = data.get("data", [])
    if not kline_list:
        logger.error("No kline data returned.")
        return pd.DataFrame()
    
    df = pd.DataFrame(kline_list)
    try:
        df["time"] = pd.to_numeric(df["time"])
        df["Date"] = pd.to_datetime(df["time"], unit="ms")
    except Exception as e:
        logger.error(f"Error processing timestamps: {e}")
        return pd.DataFrame()
    
    df.set_index("Date", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "quoteVol": "Volume"}, inplace=True)
    
    try:
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception as e:
        logger.error(f"Error converting data types: {e}")
        return pd.DataFrame()

    return df

def save_historical_cache(symbol: str, df: pd.DataFrame) -> None:
    """
    Save historical data to a cache file.

    Args:
        symbol (str): Trading symbol.
        df (pd.DataFrame): DataFrame to be cached.
    """
    cache_file = os.path.join(HIST_CACHE_DIR, f"{symbol}_history.pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        logger.info(f"Saved historical cache for {symbol}")
    except Exception as e:
        logger.error(f"Error saving cache for {symbol}: {e}")

def load_historical_cache(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load cached historical data if available.

    Args:
        symbol (str): Trading symbol.

    Returns:
        Optional[pd.DataFrame]: Cached DataFrame if found; otherwise, None.
    """
    cache_file = os.path.join(HIST_CACHE_DIR, f"{symbol}_history.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            logger.info(f"Loaded cached data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
    return None

def download_full_historical_data(symbol: str, start_date_str: str, end_date_str: str, interval: Optional[str] = None, batch_minutes: int = 1440) -> pd.DataFrame:
    """
    Download full historical data in batches and cache the result.

    Args:
        symbol (str): Trading symbol.
        start_date_str (str): Start date as a string.
        end_date_str (str): End date as a string.
        interval (Optional[str]): Time interval. Defaults to BASE_TRAINING_TIMEFRAME.
        batch_minutes (int): Batch size in minutes. Defaults to 1440 (1 day).

    Returns:
        pd.DataFrame: Concatenated historical data, or an empty DataFrame if no data is retrieved.
    """
    cached = load_historical_cache(symbol)
    if cached is not None:
        return cached

    if interval is None:
        interval = f"{BASE_TRAINING_TIMEFRAME}m"
    
    start_dt = parse_date(start_date_str)
    end_dt = parse_date(end_date_str)
    
    all_data = []
    current_start = start_dt
    batch_delta = timedelta(minutes=batch_minutes)
    
    while current_start < end_dt:
        current_end = min(current_start + batch_delta, end_dt)
        logger.info(f"Downloading data from {current_start} to {current_end}")
        df_batch = download_kline_data(
            symbol, 
            current_start.strftime("%Y-%m-%d %H:%M:%S"), 
            current_end.strftime("%Y-%m-%d %H:%M:%S"), 
            interval=interval
        )
        if not df_batch.empty:
            all_data.append(df_batch)
        else:
            logger.warning(f"No data returned for period {current_start} to {current_end}")
        time.sleep(0.5)
        current_start = current_end
    
    if all_data:
        full_df = pd.concat(all_data)
        full_df.sort_index(inplace=True)
        save_historical_cache(symbol, full_df)
        return full_df
    else:
        return pd.DataFrame()

def resample_candlesticks(df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """
    Resample candlestick data to a new time window.

    Args:
        df (pd.DataFrame): DataFrame with original candlestick data.
        window_minutes (int): The resampling window in minutes.

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    logger.info(f"Before resampling: {len(df)} candles")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('1min', method='pad')
    
    df_resampled = df.resample(f'{window_minutes}min', label='left', closed='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    df_resampled.dropna(inplace=True)
    logger.info(f"After resampling: {len(df_resampled)} candles")
    return df_resampled