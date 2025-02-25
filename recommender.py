from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Tuple, Optional, Any, Dict

from utils import get_signed_headers, session, logger
from data import download_full_historical_data, resample_candlesticks
from indicators import (
    compute_crypto_indicators,
    calculate_support_resistance,
    fibonacci_levels,
    feature_cols
)
from config import BASE_TRAINING_TIMEFRAME, MIN_ATR_RATIO, TRADING_WINDOW

def get_current_price_bitunix(symbol: str) -> Optional[float]:
    """
    Retrieve the current ticker price for the given symbol from Bitunix API.
    """
    url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
    params = {"symbols": symbol}
    query_str = urlencode(params)
    headers = get_signed_headers(query_params=query_str, request_path="/api/v1/futures/market/tickers")
    
    try:
        response = session.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error decoding ticker data: {e}")
        return None

    if data.get("code") != 0:
        logger.error("Error retrieving ticker data: " + data.get("msg", ""))
        return None

    for ticker in data.get("data", []):
        if ticker.get("symbol") == symbol:
            try:
                return float(ticker.get("lastPrice"))
            except Exception as e:
                logger.error("Error parsing lastPrice: " + str(e))
                return None
    return None

def get_decimal_places(price: float) -> int:
    """
    Determine the number of decimal places based on price magnitude.
    """
    if price >= 1000:
        return 1
    elif price >= 100:
        return 2
    elif price >= 10:
        return 3
    elif price >= 1:
        return 4
    else:
        return 5

def get_recommendation_message(model: Any, feature_columns: list, symbol: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Generate a live trading recommendation message along with key decision metrics.
    Improved for consistency, dynamic thresholding, and better logging.
    """
    # Capture a single reference timestamp for consistency
    current_time = datetime.now()
    current_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine live window parameters
    min_required_candles = 300
    live_window_hours = max(75, (TRADING_WINDOW * min_required_candles) / 60)
    live_start_date = (current_time - timedelta(hours=live_window_hours)).strftime("%Y-%m-%d %H:%M:%S")
    live_end_date = current_timestamp
    
    # Download and validate live data
    live_df = download_full_historical_data(symbol, live_start_date, live_end_date, interval=f"{BASE_TRAINING_TIMEFRAME}m")
    if live_df.empty or len(live_df) < 30:
        logger.error("Insufficient live data downloaded.")
        return ("Error: Not enough live data downloaded.", None)
    
    live_df = resample_candlesticks(live_df, TRADING_WINDOW)
    live_df = compute_crypto_indicators(live_df)
    if live_df.empty or len(live_df) < 30:
        logger.error("Insufficient data after resampling or computing indicators.")
        return ("Error: Not enough resampled data after computing indicators.", None)
    
    # Compute support, resistance, and Fibonacci levels
    support, resistance = calculate_support_resistance(live_df, window=20)
    fib_382, fib_618 = fibonacci_levels(support, resistance)
    
    latest_data = live_df.iloc[-1]
    
    # --- Enhanced Market Regime Detection ---
    # Extract new indicator values from the latest data (default to 0 if missing)
    macd = latest_data.get('macd', 0)
    macd_signal = latest_data.get('macd_signal', 0)
    adx = latest_data.get('adx', 0)
    stoch_k = latest_data.get('stoch_k', 0)
    stoch_d = latest_data.get('stoch_d', 0)
    cci = latest_data.get('cci', 0)
    
    bullish_score = 0
    bearish_score = 0
    
    # SMA condition
    if latest_data['sma_50'] > latest_data['sma_200']:
        bullish_score += 1
    else:
        bearish_score += 1

    # MACD condition
    if macd > macd_signal:
        bullish_score += 1
    else:
        bearish_score += 1

    # ADX condition: If ADX indicates a strong trend (>25), it supports the current directional bias.
    if adx > 25:
        bullish_score += 1
        bearish_score += 1  # Alternatively, you can refine this further based on directional moves.

    # Stochastic oscillator condition
    if stoch_k > stoch_d:
        bullish_score += 1
    else:
        bearish_score += 1

    # CCI condition: Above +100 suggests bullish momentum; below -100, bearish.
    if cci > 100:
        bullish_score += 1
    elif cci < -100:
        bearish_score += 1

    # Composite regime decision based on the cumulative score
    regime = "Bullish" if bullish_score > bearish_score else "Bearish"
    
    # Prepare observation data and ensure required features are available.
    try:
        obs = latest_data[feature_columns].values.astype("float32").reshape(1, -1)
        logger.info(f"Observation shape: {obs.shape}")
    except Exception as e:
        logger.error(f"Error preparing observation: {e}")
        return ("Error preparing observation data.", None)
    
    # Predict action using the model; log borderline prediction cases for review.
    action, _ = model.predict(obs, deterministic=True)
    logger.info(f"Model predicted action value: {action[0]}")
    
    # Dynamic decision based on model output and enhanced market regime signal
    if action[0] > 0.1:
        recommendation = "LONG" if regime == "Bullish" else "HOLD"
    elif action[0] < -0.1:
        recommendation = "SHORT" if regime == "Bearish" else "HOLD"
    else:
        recommendation = "HOLD"
    
    # Price reconciliation between ticker and resampled data
    resampled_price = float(latest_data['Close'])
    ticker_price = get_current_price_bitunix(symbol)
    if ticker_price is None:
        ticker_price = resampled_price
    price_diff = abs(ticker_price - resampled_price) / ticker_price
    if price_diff > 0.05:
        current_price = ticker_price
        logger.info(f"Significant price discrepancy detected ({price_diff*100:.2f}%), using ticker price.")
    else:
        current_price = resampled_price

    dp = get_decimal_places(current_price)
    
    # Ensure ATR value is realistic; log if minimum is applied.
    atr_value = float(latest_data['atr'])
    min_atr_val = MIN_ATR_RATIO * current_price
    if atr_value < min_atr_val:
        logger.info(f"ATR value ({atr_value}) below minimum threshold ({min_atr_val}). Using minimum ATR.")
        atr_value = min_atr_val

    # Define dynamic multipliers (could be optimized further via backtesting)
    tp_multiplier = 1.5  
    sl_multiplier = 1.0

    if recommendation == "LONG":
        raw_tp = current_price + (tp_multiplier * atr_value)
        raw_sl = current_price - (sl_multiplier * atr_value)
    elif recommendation == "SHORT":
        raw_tp = current_price - (tp_multiplier * atr_value)
        raw_sl = current_price + (sl_multiplier * atr_value)
    else:
        raw_tp = raw_sl = None

    # Adjust risk levels using support/resistance and Fibonacci levels
    buffer = 0.005 * current_price
    if recommendation == "LONG":
        adjusted_sl = raw_sl if raw_sl >= support else support + buffer
        adjusted_tp = raw_tp
        if raw_tp > current_price:
            if fib_382 > current_price and (raw_tp - fib_382) < 0.5 * (raw_tp - current_price):
                adjusted_tp = fib_382
            if fib_618 > current_price and (raw_tp - fib_618) < 0.5 * (raw_tp - current_price):
                adjusted_tp = min(adjusted_tp, fib_618)
    elif recommendation == "SHORT":
        adjusted_sl = raw_sl if raw_sl <= resistance else resistance - buffer
        adjusted_tp = raw_tp
        if raw_tp < current_price:
            if fib_382 < current_price and (fib_382 - raw_tp) < 0.5 * (current_price - raw_tp):
                adjusted_tp = fib_382
            if fib_618 < current_price and (fib_618 - raw_tp) < 0.5 * (current_price - raw_tp):
                adjusted_tp = max(adjusted_tp, fib_618)
    else:
        adjusted_sl = adjusted_tp = None

    # Build the recommendation message
    recommendation_message = "\n--- Live Position Recommendation ---\n"
    recommendation_message += f"Live Price (resampled): {current_price:.{dp}f}\n"
    recommendation_message += f"Live Price (ticker): {ticker_price:.{dp}f}\n"
    recommendation_message += f"Market Regime: {regime}\n"
    recommendation_message += f"Predicted Action: {recommendation}\n"
    recommendation_message += f"Support: {support:.{dp}f}, Resistance: {resistance:.{dp}f}\n"
    recommendation_message += f"Fibonacci Levels: 38.2% -> {fib_382:.{dp}f}, 61.8% -> {fib_618:.{dp}f}\n"
    
    if recommendation != "HOLD":
        recommendation_message += f"Initial ATR-based TP: {raw_tp:.{dp}f}\n"
        recommendation_message += f"Initial ATR-based SL: {raw_sl:.{dp}f}\n"
        recommendation_message += f"Adjusted TP: {adjusted_tp:.{dp}f}\n"
        recommendation_message += f"Adjusted SL: {adjusted_sl:.{dp}f}\n"
    else:
        recommendation_message += "No trade recommended (HOLD).\n"
    
    decision = {
        "action": recommendation,
        "current_price": current_price,
        "raw_tp": raw_tp,
        "raw_sl": raw_sl,
        "adjusted_tp": adjusted_tp,
        "adjusted_sl": adjusted_sl,
        "market_regime": regime,
        "timestamp": current_timestamp
    }
    
    return recommendation_message, decision
