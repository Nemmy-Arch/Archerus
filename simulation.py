# simulation.py
import time
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional

from tkinter import messagebox

from recommender import get_current_price_bitunix, get_decimal_places
from utils import logger

# Global rate limit variables
request_timestamps: deque = deque()
REQUEST_LIMIT: int = 10  # Maximum requests allowed within TIME_FRAME seconds
TIME_FRAME: int = 1      # Time frame in seconds


def enforce_rate_limit() -> None:
    """
    Enforce the rate limit by ensuring that no more than REQUEST_LIMIT requests
    are made within TIME_FRAME seconds. Waits if the limit is reached.
    """
    current_time = time.time()
    # Remove timestamps older than TIME_FRAME seconds
    while request_timestamps and request_timestamps[0] < current_time - TIME_FRAME:
        request_timestamps.popleft()
    if len(request_timestamps) >= REQUEST_LIMIT:
        wait_time = TIME_FRAME - (current_time - request_timestamps[0])
        logger.warning(f"⏳ Rate limit reached. Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
    request_timestamps.append(time.time())


class LiveSimulation:
    """
    A class to simulate live trading by executing trades based on decisions.

    Attributes:
        balance (float): The current simulated account balance.
        trades (list): A list of closed trades.
        open_trade (Optional[Dict[str, Any]]): The currently open trade, if any.
    """
    def __init__(self, initial_balance: float = 100000.0) -> None:
        self.balance: float = initial_balance
        self.trades: list = []
        self.open_trade: Optional[Dict[str, Any]] = None

    def simulate_trade(self, decision: Dict[str, Any], selected_symbol: str) -> Dict[str, Any]:
        """
        Simulate a trade based on the given decision and trading symbol.

        Args:
            decision (Dict[str, Any]): A dictionary with trade decision details, including
                                       'action' and 'current_price'.
            selected_symbol (str): The trading symbol for which to simulate the trade.

        Returns:
            Dict[str, Any]: A dictionary containing the simulated trade details.
        """
        current_price: float = decision["current_price"]

        # Enforce rate limit before making an API call for the exit price.
        enforce_rate_limit()
        exit_price: Optional[float] = get_current_price_bitunix(selected_symbol)
        if exit_price is None:
            exit_price = current_price

        timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine trade type and profit calculation based on decision action.
        if decision["action"] == "LONG":
            trade_type = "BUY"
            profit = (exit_price - current_price) / current_price * self.balance
        elif decision["action"] == "SHORT":
            trade_type = "SELL"
            profit = (current_price - exit_price) / current_price * self.balance
        else:
            trade_type = "HOLD"
            profit = 0.0

        # Open a new trade if none is active.
        if self.open_trade is None and trade_type != "HOLD":
            self.open_trade = {
                "action": trade_type,
                "entry": current_price,
                "timestamp": timestamp
            }
            logger.info(f"🚀 OPENED TRADE: {trade_type} at {current_price:.2f} on {timestamp}")
        # Close the active trade if a new non-hold signal is received.
        elif self.open_trade is not None and trade_type != "HOLD":
            entry_price = self.open_trade["entry"]
            closed_trade = {
                "action": trade_type,
                "entry": entry_price,
                "exit": exit_price,
                "profit": profit,
                "balance": self.balance + profit,
                "timestamp": timestamp
            }
            self.trades.append(closed_trade)
            self.balance += profit
            logger.info(
                f"CLOSED TRADE: {trade_type} | Entry: {entry_price:.2f}, Exit: {exit_price:.2f}, "
                f"Profit: {profit:.2f}, New Balance: {self.balance:.2f}"
            )
            self.open_trade = None

        return {
            "action": trade_type,
            "entry": current_price,
            "exit": exit_price,
            "profit": profit,
            "balance": self.balance,
            "timestamp": timestamp
        }