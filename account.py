# account.py
import json
import os
from collections import OrderedDict
from urllib.parse import urlencode
import time
from tkinter import messagebox
from typing import List, Union, Optional, Any

from utils import get_signed_headers, session, logger
# BASE_URL is not used in the current endpoints, so it's omitted here.

def get_trading_pairs() -> List[Any]:
    """
    Retrieve the list of trading pairs from the Bitunix API.

    Returns:
        List[Any]: A list of trading pairs data; returns an empty list on error.
    """
    url = "https://fapi.bitunix.com/api/v1/futures/market/trading_pairs"
    headers = get_signed_headers(request_path="/api/v1/futures/market/trading_pairs")
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error decoding trading pairs response: {e}")
        return []
    
    logger.info("DEBUG: Raw trading pairs response: " + str(data))
    if data.get("code") != 0:
        logger.error("Error retrieving trading pairs: " + data.get("msg", ""))
        return []
    
    return data.get("data", [])


def get_main_account_balance() -> Optional[float]:
    """
    Retrieve the main account balance for the margin coin USDT.

    Returns:
        Optional[float]: The available balance if retrieved successfully; otherwise, None.
    """
    params = {"marginCoin": "USDT"}
    # Ensure parameters are sorted for consistent signing.
    sorted_params = OrderedDict(sorted(params.items()))
    query_str_signature = urlencode(sorted_params)
    logger.info(f"🔍 Signing Request - Query String: {query_str_signature}")
    
    body = ""
    request_path = "/api/v1/futures/account"
    url = f"https://fapi.bitunix.com{request_path}?{query_str_signature}"
    
    headers = get_signed_headers(query_params=params, body=body, request_path=request_path)
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error decoding account response: {e}")
        return None

    logger.info(f"🔄 Raw API Response: {data}")
    
    if data.get("code") == 0:
        account_data = data.get("data")
        if isinstance(account_data, dict):
            available_balance = float(account_data.get("available", 0))
        elif isinstance(account_data, list) and len(account_data) > 0:
            available_balance = float(account_data[0].get("available", 0))
        else:
            available_balance = None
    else:
        available_balance = None

    if available_balance is None:
        logger.error("❌ Failed to retrieve account balance. Aborting auto trade.")
    
    return available_balance


def change_leverage(selected_symbol: str, desired_leverage: Union[int, float]) -> Optional[Any]:
    """
    Change the leverage for a given trading symbol.

    Args:
        selected_symbol (str): The trading symbol for which leverage is to be changed.
        desired_leverage (Union[int, float]): The desired leverage value.

    Returns:
        Optional[Any]: The API response data if successful, else None.
    """
    payload_leverage = OrderedDict([
        ("symbol", selected_symbol),
        ("leverage", desired_leverage),
        ("marginCoin", "USDT")
    ])
    # In this endpoint, query string is empty as parameters are passed in the body.
    query_str_leverage = ""
    body_leverage = json.dumps(payload_leverage, separators=(',', ':'))
    request_path = "/api/v1/futures/account/change_leverage"
    headers_leverage = get_signed_headers(query_params=query_str_leverage, body=body_leverage, request_path=request_path)
    url_leverage = "https://fapi.bitunix.com/api/v1/futures/account/change_leverage"
    
    try:
        response_leverage = session.post(url_leverage, data=body_leverage, headers=headers_leverage)
        response_leverage.raise_for_status()
        data_leverage = response_leverage.json()
    except Exception as e:
        logger.error("Error decoding leverage response: " + str(e))
        return None

    logger.info("Leverage Endpoint Response: " + str(data_leverage))
    return data_leverage