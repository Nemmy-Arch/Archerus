import os
import time
import json
import hmac
import string
import logging
import hashlib
from collections import OrderedDict
from urllib.parse import urlencode, parse_qsl
from typing import Union, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import secrets

from config import API_KEY, SECRET_KEY, PUBLIC_HEADERS

# Setup logger for the TradingApp (avoid logging sensitive data)
logger = logging.getLogger("TradingApp")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def generate_nonce(length: int = 32) -> str:
    """
    Generate a secure random nonce.
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def get_signed_headers(
    query_params: Union[str, Dict[str, Any]] = "",
    body: str = "",
    request_path: str = ""
) -> Dict[str, str]:
    """
    Generate signed headers for API requests.
    """
    nonce = generate_nonce()
    timestamp = str(int(time.time() * 1000))
    if isinstance(query_params, str) and query_params:
        query_params = dict(parse_qsl(query_params))
    sorted_params = OrderedDict(sorted(query_params.items())) if query_params else OrderedDict()
    query_string = urlencode(sorted_params)
    digest_input = f"{nonce}{timestamp}{API_KEY}{request_path}{query_string}{body}"
    inner_hash = hashlib.sha256(digest_input.encode('utf-8')).hexdigest()
    sign = hmac.new(SECRET_KEY.encode('utf-8'), inner_hash.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        "api-key": API_KEY,
        "nonce": nonce,
        "time": timestamp,
        "sign": sign,
        "language": "en-US",
        "Content-Type": "application/json"
    }
    return headers

# Custom session that applies a default timeout to every request
class TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10  # default timeout in seconds
        return super().request(*args, **kwargs)

def get_session() -> requests.Session:
    """
    Creates a session with retry logic and a default timeout.
    """
    session = TimeoutSession()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Global session instance
session = get_session()
